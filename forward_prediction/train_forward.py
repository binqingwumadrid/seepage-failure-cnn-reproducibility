"""Train and evaluate the reproducible forward parameter-to-field CNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from forward_model import (
    CompositeImageLoss,
    FieldImageDataset,
    ForwardDecoder,
    ParameterBounds,
    model_parameter_count,
    per_sample_metrics,
    tensor_to_image,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STRAIN_DATA = SCRIPT_DIR.parent / "generated_data" / "equivalent_plastic_strain" / "noise_0pct"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "equivalent_plastic_strain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_STRAIN_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--field", choices=("equivalent_plastic_strain", "pressure"), default="equivalent_plastic_strain")
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit it to generate a fresh seed for every run.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Run test-set validation every N epochs (also runs at epochs 1 and final).",
    )
    parser.add_argument(
        "--cache-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache resized images in RAM as uint8 for faster training.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-warmup", type=int, default=20)
    parser.add_argument("--inference-repeats", type=int, default=10)
    parser.add_argument("--comparison-count", type=int, default=8)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The split, initialization, and data-loader order are deterministic. We
    # keep cuDNN's fast kernels enabled because forcing deterministic
    # transposed-convolution kernels makes the RTX 5060 run many times slower.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available in PyTorch.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite-output to replace this run."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(
    model: ForwardDecoder,
    loader: DataLoader,
    criterion: CompositeImageLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mse": 0.0, "ssim": 0.0, "gradient": 0.0}
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            parameters = batch["parameters"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(parameters)
            components = criterion.components(prediction, images)
            if training:
                components["loss"].backward()
                optimizer.step()

            batch_size = images.shape[0]
            sample_count += batch_size
            for name in totals:
                totals[name] += float(components[name].detach()) * batch_size

    return {name: value / sample_count for name, value in totals.items()}


def evaluate_samples(
    model: ForwardDecoder,
    loader: DataLoader,
    device: torch.device,
    comparison_dir: Path,
    comparison_count: int,
) -> tuple[List[Dict[str, object]], Dict[str, float]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    all_targets: List[torch.Tensor] = []
    all_predictions: List[torch.Tensor] = []
    comparison_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            parameters = batch["parameters"].to(device)
            predictions = model(parameters)
            metrics = per_sample_metrics(predictions, images)
            all_targets.append(images.cpu())
            all_predictions.append(predictions.cpu())

            for index, filename in enumerate(batch["filename"]):
                raw = batch["raw_parameters"][index].tolist()
                row = {
                    "filename": filename,
                    "cohesion_pa": raw[0],
                    "friction_angle_deg": raw[1],
                    "log10_ks": raw[2],
                    "ks_m_per_s": 10.0 ** raw[2],
                    "mse": float(metrics["mse"][index]),
                    "mae": float(metrics["mae"][index]),
                    "r2": float(metrics["r2"][index]),
                    "ssim": float(metrics["ssim"][index]),
                    "gradient_loss": float(metrics["gradient"][index]),
                }
                rows.append(row)

                if len(rows) <= comparison_count:
                    true_image = tensor_to_image(images[index])
                    predicted_image = tensor_to_image(predictions[index])
                    canvas = Image.new("RGB", (true_image.width * 2, true_image.height), "white")
                    canvas.paste(true_image, (0, 0))
                    canvas.paste(predicted_image, (true_image.width, 0))
                    canvas.save(comparison_dir / f"{Path(filename).stem}_true-left_pred-right.png")

    targets = torch.cat(all_targets).reshape(-1).double()
    predictions = torch.cat(all_predictions).reshape(-1).double()
    residual_sum = (targets - predictions).square().sum()
    total_sum = (targets - targets.mean()).square().sum()
    global_r2 = float(1.0 - residual_sum / total_sum.clamp_min(1e-12))
    aggregate = {
        "mean_mse": mean(float(row["mse"]) for row in rows),
        "mean_mae": mean(float(row["mae"]) for row in rows),
        "mean_sample_r2": mean(float(row["r2"]) for row in rows),
        "global_pixel_r2": global_r2,
        "mean_ssim": mean(float(row["ssim"]) for row in rows),
        "mean_gradient_loss": mean(float(row["gradient_loss"]) for row in rows),
    }
    return rows, aggregate


def benchmark_inference(
    model: ForwardDecoder,
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    parameters = [batch["parameters"].to(device) for batch in loader]
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(parameters[0])
        synchronize(device)
        start = time.perf_counter()
        total_samples = 0
        for _ in range(repeats):
            for batch_parameters in parameters:
                model(batch_parameters)
                total_samples += batch_parameters.shape[0]
        synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "repeats": repeats,
        "total_timed_samples": total_samples,
        "total_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / total_samples,
        "samples_per_second": total_samples / elapsed,
    }


def main() -> None:
    args = parse_args()
    if min(args.train_size, args.test_size, args.epochs, args.batch_size, args.eval_every) <= 0:
        raise ValueError("Train size, test size, epochs, batch size, and eval-every must be positive.")
    actual_seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    set_reproducible_seed(actual_seed)
    device = select_device(args.device)
    output_dir = prepare_output(args.output_dir, args.overwrite_output)

    bounds = ParameterBounds()
    dataset = FieldImageDataset(
        args.data_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        bounds=bounds,
        cache_images=args.cache_images,
    )
    requested = args.train_size + args.test_size
    if requested > len(dataset):
        raise ValueError(
            f"Dataset contains {len(dataset)} images, but train-size + test-size = {requested}."
        )

    generator = torch.Generator().manual_seed(actual_seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    test_indices = permutation[: args.test_size]
    train_indices = permutation[args.test_size : requested]
    unused_indices = permutation[requested:]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    split_rows: List[Dict[str, object]] = []
    for split, indices in (("train", train_indices), ("test", test_indices), ("unused", unused_indices)):
        for index in indices:
            c_value, phi_value, logk_value = dataset.raw_parameters[index]
            split_rows.append(
                {
                    "split": split,
                    "filename": dataset.paths[index].name,
                    "cohesion_pa": c_value,
                    "friction_angle_deg": phi_value,
                    "log10_ks": logk_value,
                    "ks_m_per_s": 10.0**logk_value,
                }
            )
    save_csv(
        output_dir / "data_split.csv",
        split_rows,
        ["split", "filename", "cohesion_pa", "friction_angle_deg", "log10_ks", "ks_m_per_s"],
    )

    loader_generator = torch.Generator().manual_seed(actual_seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=loader_generator, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    inference_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ForwardDecoder(args.image_height, args.image_width).to(device)
    criterion = CompositeImageLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_config = {
        "field": args.field,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "inference_batch_size": 1,
        "learning_rate": args.learning_rate,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "latent_shape": [256, args.image_height // 8, args.image_width // 8],
        "seed": actual_seed,
        "seed_mode": "user_supplied" if args.seed is not None else "random_per_run",
        "evaluation_epochs": sorted(set([1, args.epochs, *range(args.eval_every, args.epochs + 1, args.eval_every)])),
        "cache_images": args.cache_images,
        "deterministic_cuda_kernels": False,
        "parameter_bounds": bounds.as_dict(),
        "loss_weights": {"mse": 1.0, "ssim_loss": 0.2, "gradient": 0.1},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_parameters": model_parameter_count(model),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(json.dumps(run_config, indent=2))
    history: List[Dict[str, object]] = []
    best_test_loss = float("inf")
    best_checkpoint: Dict[str, object] | None = None
    synchronize(device)
    stage_start = time.perf_counter()
    pure_training_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        train_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        synchronize(device)
        pure_training_seconds += time.perf_counter() - train_start
        row: Dict[str, object] = {"epoch": epoch}
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        should_evaluate = epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0
        test_metrics = (
            run_epoch(model, test_loader, criterion, device, optimizer=None)
            if should_evaluate
            else None
        )
        row.update(
            {
                f"test_{name}": test_metrics[name] if test_metrics is not None else None
                for name in ("loss", "mse", "ssim", "gradient")
            }
        )
        history.append(row)

        checkpoint: Dict[str, object] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "test_loss": test_metrics["loss"] if test_metrics is not None else None,
            "config": run_config,
        }
        if epoch == args.epochs:
            torch.save(checkpoint, output_dir / "last_model.pt")
        if test_metrics is not None and test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            best_checkpoint = {
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "epoch": epoch,
                "test_loss": test_metrics["loss"],
                "config": run_config,
            }

        message = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.6f}, mse={train_metrics['mse']:.6f}, "
            f"ssim={train_metrics['ssim']:.6f}, grad={train_metrics['gradient']:.6f}"
        )
        if test_metrics is not None:
            message += (
                f" | test loss={test_metrics['loss']:.6f}, mse={test_metrics['mse']:.6f}, "
                f"ssim={test_metrics['ssim']:.6f}, grad={test_metrics['gradient']:.6f}"
            )
        print(message)

    synchronize(device)
    training_stage_seconds = time.perf_counter() - stage_start
    history_fields = list(history[0].keys())
    save_csv(output_dir / "training_history.csv", history, history_fields)
    if best_checkpoint is None:
        raise RuntimeError("Training ended without producing a best checkpoint.")
    torch.save(best_checkpoint, output_dir / "best_model.pt")

    try:
        final_checkpoint = torch.load(output_dir / "last_model.pt", map_location=device, weights_only=False)
    except TypeError:
        final_checkpoint = torch.load(output_dir / "last_model.pt", map_location=device)
    model.load_state_dict(final_checkpoint["model_state_dict"])
    sample_rows, test_summary = evaluate_samples(
        model,
        test_loader,
        device,
        output_dir / "comparisons",
        args.comparison_count,
    )
    save_csv(output_dir / "test_sample_metrics.csv", sample_rows, list(sample_rows[0].keys()))
    inference = benchmark_inference(
        model, inference_loader, device, args.inference_warmup, args.inference_repeats
    )
    summary = {
        "evaluated_checkpoint": "last_model.pt",
        "evaluated_epoch": final_checkpoint["epoch"],
        "best_epoch": best_checkpoint["epoch"],
        "best_test_composite_loss": best_checkpoint["test_loss"],
        "test_metrics": test_summary,
        "timing": {
            "pure_training_seconds": pure_training_seconds,
            "training_and_validation_seconds": training_stage_seconds,
            "inference": inference,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

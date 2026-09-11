"""Train and evaluate the archived CNN for field-image parameter inversion."""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from inverse_model import (
    InverseImageDataset,
    InverseRegressor,
    ParameterBounds,
    model_parameter_count,
    parameter_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "generated_data" / "equivalent_plastic_strain" / "noise_3pct"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "equivalent_plastic_strain_noise_3pct"
PARAMETER_NAMES = ("cohesion_pa", "friction_angle_deg", "log10_ks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None, help="Omit for a fresh random seed per run.")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-warmup", type=int, default=20)
    parser.add_argument("--inference-repeats", type=int, default=10)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available in PyTorch.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def save_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(
    model: InverseRegressor,
    loader: DataLoader,
    bounds: ParameterBounds,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    squared_norm = torch.zeros(3, dtype=torch.float64)
    squared_raw = torch.zeros(3, dtype=torch.float64)
    absolute_raw = torch.zeros(3, dtype=torch.float64)
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["parameters"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = F.mse_loss(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
            errors_norm = predictions.detach() - targets
            errors_raw = bounds.denormalize(predictions.detach()) - bounds.denormalize(targets)
            squared_norm += errors_norm.square().sum(dim=0).cpu().double()
            squared_raw += errors_raw.square().sum(dim=0).cpu().double()
            absolute_raw += errors_raw.abs().sum(dim=0).cpu().double()
            count += images.shape[0]
    mse_norm = squared_norm / count
    mse_raw = squared_raw / count
    mae_raw = absolute_raw / count
    return {
        "mse_norm": float(mse_norm.mean()),
        "mse_norm_per_parameter": mse_norm.tolist(),
        "mse_raw_per_parameter": mse_raw.tolist(),
        "mae_raw_per_parameter": mae_raw.tolist(),
    }


def evaluate_samples(
    model: InverseRegressor,
    loader: DataLoader,
    bounds: ParameterBounds,
    device: torch.device,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    predictions_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target_raw = batch["raw_parameters"].to(device)
            prediction_raw = bounds.denormalize(model(images))
            predictions_all.append(prediction_raw.cpu())
            targets_all.append(target_raw.cpu())
            for index, filename in enumerate(batch["filename"]):
                target = target_raw[index].tolist()
                prediction = prediction_raw[index].tolist()
                row: Dict[str, object] = {"filename": filename}
                for position, name in enumerate(PARAMETER_NAMES):
                    row[f"true_{name}"] = target[position]
                    row[f"pred_{name}"] = prediction[position]
                    row[f"error_{name}"] = prediction[position] - target[position]
                    row[f"absolute_error_{name}"] = abs(prediction[position] - target[position])
                rows.append(row)
    predictions = torch.cat(predictions_all).double()
    targets = torch.cat(targets_all).double()
    values = parameter_metrics(predictions, targets)
    summary = {
        metric: {name: float(tensor[index]) for index, name in enumerate(PARAMETER_NAMES)}
        for metric, tensor in values.items()
    }
    return rows, summary


def benchmark(model: InverseRegressor, loader: DataLoader, device: torch.device, warmup: int, repeats: int) -> Dict[str, float]:
    batches = [batch["image"].to(device) for batch in loader]
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(batches[0])
        synchronize(device)
        start = time.perf_counter()
        count = 0
        for _ in range(repeats):
            for images in batches:
                model(images)
                count += images.shape[0]
        synchronize(device)
    elapsed = time.perf_counter() - start
    return {"milliseconds_per_sample": 1000.0 * elapsed / count, "samples_per_second": count / elapsed}


def main() -> None:
    args = parse_args()
    if min(args.train_size, args.test_size, args.epochs, args.batch_size, args.eval_every) <= 0:
        raise ValueError("Sizes, epochs, batch size, and eval-every must be positive.")
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    set_seed(seed)
    device = select_device(args.device)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite_output:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    bounds = ParameterBounds()
    dataset = InverseImageDataset(
        args.data_dir, args.image_height, args.image_width, bounds, cache_images=args.cache_images
    )
    requested = args.train_size + args.test_size
    if requested > len(dataset):
        raise ValueError(
            f"Found {len(dataset)} images, but train-size + test-size is {requested}."
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    test_indices = order[: args.test_size]
    train_indices = order[args.test_size : requested]
    unused_indices = order[requested:]
    split_rows: List[Dict[str, object]] = []
    for split, indices in (("train", train_indices), ("test", test_indices), ("unused", unused_indices)):
        for index in indices:
            raw = dataset.raw_parameters[index]
            split_rows.append({"split": split, "filename": dataset.paths[index].name, **dict(zip(PARAMETER_NAMES, raw))})
    save_csv(output / "data_split.csv", split_rows, ["split", "filename", *PARAMETER_NAMES])

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True,
        generator=torch.Generator().manual_seed(seed), **loader_options
    )
    test_loader = DataLoader(Subset(dataset, test_indices), shuffle=False, **loader_options)
    inference_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    model = InverseRegressor(args.image_height, args.image_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    evaluation_epochs = sorted(set([1, args.epochs, *range(args.eval_every, args.epochs + 1, args.eval_every)]))
    config = {
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "output_dir": str(output),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "parameter_bounds": bounds.as_dict(),
        "seed": seed,
        "seed_mode": "user_supplied" if args.seed is not None else "random_per_run",
        "evaluation_epochs": evaluation_epochs,
        "cache_images": args.cache_images,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_parameters": model_parameter_count(model),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    history: List[Dict[str, object]] = []
    best_loss = float("inf")
    best_checkpoint: Dict[str, object] | None = None
    pure_training_seconds = 0.0
    synchronize(device)
    stage_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, bounds, device, optimizer)
        synchronize(device)
        pure_training_seconds += time.perf_counter() - epoch_start
        test_metrics = run_epoch(model, test_loader, bounds, device, None) if epoch in evaluation_epochs else None
        row: Dict[str, object] = {"epoch": epoch, "train_mse_norm": train_metrics["mse_norm"]}
        for index, name in enumerate(PARAMETER_NAMES):
            row[f"train_mse_raw_{name}"] = train_metrics["mse_raw_per_parameter"][index]
        row["test_mse_norm"] = test_metrics["mse_norm"] if test_metrics else None
        for index, name in enumerate(PARAMETER_NAMES):
            row[f"test_mse_raw_{name}"] = test_metrics["mse_raw_per_parameter"][index] if test_metrics else None
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config}
        if epoch == args.epochs:
            torch.save(checkpoint, output / "last_model.pt")
        if test_metrics is not None and test_metrics["mse_norm"] < best_loss:
            best_loss = float(test_metrics["mse_norm"])
            best_checkpoint = {
                "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "epoch": epoch,
                "test_mse_norm": best_loss,
                "config": config,
            }
        message = f"Epoch {epoch:03d}/{args.epochs} | train normalized MSE={train_metrics['mse_norm']:.8f}"
        if test_metrics:
            message += f" | test normalized MSE={test_metrics['mse_norm']:.8f}"
        print(message, flush=True)

    synchronize(device)
    stage_seconds = time.perf_counter() - stage_start
    save_csv(output / "training_history.csv", history, list(history[0].keys()))
    if best_checkpoint is None:
        raise RuntimeError("No validation checkpoint was produced.")
    torch.save(best_checkpoint, output / "best_model.pt")
    rows, metrics = evaluate_samples(model, test_loader, bounds, device)
    save_csv(output / "test_predictions.csv", rows, list(rows[0].keys()))
    summary = {
        "evaluated_checkpoint": "last_model.pt",
        "evaluated_epoch": args.epochs,
        "best_epoch": best_checkpoint["epoch"],
        "test_metrics": metrics,
        "timing": {
            "pure_training_seconds": pure_training_seconds,
            "training_and_validation_seconds": stage_seconds,
            "inference": benchmark(model, inference_loader, device, args.inference_warmup, args.inference_repeats),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

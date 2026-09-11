"""Generate one RGB physical-field image from a trained forward CNN checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from forward_model import ForwardDecoder, ParameterBounds, tensor_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohesion-pa", type=float, required=True)
    parser.add_argument("--friction-angle-deg", type=float, required=True)
    permeability = parser.add_mutually_exclusive_group(required=True)
    permeability.add_argument("--ks", type=float, help="Permeability coefficient in m/s")
    permeability.add_argument("--log10-ks", type=float, help="Base-10 logarithm of permeability")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-extrapolation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ks is not None and args.ks <= 0:
        raise ValueError("--ks must be positive.")
    log10_ks = math.log10(args.ks) if args.ks is not None else args.log10_ks
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available in PyTorch.")
    device = torch.device(device_name)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    bounds = ParameterBounds.from_dict(config["parameter_bounds"])
    raw = torch.tensor(
        [[args.cohesion_pa, args.friction_angle_deg, log10_ks]], dtype=torch.float32, device=device
    )
    normalized = bounds.normalize(raw)
    if not args.allow_extrapolation and torch.any((normalized < -1.0) | (normalized > 1.0)):
        raise ValueError(
            "The requested parameters are outside the training-domain bounds. "
            f"Bounds: {bounds.as_dict()}. Use --allow-extrapolation only if this is intentional."
        )

    model = ForwardDecoder(config["image_height"], config["image_width"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(normalized)[0]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(prediction).save(output)
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "field": config["field"],
        "cohesion_pa": args.cohesion_pa,
        "friction_angle_deg": args.friction_angle_deg,
        "ks_m_per_s": 10.0**log10_ks,
        "log10_ks": log10_ks,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

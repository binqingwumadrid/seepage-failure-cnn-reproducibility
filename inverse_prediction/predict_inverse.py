"""Infer c, friction angle, and permeability from one pseudocolor image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from inverse_model import InverseRegressor, ParameterBounds, load_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional prediction JSON path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = InverseRegressor(config["image_height"], config["image_width"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = load_image_tensor(args.image, config["image_height"], config["image_width"]).unsqueeze(0).to(device)
    bounds = ParameterBounds.from_dict(config["parameter_bounds"])
    with torch.no_grad():
        raw = bounds.denormalize(model(image))[0].cpu().tolist()
    result = {
        "image": str(args.image.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cohesion_pa": raw[0],
        "friction_angle_deg": raw[1],
        "log10_ks": raw[2],
        "ks_m_per_s": 10.0 ** raw[2],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

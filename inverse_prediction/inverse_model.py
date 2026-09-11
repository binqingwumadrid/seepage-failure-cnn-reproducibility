"""Dataset, parameter scaling, model, and metrics for inverse prediction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


PREFIXED_PATTERN = re.compile(
    r"^c(?P<c>[-+]?\d+(?:\.\d+)?)_phi(?P<phi>[-+]?\d+(?:\.\d+)?)_"
    r"logk(?P<logk>[-+]?\d+(?:\.\d+)?)$",
    flags=re.IGNORECASE,
)
LEGACY_PATTERN = re.compile(
    r"^(?P<c>[-+]?\d+(?:\.\d+)?)_(?P<phi>[-+]?\d+(?:\.\d+)?)_"
    r"(?P<logk>[-+]?\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class ParameterBounds:
    """Bounds used to map c, phi, and log10(ks) between physical values and [0, 1]."""

    minimum: Tuple[float, float, float] = (3000.0, 18.0, -2.3)
    maximum: Tuple[float, float, float] = (7000.0, 27.0, -0.3)

    def normalize(self, values: Tensor) -> Tensor:
        low = torch.as_tensor(self.minimum, dtype=values.dtype, device=values.device)
        high = torch.as_tensor(self.maximum, dtype=values.dtype, device=values.device)
        return (values - low) / (high - low)

    def denormalize(self, values: Tensor) -> Tensor:
        low = torch.as_tensor(self.minimum, dtype=values.dtype, device=values.device)
        high = torch.as_tensor(self.maximum, dtype=values.dtype, device=values.device)
        return values * (high - low) + low

    def as_dict(self) -> Dict[str, List[float]]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}

    @classmethod
    def from_dict(cls, value: Dict[str, Sequence[float]]) -> "ParameterBounds":
        return cls(tuple(value["minimum"]), tuple(value["maximum"]))


def parse_parameters(path: Path | str) -> Tuple[float, float, float]:
    """Read (cohesion Pa, friction angle degrees, log10 permeability) from a name."""

    stem = Path(path).stem
    match = PREFIXED_PATTERN.fullmatch(stem) or LEGACY_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Cannot parse '{Path(path).name}'. Expected "
            "c3000_phi18.00_logk-2.300000.png."
        )
    return float(match.group("c")), float(match.group("phi")), float(match.group("logk"))


def load_image_tensor(path: Path | str, image_height: int, image_width: int) -> Tensor:
    """Load and resize an RGB image, returning a float CHW tensor in [0, 1]."""

    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (image_width, image_height):
            image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


class InverseImageDataset(Dataset):
    """Pseudocolor field images paired with parameters encoded in their filenames."""

    def __init__(
        self,
        image_dir: Path | str,
        image_height: int = 256,
        image_width: int = 512,
        bounds: ParameterBounds | None = None,
        paths: Iterable[Path | str] | None = None,
        cache_images: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.image_height = image_height
        self.image_width = image_width
        self.bounds = bounds or ParameterBounds()
        if paths is None:
            if not self.image_dir.is_dir():
                raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
            self.paths = sorted(
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        else:
            self.paths = [Path(path).expanduser().resolve() for path in paths]
        if not self.paths:
            raise ValueError(f"No PNG/JPEG images found in {self.image_dir}")

        self.raw_parameters = [parse_parameters(path) for path in self.paths]
        if len(set(self.raw_parameters)) != len(self.raw_parameters):
            raise ValueError("Duplicate parameter combinations were found in the image dataset.")
        self._cache = (
            [self._load_uint8(path) for path in self.paths] if cache_images else None
        )

    def _load_uint8(self, path: Path) -> Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (self.image_width, self.image_height):
                image = image.resize((self.image_width, self.image_height), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_uint8 = self._cache[index] if self._cache is not None else self._load_uint8(self.paths[index])
        raw = torch.tensor(self.raw_parameters[index], dtype=torch.float32)
        return {
            "image": image_uint8.float().div_(255.0),
            "parameters": self.bounds.normalize(raw),
            "raw_parameters": raw,
            "filename": self.paths[index].name,
        }


class InverseRegressor(nn.Module):
    """Archived four-block CNN regressor for image-to-three-parameter inversion."""

    def __init__(self, image_height: int = 256, image_width: int = 512) -> None:
        super().__init__()
        if image_height % 16 or image_width % 16:
            raise ValueError("Image height and width must both be divisible by 16.")
        self.image_height = image_height
        self.image_width = image_width
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        flattened = 256 * (image_height // 16) * (image_width // 16)
        self.regressor = nn.Sequential(
            nn.Linear(flattened, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return self.regressor(features.flatten(1))


def parameter_metrics(prediction_raw: Tensor, target_raw: Tensor) -> Dict[str, Tensor]:
    """Return per-parameter MSE, RMSE, MAE, and R2."""

    error = prediction_raw - target_raw
    mse = error.square().mean(dim=0)
    mae = error.abs().mean(dim=0)
    residual = error.square().sum(dim=0)
    centered = target_raw - target_raw.mean(dim=0, keepdim=True)
    total = centered.square().sum(dim=0)
    r2 = torch.where(total > 1e-12, 1.0 - residual / total.clamp_min(1e-12), torch.nan)
    return {"mse": mse, "rmse": mse.sqrt(), "mae": mae, "r2": r2}


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

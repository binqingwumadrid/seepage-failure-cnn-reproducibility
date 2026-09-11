"""Core dataset, model, loss, and metric code for forward field prediction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset


FILENAME_PATTERN = re.compile(
    r"^c(?P<c>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_"
    r"phi(?P<phi>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_"
    r"logk(?P<logk>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParameterBounds:
    """Physical bounds used to scale c, phi, and log10(ks) to [-1, 1]."""

    minimum: Tuple[float, float, float] = (3000.0, 18.0, -2.3)
    maximum: Tuple[float, float, float] = (7000.0, 27.0, -0.3)

    def normalize(self, values: Tensor) -> Tensor:
        low = torch.as_tensor(self.minimum, dtype=values.dtype, device=values.device)
        high = torch.as_tensor(self.maximum, dtype=values.dtype, device=values.device)
        return 2.0 * (values - low) / (high - low) - 1.0

    def denormalize(self, values: Tensor) -> Tensor:
        low = torch.as_tensor(self.minimum, dtype=values.dtype, device=values.device)
        high = torch.as_tensor(self.maximum, dtype=values.dtype, device=values.device)
        return (values + 1.0) * 0.5 * (high - low) + low

    def as_dict(self) -> Dict[str, List[float]]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}

    @classmethod
    def from_dict(cls, value: Dict[str, Sequence[float]]) -> "ParameterBounds":
        return cls(tuple(value["minimum"]), tuple(value["maximum"]))


def parse_parameters(path: Path | str) -> Tuple[float, float, float]:
    """Parse (cohesion in Pa, friction angle in degrees, log10(ks)) from a filename."""

    stem = Path(path).stem
    match = FILENAME_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Cannot parse parameters from '{Path(path).name}'. Expected a name such as "
            "c3000_phi18.00_logk-2.300000.png."
        )
    return (
        float(match.group("c")),
        float(match.group("phi")),
        float(match.group("logk")),
    )


class FieldImageDataset(Dataset):
    """RGB field images paired with the three parameters encoded in each filename."""

    def __init__(
        self,
        image_dir: Path | str,
        image_height: int = 128,
        image_width: int = 256,
        bounds: ParameterBounds | None = None,
        paths: Iterable[Path | str] | None = None,
        cache_images: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.image_height = image_height
        self.image_width = image_width
        self.bounds = bounds or ParameterBounds()
        self.cache_images = cache_images

        if paths is None:
            if not self.image_dir.is_dir():
                raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
            self.paths = sorted(
                p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        else:
            self.paths = [Path(p).expanduser().resolve() for p in paths]

        if not self.paths:
            raise ValueError(f"No PNG/JPEG images found in {self.image_dir}")

        parsed = [parse_parameters(p) for p in self.paths]
        if len(set(parsed)) != len(parsed):
            raise ValueError("Duplicate parameter combinations were found in the image dataset.")
        self.raw_parameters = parsed
        self._image_cache = (
            [self._load_image(path) for path in self.paths] if self.cache_images else None
        )

    def _load_image(self, path: Path) -> Tensor:
        """Load one resized RGB image as uint8; conversion to float happens per batch."""

        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (self.image_width, self.image_height):
                image = image.resize((self.image_width, self.image_height), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.paths[index]
        image_uint8 = self._image_cache[index] if self._image_cache is not None else self._load_image(path)
        image_tensor = image_uint8.to(dtype=torch.float32).div_(255.0)
        raw = torch.tensor(self.raw_parameters[index], dtype=torch.float32)
        normalized = self.bounds.normalize(raw)
        return {
            "image": image_tensor,
            "parameters": normalized,
            "raw_parameters": raw,
            "filename": path.name,
        }


class ForwardDecoder(nn.Module):
    """Three-parameter fully connected embedding followed by three deconvolutions."""

    def __init__(self, image_height: int = 128, image_width: int = 256) -> None:
        super().__init__()
        if image_height % 8 or image_width % 8:
            raise ValueError("Image height and width must both be divisible by 8.")

        self.image_height = image_height
        self.image_width = image_width
        self.latent_height = image_height // 8
        self.latent_width = image_width // 8

        self.embedding = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * self.latent_height * self.latent_width),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, parameters: Tensor) -> Tensor:
        latent = self.embedding(parameters)
        latent = latent.reshape(-1, 256, self.latent_height, self.latent_width)
        return self.decoder(latent)


def _gaussian_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coordinates = torch.arange(5, device=device, dtype=dtype) - 2.0
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, 5, 5).contiguous()


def ssim_index(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean local SSIM over a batch of RGB images, using a 5x5 Gaussian window."""

    channels = prediction.shape[1]
    kernel = _gaussian_kernel(channels, prediction.device, prediction.dtype)

    def blur(value: Tensor) -> Tensor:
        padded = F.pad(value, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)

    mean_pred = blur(prediction)
    mean_target = blur(target)
    var_pred = (blur(prediction.square()) - mean_pred.square()).clamp_min(0.0)
    var_target = (blur(target.square()) - mean_target.square()).clamp_min(0.0)
    covariance = blur(prediction * target) - mean_pred * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_pred * mean_target + c1) * (2.0 * covariance + c2)
    denominator = (mean_pred.square() + mean_target.square() + c1) * (
        var_pred + var_target + c2
    )
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean()


def gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class CompositeImageLoss(nn.Module):
    """Paper loss: MSE + 0.2 * (1 - SSIM) + 0.1 * gradient loss."""

    def __init__(self, mse_weight: float = 1.0, ssim_weight: float = 0.2, gradient_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight

    def components(self, prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
        mse = F.mse_loss(prediction, target)
        ssim = ssim_index(prediction, target)
        gradient = gradient_loss(prediction, target)
        total = self.mse_weight * mse + self.ssim_weight * (1.0 - ssim) + self.gradient_weight * gradient
        return {"loss": total, "mse": mse, "ssim": ssim, "gradient": gradient}

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.components(prediction, target)["loss"]


def per_sample_metrics(prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
    """Return one MSE, MAE, R2, SSIM, and gradient value per sample."""

    batch = prediction.shape[0]
    difference = prediction - target
    flat_difference = difference.reshape(batch, -1)
    flat_target = target.reshape(batch, -1)
    mse = flat_difference.square().mean(dim=1)
    mae = flat_difference.abs().mean(dim=1)
    residual = flat_difference.square().sum(dim=1)
    target_mean = flat_target.mean(dim=1, keepdim=True)
    total = (flat_target - target_mean).square().sum(dim=1)
    r2 = torch.where(
        total > 1e-12,
        1.0 - residual / total.clamp_min(1e-12),
        torch.where(residual <= 1e-12, torch.ones_like(residual), torch.zeros_like(residual)),
    )

    ssim_values: List[Tensor] = []
    gradient_values: List[Tensor] = []
    for index in range(batch):
        sample_prediction = prediction[index : index + 1]
        sample_target = target[index : index + 1]
        ssim_values.append(ssim_index(sample_prediction, sample_target))
        gradient_values.append(gradient_loss(sample_prediction, sample_target))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "ssim": torch.stack(ssim_values),
        "gradient": torch.stack(gradient_values),
    }


def tensor_to_image(value: Tensor) -> Image.Image:
    array = value.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

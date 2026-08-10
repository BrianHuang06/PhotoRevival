"""Shared data, logging, optimization, and loss helpers."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
Pair = tuple[Path, Path, str]


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_training_pairs(clean_dir: str | Path, degraded_dir: str | Path) -> list[Pair]:
    clean_dir = Path(clean_dir)
    degraded_dir = Path(degraded_dir)
    if not clean_dir.is_dir() or not degraded_dir.is_dir():
        raise FileNotFoundError(f"Dataset directories not found: {clean_dir}, {degraded_dir}")

    clean_files = {
        path.name: path
        for path in clean_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = [
        (path, clean_files[path.name], path.name)
        for path in sorted(degraded_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name in clean_files
    ]
    if not pairs:
        raise RuntimeError("No matching clean/degraded image pairs were found")
    return pairs


def split_training_pairs(pairs: list[Pair], test_samples: int) -> tuple[list[Pair], list[Pair]]:
    if len(pairs) < 2:
        raise ValueError("At least two image pairs are required")
    count = min(max(1, test_samples), len(pairs) - 1)
    indices = set(np.linspace(0, len(pairs) - 1, count, dtype=int).tolist())
    test = [pair for index, pair in enumerate(pairs) if index in indices]
    train = [pair for index, pair in enumerate(pairs) if index not in indices]
    return train, test


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name])


def min_snr_weight(noise_scheduler, timesteps: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return Min-SNR weights using alpha_cumprod / (1 - alpha_cumprod)."""
    if gamma <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    alpha = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
    snr = alpha / (1 - alpha).clamp_min(1e-8)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(1e-8)


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Differentiable SSIM loss for images in the [-1, 1] range."""
    channels = prediction.shape[1]
    kernel = torch.ones(
        channels,
        1,
        window_size,
        window_size,
        device=prediction.device,
        dtype=prediction.dtype,
    ) / (window_size**2)
    padding = window_size // 2
    mean_pred = F.conv2d(prediction, kernel, groups=channels, padding=padding)
    mean_target = F.conv2d(target, kernel, groups=channels, padding=padding)
    var_pred = F.conv2d(prediction.square(), kernel, groups=channels, padding=padding) - mean_pred.square()
    var_target = F.conv2d(target.square(), kernel, groups=channels, padding=padding) - mean_target.square()
    covariance = F.conv2d(prediction * target, kernel, groups=channels, padding=padding) - mean_pred * mean_target
    c1, c2 = 0.02**2, 0.06**2
    score = ((2 * mean_pred * mean_target + c1) * (2 * covariance + c2)) / (
        (mean_pred.square() + mean_target.square() + c1) * (var_pred + var_target + c2)
    ).clamp_min(1e-8)
    return 1 - score.mean()

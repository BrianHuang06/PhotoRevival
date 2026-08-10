"""Configuration model for the unified SDXL LoRA trainer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from photo_revival.paths import (
    BASE_MODELS_DIR,
    CACHE_DIR,
    CHECKPOINTS_DIR,
    DATASETS_DIR,
)


@dataclass
class TrainingConfig:
    base_model: str = str(
        BASE_MODELS_DIR
        / "AI-ModelScope"
        / "stable-diffusion-xl-1___0-inpainting-0___1"
    )
    clean_dir: str = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
    degraded_dir: str = str(
        DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded"
    )
    output_dir: str = str(CHECKPOINTS_DIR / "lora_inpainting")
    swinir_cache_dir: str = str(CACHE_DIR / "swinir")
    mask_cache_dir: str = str(CACHE_DIR / "bopbtl_masks")
    swinir_weights: str | None = None
    preprocessor: str = "swinir"
    mask_mode: str = "bopbtl"
    prompt: str = "restored old photo, clear, sharp, natural colors, high quality"
    resolution: int = 512
    epochs: int = 50
    test_samples: int = 6
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    warmup_steps: int = 100
    patience: int = 6
    eval_every: int = 5
    save_every: int = 5
    noise_offset: float = 0.1
    snr_gamma: float = 5.0
    mse_weight: float = 1.0
    lpips_weight: float = 0.0
    ssim_weight: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_8bit_adam: bool = False
    gradient_checkpointing: bool = True
    resume_from: str | None = None
    start_epoch: int = 0
    seed: int = 42

    def validate(self) -> None:
        if self.preprocessor not in {"none", "swinir"}:
            raise ValueError("preprocessor must be 'none' or 'swinir'")
        if self.mask_mode not in {"full", "bopbtl"}:
            raise ValueError("mask_mode must be 'full' or 'bopbtl'")
        if self.resolution % 8:
            raise ValueError("resolution must be divisible by 8")
        if self.epochs < 1 or self.gradient_accumulation < 1:
            raise ValueError("epochs and gradient_accumulation must be positive")
        if min(self.mse_weight, self.lpips_weight, self.ssim_weight) < 0:
            raise ValueError("loss weights cannot be negative")
        if self.mse_weight + self.lpips_weight + self.ssim_weight == 0:
            raise ValueError("at least one loss weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

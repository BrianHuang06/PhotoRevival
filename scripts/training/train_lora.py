"""Train the unified PhotoRevive SDXL inpainting LoRA pipeline."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from photo_revival.training.config import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--prepare-only", action="store_true", help="Build caches without training")
    parser.add_argument("--base-model", dest="base_model")
    parser.add_argument("--clean-dir", dest="clean_dir")
    parser.add_argument("--degraded-dir", dest="degraded_dir")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--swinir-cache-dir", dest="swinir_cache_dir")
    parser.add_argument("--mask-cache-dir", dest="mask_cache_dir")
    parser.add_argument("--swinir-weights", dest="swinir_weights")
    parser.add_argument("--preprocessor", choices=["none", "swinir"])
    parser.add_argument("--mask-mode", choices=["full", "bopbtl"])
    parser.add_argument("--prompt")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--test-samples", dest="test_samples", type=int)
    parser.add_argument("--lora-r", dest="lora_r", type=int)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--gradient-accumulation", dest="gradient_accumulation", type=int)
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval-every", dest="eval_every", type=int)
    parser.add_argument("--save-every", dest="save_every", type=int)
    parser.add_argument("--noise-offset", dest="noise_offset", type=float)
    parser.add_argument("--snr-gamma", dest="snr_gamma", type=float)
    parser.add_argument("--mse-weight", dest="mse_weight", type=float)
    parser.add_argument("--lpips-weight", dest="lpips_weight", type=float)
    parser.add_argument("--ssim-weight", dest="ssim_weight", type=float)
    parser.add_argument("--ema", dest="use_ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--8bit-adam",
        dest="use_8bit_adam",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume-from", dest="resume_from")
    parser.add_argument("--start-epoch", dest="start_epoch", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()
    valid_fields = {field.name for field in fields(TrainingConfig)}
    for name, value in vars(args).items():
        if name in valid_fields and value is not None:
            setattr(config, name, value)
    config.validate()
    return config


def prepare_split(source_pairs, split_name: str, config: TrainingConfig):
    from photo_revival.training.preprocessing import (
        build_training_samples,
        prepare_condition_images,
        prepare_masks,
    )

    conditions = prepare_condition_images(
        source_pairs,
        config.preprocessor,
        Path(config.swinir_cache_dir) / split_name,
        config.swinir_weights,
        config.resolution,
    )
    masks = prepare_masks(
        source_pairs,
        config.mask_mode,
        Path(config.mask_cache_dir) / split_name,
    )
    return build_training_samples(source_pairs, conditions, masks)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    import torch

    from photo_revival.training.common import (
        find_training_pairs,
        log,
        set_seed,
        split_training_pairs,
    )
    from photo_revival.training.lora_trainer import train_lora

    set_seed(config.seed)
    log(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")

    pairs = find_training_pairs(config.clean_dir, config.degraded_dir)
    train_pairs, test_pairs = split_training_pairs(pairs, config.test_samples)
    log(f"Dataset: {len(train_pairs)} train pairs, {len(test_pairs)} evaluation pairs")
    train_samples = prepare_split(train_pairs, "train", config)
    test_samples = prepare_split(test_pairs, "test", config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    if args.prepare_only:
        log("Preprocessing complete; training skipped by --prepare-only")
        return
    train_lora(train_samples, test_samples, config)


if __name__ == "__main__":
    main()

"""Unified SDXL inpainting LoRA training loop."""

from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from photo_revival.training.common import EMAModel, log, min_snr_weight, ssim_loss
from photo_revival.training.config import TrainingConfig
from photo_revival.training.preprocessing import TrainingSample


def _image_tensor(image: Image.Image, resolution: int, device: str) -> torch.Tensor:
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)


def _mask_tensor(mask_path: Path | None, resolution: int, device: str) -> torch.Tensor:
    if mask_path is None:
        return torch.ones(1, 1, resolution, resolution, device=device, dtype=torch.float16)
    image = Image.open(mask_path).convert("L").resize(
        (resolution, resolution), Image.Resampling.NEAREST
    )
    values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)


def _load_lpips(device: str):
    import lpips

    model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _build_optimizer(model, config: TrainingConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            log("Using 8-bit AdamW")
            return bnb.optim.AdamW8bit(
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        except ImportError:
            log("bitsandbytes is unavailable; using torch AdamW", "WARN")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noisy_latent: torch.Tensor,
    clean_image: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    vae,
    vae_scaling_factor: float,
    config: TrainingConfig,
    lpips_model,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_sample_mse = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(
        dim=(1, 2, 3)
    )
    weights = min_snr_weight(noise_scheduler, timesteps, config.snr_gamma)
    mse = (per_sample_mse * weights).mean()
    total = config.mse_weight * mse
    values = {"mse": float(mse.detach())}

    if config.lpips_weight <= 0 and config.ssim_weight <= 0:
        return total, values
    if noise_scheduler.config.prediction_type != "epsilon":
        raise ValueError("Image-space losses currently require epsilon prediction")

    alpha = noise_scheduler.alphas_cumprod.to(noisy_latent.device)[timesteps]
    alpha = alpha.to(noisy_latent.dtype).view(-1, 1, 1, 1)
    predicted_clean_latent = (
        noisy_latent - (1 - alpha).sqrt() * prediction
    ) / alpha.sqrt().clamp_min(1e-6)
    predicted_image = vae.decode(predicted_clean_latent / vae_scaling_factor).sample.clamp(-1, 1)

    if config.lpips_weight > 0:
        lpips_value = lpips_model(predicted_image.float(), clean_image.float()).mean()
        total = total + config.lpips_weight * lpips_value
        values["lpips"] = float(lpips_value.detach())
    if config.ssim_weight > 0:
        ssim_value = ssim_loss(predicted_image.float(), clean_image.float())
        total = total + config.ssim_weight * ssim_value
        values["ssim"] = float(ssim_value.detach())
    return total, values


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    samples: list[TrainingSample],
    config: TrainingConfig,
) -> dict[str, float]:
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_path)
    pipe.enable_model_cpu_offload()

    psnr_values = []
    ssim_values = []
    for condition_path, clean_path, mask_path, _ in samples:
        condition = Image.open(condition_path).convert("RGB").resize(
            (config.resolution, config.resolution), Image.Resampling.LANCZOS
        )
        clean = Image.open(clean_path).convert("RGB").resize(
            (config.resolution, config.resolution), Image.Resampling.LANCZOS
        )
        mask = (
            Image.new("L", condition.size, 255)
            if mask_path is None
            else Image.open(mask_path).convert("L").resize(condition.size, Image.Resampling.NEAREST)
        )
        generator = torch.Generator().manual_seed(config.seed)
        restored = pipe(
            prompt=config.prompt,
            image=condition,
            mask_image=mask,
            strength=0.2,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        clean_values = np.asarray(clean)
        restored_values = np.asarray(restored)
        psnr_values.append(peak_signal_noise_ratio(clean_values, restored_values, data_range=255))
        ssim_values.append(
            structural_similarity(clean_values, restored_values, channel_axis=2, data_range=255)
        )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def train_lora(
    train_samples: list[TrainingSample],
    test_samples: list[TrainingSample],
    config: TrainingConfig,
) -> Path:
    from diffusers import DDPMScheduler, StableDiffusionXLInpaintPipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("Unified LoRA training currently requires CUDA")
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    log("Loading SDXL inpainting pipeline")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    if config.resume_from:
        log(f"Resuming trainable LoRA from {config.resume_from}")
        unet = PeftModel.from_pretrained(
            pipe.unet,
            config.resume_from,
            is_trainable=True,
        )
    else:
        unet = get_peft_model(
            pipe.unet,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none",
            ),
        )
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    device = "cuda"
    unet = unet.to(device)
    pipe.vae = pipe.vae.to(device).eval()
    for parameter in pipe.vae.parameters():
        parameter.requires_grad = False
    scaling_factor = pipe.vae.config.scaling_factor

    optimizer = _build_optimizer(unet, config)
    updates_per_epoch = math.ceil(len(train_samples) / config.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * (config.epochs - config.start_epoch))
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_updates,
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            config.prompt,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_embeds = pooled_embeds.to(device)
    time_ids = torch.tensor(
        [[config.resolution, config.resolution, 0, 0, config.resolution, config.resolution]],
        device=device,
    )
    lpips_model = _load_lpips(device) if config.lpips_weight > 0 else None
    ema = EMAModel(unet, config.ema_decay) if config.use_ema else None
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()

    best_psnr = float("-inf")
    patience_count = 0
    history = []
    log(
        f"Training {len(train_samples)} pairs with preprocessor={config.preprocessor}, "
        f"mask={config.mask_mode}"
    )
    log(
        f"Loss weights: MSE={config.mse_weight}, LPIPS={config.lpips_weight}, "
        f"SSIM={config.ssim_weight}"
    )

    for epoch in range(config.start_epoch, config.epochs):
        unet.train()
        random.shuffle(train_samples)
        epoch_loss = 0.0
        successful_steps = 0
        progress = tqdm(enumerate(train_samples, start=1), total=len(train_samples), desc=f"Epoch {epoch + 1}")
        for sample_index, (condition_path, clean_path, mask_path, name) in progress:
            try:
                clean_image = Image.open(clean_path).convert("RGB")
                condition_image = Image.open(condition_path).convert("RGB")
                mask_tensor = _mask_tensor(mask_path, config.resolution, device)
                clean_tensor = _image_tensor(clean_image, config.resolution, device)
                condition_tensor = _image_tensor(condition_image, config.resolution, device)

                if random.random() < 0.5:
                    clean_tensor = torch.flip(clean_tensor, dims=(3,))
                    condition_tensor = torch.flip(condition_tensor, dims=(3,))
                    mask_tensor = torch.flip(mask_tensor, dims=(3,))

                masked_condition = condition_tensor * (mask_tensor < 0.5)
                with torch.no_grad():
                    clean_latent = pipe.vae.encode(clean_tensor).latent_dist.sample() * scaling_factor
                    condition_latent = pipe.vae.encode(masked_condition).latent_dist.sample() * scaling_factor

                noise = torch.randn_like(clean_latent)
                if config.noise_offset > 0:
                    noise += config.noise_offset * torch.randn(
                        noise.shape[0], noise.shape[1], 1, 1, device=device
                    )
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_latent.shape[0],),
                    device=device,
                )
                noisy_latent = noise_scheduler.add_noise(clean_latent, noise, timesteps)
                latent_mask = F.interpolate(mask_tensor, size=clean_latent.shape[-2:], mode="nearest")

                with torch.amp.autocast("cuda"):
                    # Diffusers inpainting order: noisy latents, mask, masked-image latents.
                    model_input = torch.cat([noisy_latent, latent_mask, condition_latent], dim=1)
                    prediction = unet(
                        model_input,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                    ).sample
                    target = (
                        noise
                        if noise_scheduler.config.prediction_type == "epsilon"
                        else noise_scheduler.get_velocity(clean_latent, noise, timesteps)
                    )
                    loss, loss_values = _training_loss(
                        prediction,
                        target,
                        noisy_latent,
                        clean_tensor,
                        timesteps,
                        noise_scheduler,
                        pipe.vae,
                        scaling_factor,
                        config,
                        lpips_model,
                    )

                scaler.scale(loss / config.gradient_accumulation).backward()
                should_update = (
                    sample_index % config.gradient_accumulation == 0
                    or sample_index == len(train_samples)
                )
                if should_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(unet)

                loss_value = float(loss.detach())
                epoch_loss += loss_value
                successful_steps += 1
                progress.set_postfix(loss=f"{loss_value:.4f}", **loss_values)
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                log(f"Training step failed for {name}: {error}", "WARN")

        if successful_steps == 0:
            raise RuntimeError("Every training step failed; inspect the preceding warnings")
        average_loss = epoch_loss / successful_steps
        history.append({"epoch": epoch + 1, "loss": average_loss, "steps": successful_steps})
        with (output_dir / "loss_log.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        log(f"Epoch {epoch + 1}: loss={average_loss:.6f}")

        should_save = (epoch + 1) % config.save_every == 0
        should_evaluate = test_samples and (epoch + 1) % config.eval_every == 0
        if should_save or should_evaluate:
            backup = ema.apply(unet) if ema else None
            checkpoint = output_dir / f"epoch{epoch + 1}"
            unet.save_pretrained(checkpoint)
            if should_evaluate:
                unet.to("cpu")
                torch.cuda.empty_cache()
                metrics = evaluate_checkpoint(checkpoint, test_samples, config)
                unet.to(device)
                log(f"Evaluation: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    unet.save_pretrained(output_dir / "best")
                    patience_count = 0
                else:
                    patience_count += 1
            if ema and backup is not None:
                ema.restore(unet, backup)
            if should_evaluate and patience_count >= config.patience:
                log("Early stopping")
                break

    final_backup = ema.apply(unet) if ema else None
    final_path = output_dir / "final"
    unet.save_pretrained(final_path)
    if ema and final_backup is not None:
        ema.restore(unet, final_backup)
    log(f"Final LoRA saved to {final_path}")
    return final_path

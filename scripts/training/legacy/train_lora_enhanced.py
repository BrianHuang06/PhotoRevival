# -*- coding: utf-8 -*-
"""
Enhanced Photo Restoration Training - Inpainting + LoRA

Features:
  1. More configurable parameters
  2. Multiple optimizer options
  3. Learning rate scheduler options
  4. Data augmentation
  5. EMA (Exponential Moving Average)
  6. SNR Gamma weighting
  7. Noise offset
  8. Multi-resolution training
  9. Gradient checkpointing options
  10. Better logging and checkpointing
"""

import os
import sys
import gc
import json
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from photo_revival.paths import ARTIFACTS_DIR, BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, LOGS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONFIG = {
    "base_model": str(BASE_MODELS_DIR / "sdxl-inpainting"),
    "clean_dir": str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean"),
    "degraded_dir": str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded"),
    "output_dir": str(CHECKPOINTS_DIR / "lora_inpainting_v26"),
    "test_output_dir": str(ARTIFACTS_DIR / "evaluations" / "test_outputs_v26"),
    "log_file": str(LOGS_DIR / "training_log_v26.json"),
    
    "prompt": "restored old photo, clear, sharp, natural colors, high quality, no damage",
    "resolution": 512,
    "multi_resolution": False,
    "resolutions": [512, 640, 768],
    
    "num_epochs": 50,
    "batch_size": 1,
    "gradient_accumulation": 4,
    
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    
    "optimizer": "AdamW",
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "use_8bit_adam": False,
    "use_adafactor": False,
    
    "lr_scheduler": "cosine",
    "warmup_steps": 100,
    "warmup_ratio": 0.0,
    "min_lr": 0.0,
    "lr_decay_cycles": 0.5,
    
    "max_grad_norm": 1.0,
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    
    "noise_offset": 0.0,
    "snr_gamma": 0.0,
    "input_perturbation": 0.0,
    
    "use_ema": False,
    "ema_decay": 0.9999,
    "ema_update_after_step": 0,
    
    "data_augmentation": True,
    "random_flip": True,
    "random_rotation": False,
    "rotation_degrees": 5,
    "color_jitter": False,
    "brightness": 0.1,
    "contrast": 0.1,
    "saturation": 0.1,
    
    "eval_every": 5,
    "save_every": 5,
    "test_samples": 10,
    "patience": 6,
    
    "seed": 42,
    "deterministic": False,
    
    "strength_test_values": [0.1, 0.2, 0.3],
    "guidance_test_values": [3.0, 5.0],
}


def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_msg = msg.encode('utf-8', errors='ignore').decode('utf-8')
    print(f"[{timestamp}] [{level}] {safe_msg}")
    sys.stdout.flush()


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_training_pairs(clean_dir: str, degraded_dir: str) -> List[Tuple[str, str, str]]:
    clean_files = set(f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    degraded_files = sorted(f for f in os.listdir(degraded_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    
    pairs = []
    for df in degraded_files:
        if df in clean_files:
            pairs.append((
                os.path.join(degraded_dir, df),
                os.path.join(clean_dir, df),
                df,
            ))
    
    log(f"Found {len(pairs)} training pairs")
    return pairs


def get_test_pairs(pairs: List, num: int = 10) -> List:
    step = max(1, len(pairs) // num)
    return [pairs[i] for i in range(0, len(pairs), step)][:num]


class DataAugmentation:
    def __init__(self, config: Dict):
        self.config = config
        
    def apply(self, clean_pil: Image.Image, degraded_pil: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.config.get("random_flip", True) and random.random() < 0.5:
            clean_pil = clean_pil.transpose(Image.FLIP_LEFT_RIGHT)
            degraded_pil = degraded_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.config.get("random_rotation", False):
            angle = random.uniform(-self.config.get("rotation_degrees", 5), self.config.get("rotation_degrees", 5))
            clean_pil = clean_pil.rotate(angle, resample=Image.BICUBIC, expand=False)
            degraded_pil = degraded_pil.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        if self.config.get("color_jitter", False):
            from torchvision import transforms
            jitter = transforms.ColorJitter(
                brightness=self.config.get("brightness", 0.1),
                contrast=self.config.get("contrast", 0.1),
                saturation=self.config.get("saturation", 0.1),
            )
            clean_pil = jitter(clean_pil)
            degraded_pil = jitter(degraded_pil)
        
        return clean_pil, degraded_pil


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, update_after_step: int = 0):
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0
        self.shadow_params = {}
        self.collected_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self, model: torch.nn.Module):
        self.step += 1
        if self.step < self.update_after_step:
            return
        
        decay = self.decay
        if self.step < 1000:
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def copy_to(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])
    
    def store(self, model: torch.nn.Module):
        self.collected_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collected_params.append(param.data.clone())
    
    def restore(self, model: torch.nn.Module):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.collected_params[idx])
                idx += 1


def compute_snr_weight(noise_scheduler, timesteps: torch.Tensor, snr_gamma: float) -> torch.Tensor:
    if snr_gamma <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    
    snr = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2
    snr_weight = torch.minimum(snr, torch.ones_like(snr) * snr_gamma) / snr
    
    return snr_weight


def get_optimizer(model_params, config: Dict) -> torch.optim.Optimizer:
    optimizer_type = config.get("optimizer", "AdamW")
    lr = config.get("learning_rate", 5e-5)
    weight_decay = config.get("weight_decay", 0.01)
    
    if config.get("use_adafactor", False):
        from transformers import Adafactor
        return Adafactor(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=True,
            relative_step=False,
        )
    
    if config.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            )
        except ImportError:
            log("bitsandbytes not available, falling back to AdamW", "WARN")
    
    if optimizer_type == "Adam":
        return torch.optim.Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    
    return torch.optim.AdamW(
        model_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
        eps=config.get("adam_epsilon", 1e-8),
    )


def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict, total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    from diffusers.optimization import get_scheduler
    
    scheduler_type = config.get("lr_scheduler", "cosine")
    warmup_steps = config.get("warmup_steps", 100)
    
    if config.get("warmup_ratio", 0.0) > 0:
        warmup_steps = int(total_steps * config["warmup_ratio"])
    
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=config.get("lr_decay_cycles", 0.5),
    )


def evaluate(pipe, pairs, name: str, config: Dict, lpips_fn=None) -> Optional[Dict]:
    from diffusers import DDIMScheduler
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    strength_values = config.get("strength_test_values", [0.3])
    guidance_values = config.get("guidance_test_values", [7.5])
    
    log(f"Evaluating {name} on {len(pairs)} images...")
    
    results = {"psnr_before": [], "psnr_after": [], "ssim_before": [], "ssim_after": []}
    if lpips_fn:
        results["lpips_after"] = []
    
    original_scheduler = pipe.scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    for idx, (dmg_path, clean_path, fname) in enumerate(pairs):
        try:
            damaged_img = Image.open(dmg_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")
            w, h = clean_img.size
            damaged_img = damaged_img.resize((w, h), Image.LANCZOS)
            
            resolution = config.get("resolution", 512)
            input_resized = damaged_img.resize((resolution, resolution), Image.LANCZOS)
            full_mask = Image.new("L", (resolution, resolution), 255)
            
            generator = torch.Generator("cuda").manual_seed(config.get("seed", 42))
            
            result = pipe(
                prompt=config.get("prompt", "restored photo"),
                image=input_resized,
                mask_image=full_mask,
                strength=strength_values[0],
                num_inference_steps=25,
                guidance_scale=guidance_values[0],
                generator=generator,
            ).images[0]
            
            restored = result.resize((w, h), Image.LANCZOS)
            
            damaged_arr = np.array(damaged_img)
            clean_arr = np.array(clean_img)
            restored_arr = np.array(restored)
            
            results["psnr_before"].append(psnr(clean_arr, damaged_arr))
            results["ssim_before"].append(ssim(clean_arr, damaged_arr, channel_axis=2))
            results["psnr_after"].append(psnr(clean_arr, restored_arr))
            results["ssim_after"].append(ssim(clean_arr, restored_arr, channel_axis=2))
            
            if lpips_fn:
                t1 = torch.from_numpy(clean_arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
                t2 = torch.from_numpy(restored_arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
                with torch.no_grad():
                    results["lpips_after"].append(lpips_fn(t1.cuda(), t2.cuda()).item())
            
        except Exception as e:
            log(f"  Error on {fname}: {e}", "WARN")
    
    pipe.scheduler = original_scheduler
    
    if not results["psnr_before"]:
        return None
    
    return {
        "psnr_before": float(np.mean(results["psnr_before"])),
        "psnr_after": float(np.mean(results["psnr_after"])),
        "ssim_before": float(np.mean(results["ssim_before"])),
        "ssim_after": float(np.mean(results["ssim_after"])),
        "lpips_after": float(np.mean(results["lpips_after"])) if results.get("lpips_after") else None,
    }


def train(train_pairs: List, test_pairs: List, config: Dict, lpips_fn=None) -> Tuple[str, str, List]:
    from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    
    log(f"\n{'='*60}")
    log(f"Enhanced Training - Inpainting + LoRA")
    log(f"{'='*60}")
    
    output_dir = config.get("output_dir", "lora_output")
    test_output_dir = config.get("test_output_dir", "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    best_path = os.path.join(output_dir, "best")
    os.makedirs(best_path, exist_ok=True)
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    log("Loading base model...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        config.get("base_model", str(BASE_MODELS_DIR / "sdxl-inpainting")),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    
    log("Adding LoRA to UNet...")
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
    )
    unet = get_peft_model(pipe.unet, lora_config)
    
    if config.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()
    
    params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    log(f"Trainable LoRA params: {params:,}")
    
    vae_scaling_factor = pipe.vae.config.scaling_factor
    unet = unet.to("cuda")
    
    pipe.vae = pipe.vae.to("cuda").eval()
    for p in pipe.vae.parameters():
        p.requires_grad = False
    
    optimizer = get_optimizer(unet.parameters(), config)
    
    grad_acc = config.get("gradient_accumulation", 4)
    total_steps = (config.get("num_epochs", 50) * len(train_pairs)) // grad_acc
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    log("Encoding prompt...")
    with torch.no_grad():
        emb, _, pooled, _ = pipe.encode_prompt(
            prompt=config.get("prompt", "restored photo"),
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        emb = emb.to("cuda")
        pooled = pooled.to("cuda")
    
    resolution = config.get("resolution", 512)
    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device="cuda")
    
    ema = None
    if config.get("use_ema", False):
        ema = EMAModel(
            unet,
            decay=config.get("ema_decay", 0.9999),
            update_after_step=config.get("ema_update_after_step", 0),
        )
        log(f"EMA enabled with decay={config.get('ema_decay', 0.9999)}")
    
    augmentation = DataAugmentation(config) if config.get("data_augmentation", True) else None
    
    log(f"Training: {len(train_pairs)} pairs, {config.get('num_epochs', 50)} epochs, {total_steps} steps")
    
    global_step = 0
    optimizer_step = 0
    loss_history = []
    best_psnr = 0
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')
    
    noise_offset = config.get("noise_offset", 0.0)
    snr_gamma = config.get("snr_gamma", 0.0)
    input_perturbation = config.get("input_perturbation", 0.0)
    
    for epoch in range(config.get("num_epochs", 50)):
        epoch_loss = 0.0
        epoch_count = 0
        unet.train()
        
        np.random.shuffle(train_pairs)
        pbar = tqdm(train_pairs, desc=f"Epoch {epoch+1}/{config.get('num_epochs', 50)}")
        
        for dmg_path, clean_path, fname in pbar:
            try:
                current_resolution = resolution
                if config.get("multi_resolution", False):
                    current_resolution = random.choice(config.get("resolutions", [512, 640, 768]))
                
                clean_pil = Image.open(clean_path).convert("RGB").resize(
                    (current_resolution, current_resolution), Image.LANCZOS
                )
                dmg_pil = Image.open(dmg_path).convert("RGB").resize(
                    (current_resolution, current_resolution), Image.LANCZOS
                )
                
                if augmentation:
                    clean_pil, dmg_pil = augmentation.apply(clean_pil, dmg_pil)
                
                clean_img = torch.from_numpy(
                    np.array(clean_pil).astype(np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).cuda().half()
                
                dmg_img = torch.from_numpy(
                    np.array(dmg_pil).astype(np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).cuda().half()
                
                with torch.no_grad():
                    clean_latent = pipe.vae.encode(clean_img).latent_dist.sample() * vae_scaling_factor
                    dmg_latent = pipe.vae.encode(dmg_img).latent_dist.sample() * vae_scaling_factor
                
                noise = torch.randn_like(dmg_latent)
                
                if noise_offset > 0:
                    noise += noise_offset * torch.randn(
                        noise.shape[0], noise.shape[1], 1, 1, device=noise.device
                    )
                
                t = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda"
                )
                
                noisy = noise_scheduler.add_noise(dmg_latent, noise, t)
                
                if input_perturbation > 0:
                    noisy = noisy + input_perturbation * torch.randn_like(noisy)
                
                latent_h, latent_w = noisy.shape[2], noisy.shape[3]
                mask_latent = torch.ones(1, 1, latent_h, latent_w, device="cuda", dtype=torch.float16)
                
                unet_in = torch.cat([noisy, dmg_latent, mask_latent], dim=1)
                
                current_time_ids = torch.tensor(
                    [[current_resolution, current_resolution, 0, 0, current_resolution, current_resolution]],
                    device="cuda"
                )
                cond = {"text_embeds": pooled, "time_ids": current_time_ids}
                
                with torch.amp.autocast('cuda'):
                    pred = unet(unet_in, t, encoder_hidden_states=emb, added_cond_kwargs=cond).sample
                    loss = F.mse_loss(pred, noise, reduction="none")
                    
                    if snr_gamma > 0:
                        snr_weight = compute_snr_weight(noise_scheduler, t, snr_gamma)
                        loss = loss * snr_weight.view(-1, 1, 1, 1)
                    
                    loss = loss.mean()
                    loss_scaled = loss / grad_acc
                
                scaler.scale(loss_scaled).backward()
                
                if (global_step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), 
                        max_norm=config.get("max_grad_norm", 1.0)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1
                    
                    if ema:
                        ema.update(unet)
                
                epoch_loss += loss.item()
                epoch_count += 1
                global_step += 1
                
                if global_step % 50 == 0:
                    avg_loss = epoch_loss / epoch_count
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": optimizer_step
                    })
                
                del clean_latent, dmg_latent, noisy, noise, unet_in, pred
                
            except Exception as e:
                log(f"  Error on {fname}: {e}", "WARN")
                continue
        
        if epoch_count == 0:
            continue
        
        avg_epoch_loss = epoch_loss / epoch_count
        loss_history.append(avg_epoch_loss)
        log(f"\nEpoch {epoch+1}: loss={avg_epoch_loss:.6f}, steps={optimizer_step}")
        
        if (epoch + 1) % config.get("save_every", 5) == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}")
            unet.save_pretrained(checkpoint_path)
            log(f"  Checkpoint saved: {checkpoint_path}")
        
        if (epoch + 1) % config.get("eval_every", 5) == 0:
            log(f"\n--- Evaluation at epoch {epoch+1} ---")
            
            if ema:
                ema.store(unet)
                ema.copy_to(unet)
            
            eval_path = os.path.join(output_dir, f"eval_epoch{epoch+1}")
            unet.save_pretrained(eval_path)
            
            from peft import PeftModel
            eval_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                config.get("base_model", str(BASE_MODELS_DIR / "sdxl-inpainting")),
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                low_cpu_mem_usage=True,
            )
            eval_pipe.unet = PeftModel.from_pretrained(eval_pipe.unet, eval_path)
            eval_pipe.enable_model_cpu_offload()
            
            best_so_far = False
            for strength in config.get("strength_test_values", [0.3]):
                r = evaluate(eval_pipe, test_pairs, f"epoch{epoch+1}_s{strength}", config, lpips_fn)
                if r and r["psnr_after"] > best_psnr:
                    best_psnr = r["psnr_after"]
                    best_so_far = True
                    log(f"  New best PSNR: {best_psnr:.2f}")
            
            if best_so_far:
                patience_counter = 0
                unet.save_pretrained(best_path)
                log(f"  Best checkpoint saved!")
            else:
                patience_counter += 1
                log(f"  No improvement (patience: {patience_counter}/{config.get('patience', 6)})")
                if patience_counter >= config.get("patience", 6):
                    log(f"  Early stopping at epoch {epoch+1}")
                    del eval_pipe
                    torch.cuda.empty_cache()
                    gc.collect()
                    break
            
            if ema:
                ema.restore(unet)
            
            del eval_pipe
            torch.cuda.empty_cache()
            gc.collect()
            unet.train()
    
    final_path = os.path.join(output_dir, "final")
    unet.save_pretrained(final_path)
    log(f"Final model saved: {final_path}")
    
    del unet, pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_path, final_path, loss_history


def main():
    parser = argparse.ArgumentParser(description="Enhanced LoRA Training for Photo Restoration")
    
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--clean_dir", type=str, default=None)
    parser.add_argument("--degraded_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    
    parser.add_argument("--noise_offset", type=float, default=None, help="Noise offset for training")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR gamma weighting")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA")
    parser.add_argument("--ema_decay", type=float, default=None)
    
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam")
    parser.add_argument("--multi_resolution", action="store_true", help="Enable multi-resolution training")
    
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config.update(json.load(f))
    
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    
    log("=" * 60)
    log("Enhanced LoRA Training for Photo Restoration")
    log("=" * 60)
    
    set_seed(config.get("seed", 42), config.get("deterministic", False))
    
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    log(f"\nConfig:")
    for key, value in sorted(config.items()):
        log(f"  {key}: {value}")
    
    all_pairs = get_training_pairs(config["clean_dir"], config["degraded_dir"])
    if not all_pairs:
        log("No training pairs found!", "ERROR")
        return
    
    test_pairs = get_test_pairs(all_pairs, num=config.get("test_samples", 10))
    train_pairs = [p for p in all_pairs if p not in test_pairs]
    log(f"\nDataset: {len(all_pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")
    
    log("\nLoading LPIPS...")
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
        log("LPIPS loaded")
    except ImportError:
        log("LPIPS not available, skipping perceptual metrics")
        lpips_fn = None
    
    best_path, final_path, loss_history = train(train_pairs, test_pairs, config, lpips_fn)
    
    log("\n" + "=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    log(f"Best model: {best_path}")
    log(f"Final model: {final_path}")
    log(f"Loss history: {len(loss_history)} epochs")


if __name__ == "__main__":
    main()

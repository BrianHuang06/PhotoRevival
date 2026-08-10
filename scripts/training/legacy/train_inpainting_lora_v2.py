# -*- coding: utf-8 -*-
"""
Photo Restoration Training - Inpainting + LoRA (Fixed)

KEY FIXES from previous versions:
  1. Training adds noise to DEGRADED latent (not clean) -> matches inference
  2. Model predicts noise (epsilon), loss = MSE(pred, noise)
  3. Training input: noisy_degraded + degraded_image + mask -> UNet
  4. Inference input: noisy (from degraded) + degraded_image + mask -> UNet
  5. Training and inference are now CONSISTENT

Why this works:
  - At inference, the pipeline: encodes degraded image -> adds noise to step t
    -> UNet predicts noise -> removes it -> gets cleaner image
  - At training, we do the same thing with known (noise, degraded, clean) pairs
  - The model learns to remove noise AND degradation together
"""

import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from photo_revival.paths import ARTIFACTS_DIR, BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, LOGS_DIR

BASE_MODEL_ID = str(BASE_MODELS_DIR / "sdxl-inpainting")
CLEAN_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
DEGRADED_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
OUTPUT_DIR = str(CHECKPOINTS_DIR / "lora_inpainting_v25")
TEST_OUTPUT_DIR = str(ARTIFACTS_DIR / "evaluations" / "test_outputs_v25")
LOG_FILE = str(LOGS_DIR / "training_log_v25.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

PROMPT = "restored old photo, clear, sharp, natural colors, high quality, no damage"
RESOLUTION = 1024


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_msg = msg.encode('utf-8', errors='ignore').decode('utf-8')
    print(f"[{timestamp}] [{level}] {safe_msg}")
    sys.stdout.flush()


def get_training_pairs():
    clean_files = set(f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    degraded_files = sorted(f for f in os.listdir(DEGRADED_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pairs = []
    for df in degraded_files:
        if df in clean_files:
            pairs.append((
                os.path.join(DEGRADED_DIR, df),
                os.path.join(CLEAN_DIR, df),
                df,
            ))

    log(f"Found {len(pairs)} training pairs")
    return pairs


def get_test_pairs(pairs, num=10):
    step = max(1, len(pairs) // num)
    return [pairs[i] for i in range(0, len(pairs), step)][:num]


def evaluate(pipe, pairs, name="base", strength=0.3, guidance_scale=7.5, lpips_fn=None):
    from diffusers import DDIMScheduler
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    log(f"Evaluating {name} on {len(pairs)} images (strength={strength}, guidance={guidance_scale})...")
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

            damaged_arr = np.array(damaged_img)
            clean_arr = np.array(clean_img)

            p_before = psnr(clean_arr, damaged_arr)
            s_before = ssim(clean_arr, damaged_arr, channel_axis=2)

            input_512 = damaged_img.resize((512, 512), Image.LANCZOS)
            full_mask = Image.new("L", (512, 512), 255)

            generator = torch.Generator("cuda").manual_seed(42)
            result = pipe(
                prompt=PROMPT,
                image=input_512,
                mask_image=full_mask,
                strength=strength,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                eta=0.0,
                generator=generator,
            ).images[0]

            restored = result.resize((w, h), Image.LANCZOS)
            restored_arr = np.array(restored)

            p_after = psnr(clean_arr, restored_arr)
            s_after = ssim(clean_arr, restored_arr, channel_axis=2)

            results["psnr_before"].append(p_before)
            results["psnr_after"].append(p_after)
            results["ssim_before"].append(s_before)
            results["ssim_after"].append(s_after)

            if lpips_fn:
                lpips_val = compute_lpips(clean_img, restored, lpips_fn)
                results["lpips_after"].append(lpips_val)

            if idx < 3:
                restored.save(os.path.join(TEST_OUTPUT_DIR, f"{name}_{fname}"))

            log(f"  [{idx+1}/{len(pairs)}] {fname}: PSNR {p_before:.1f}->{p_after:.1f}, SSIM {s_before:.4f}->{s_after:.4f}" +
                (f", LPIPS {lpips_val:.4f}" if lpips_fn else ""))

        except Exception as e:
            log(f"  Error on {fname}: {e}", "WARN")

    pipe.scheduler = original_scheduler

    if not results["psnr_before"]:
        return None

    result = {
        "psnr_before": float(np.mean(results["psnr_before"])),
        "psnr_after": float(np.mean(results["psnr_after"])),
        "ssim_before": float(np.mean(results["ssim_before"])),
        "ssim_after": float(np.mean(results["ssim_after"])),
    }
    if results.get("lpips_after"):
        result["lpips_after"] = float(np.mean(results["lpips_after"]))

    log(f"  AVG: PSNR {result['psnr_before']:.2f}->{result['psnr_after']:.2f}, " +
        f"SSIM {result['ssim_before']:.4f}->{result['ssim_after']:.4f}" +
        (f", LPIPS {result['lpips_after']:.4f}" if "lpips_after" in result else ""))
    return result


def compute_lpips(img1, img2, lpips_fn):
    img1_t = torch.from_numpy(np.array(img1)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_t = torch.from_numpy(np.array(img2)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img1_t = img1_t * 2 - 1
    img2_t = img2_t * 2 - 1
    with torch.no_grad():
        return lpips_fn(img1_t.cuda(), img2_t.cuda()).item()


def train(train_pairs, test_pairs, config, lpips_fn):
    from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model

    log(f"\n{'='*60}")
    log(f"Training - Inpainting + LoRA (FIXED: train-test match)")
    log(f"Config: {json.dumps(config, indent=2, default=str)}")
    log(f"{'='*60}")

    best_path = os.path.join(OUTPUT_DIR, "best")
    os.makedirs(best_path, exist_ok=True)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16", low_cpu_mem_usage=True
    )

    # Add LoRA to UNet
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.enable_gradient_checkpointing()

    params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    log(f"Trainable LoRA params: {params:,}")

    vae_scaling_factor = pipe.vae.config.scaling_factor
    unet = unet.to("cuda")

    # Freeze VAE, move to GPU
    pipe.vae = pipe.vae.to("cuda").eval()
    for p in pipe.vae.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    grad_acc = config["gradient_accumulation"]
    total_steps = (config["num_epochs"] * len(train_pairs)) // grad_acc
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps
    )

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Pre-compute text embeddings
    log("Encoding prompt...")
    with torch.no_grad():
        emb, _, pooled, _ = pipe.encode_prompt(
            prompt=PROMPT,
            device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        emb = emb.to("cuda")
        pooled = pooled.to("cuda")
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device="cuda")

    log(f"Training on {len(train_pairs)} pairs, {config['num_epochs']} epochs")
    log(f"Total optimizer steps: {total_steps}")

    global_step = 0
    optimizer_step = 0
    loss_history = []
    best_psnr = 0
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        epoch_count = 0
        unet.train()

        np.random.shuffle(train_pairs)
        pbar = tqdm(train_pairs, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for dmg_path, clean_path, fname in pbar:
            try:
                clean_pil = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                dmg_pil = Image.open(dmg_path).convert("RGB").resize((512, 512), Image.LANCZOS)

                # Random horizontal flip
                if np.random.random() < 0.5:
                    dmg_pil = dmg_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    clean_pil = clean_pil.transpose(Image.FLIP_LEFT_RIGHT)

                # Encode to latents
                clean_img = torch.from_numpy(
                    np.array(clean_pil).astype(np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).cuda().half()

                dmg_img = torch.from_numpy(
                    np.array(dmg_pil).astype(np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).cuda().half()

                with torch.no_grad():
                    clean_latent = pipe.vae.encode(clean_img).latent_dist.sample() * vae_scaling_factor
                    dmg_latent = pipe.vae.encode(dmg_img).latent_dist.sample() * vae_scaling_factor

                # === KEY FIX: Add noise to DEGRADED latent (not clean) ===
                # This matches what the inpainting pipeline does at inference:
                #   1. Encode degraded image -> dmg_latent
                #   2. Add noise to dmg_latent at timestep t
                #   3. UNet receives: [noisy, dmg_latent, mask]
                #   4. UNet predicts the noise
                #   5. Remove noise -> cleaner image
                noise = torch.randn_like(dmg_latent)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda")
                noisy = noise_scheduler.add_noise(dmg_latent, noise, t)

                # Full mask (tell model to restore entire image)
                mask_latent = torch.ones(1, 1, 64, 64, device="cuda", dtype=torch.float16)

                # UNet input: [noisy_latent, degraded_image_latent, mask]
                # This is exactly what the inpainting pipeline does
                unet_in = torch.cat([noisy, dmg_latent, mask_latent], dim=1)

                cond = {"text_embeds": pooled, "time_ids": time_ids}

                with torch.amp.autocast('cuda'):
                    pred = unet(unet_in, t, encoder_hidden_states=emb, added_cond_kwargs=cond).sample
                    # Epsilon prediction: model predicts the noise
                    loss = F.mse_loss(pred, noise)
                    loss_scaled = loss / grad_acc

                scaler.scale(loss_scaled).backward()

                if (global_step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

                epoch_loss += loss.item()
                epoch_count += 1
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / epoch_count
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "opt_step": optimizer_step})

                del clean_latent, dmg_latent, noisy, noise, unet_in, pred

            except Exception as e:
                log(f"  Error on {fname}: {e}", "WARN")
                continue

        if epoch_count == 0:
            continue

        avg_epoch_loss = epoch_loss / epoch_count
        loss_history.append(avg_epoch_loss)
        log(f"\nEpoch {epoch+1} complete: loss={avg_epoch_loss:.6f}, optimizer_steps={optimizer_step}")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            log(f"\n--- Evaluation at epoch {epoch+1} ---")

            # Save current LoRA weights
            current_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
            unet.save_pretrained(current_path)

            from peft import PeftModel
            eval_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                BASE_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True,
                variant="fp16", low_cpu_mem_usage=True
            )
            eval_pipe.unet = PeftModel.from_pretrained(eval_pipe.unet, current_path)
            eval_pipe.enable_model_cpu_offload()

            best_so_far = False
            for strength in [0.1, 0.2, 0.3]:
                r = evaluate(
                    eval_pipe, test_pairs, f"epoch{epoch+1}_s{strength}",
                    strength=strength, guidance_scale=3.0, lpips_fn=lpips_fn
                )
                if r and r["psnr_after"] > best_psnr:
                    best_psnr = r["psnr_after"]
                    best_so_far = True

            if best_so_far:
                patience_counter = 0
                unet.save_pretrained(best_path)
                log(f"  Best checkpoint saved (PSNR: {best_psnr:.2f})")
            else:
                patience_counter += 1
                log(f"  No improvement (patience: {patience_counter}/{config['patience']})")
                if patience_counter >= config["patience"]:
                    log(f"  Early stopping at epoch {epoch+1}")
                    del eval_pipe
                    torch.cuda.empty_cache()
                    gc.collect()
                    break

            del eval_pipe
            torch.cuda.empty_cache()
            gc.collect()
            unet.train()

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final")
    unet.save_pretrained(final_path)
    log(f"Final model saved: {final_path}")

    del unet, pipe
    torch.cuda.empty_cache()
    gc.collect()

    return best_path, final_path, loss_history


def main():
    log("=" * 60)
    log("Photo Restoration - Inpainting + LoRA (FIXED)")
    log("Key fix: noise added to DEGRADED latent, not clean")
    log("=" * 60)
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_pairs = get_training_pairs()
    if not all_pairs:
        log("No training pairs found!", "ERROR")
        return

    test_pairs = get_test_pairs(all_pairs, num=10)
    train_pairs = [p for p in all_pairs if p not in test_pairs]
    log(f"Dataset: {len(all_pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")

    # Load LPIPS
    log("\nLoading LPIPS...")
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    log("LPIPS loaded")

    # Evaluate base model
    log("\n" + "=" * 60)
    log("Step 0: Evaluate Base Inpainting Model")
    log("=" * 60)

    from diffusers import StableDiffusionXLInpaintPipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16", low_cpu_mem_usage=True
    )
    pipe.enable_model_cpu_offload()

    base_results = {}
    for strength in [0.1, 0.2, 0.3]:
        for guidance in [3.0, 5.0]:
            key = f"s{strength}_g{guidance}"
            log(f"\n  Testing {key}...")
            r = evaluate(pipe, test_pairs, f"base_{key}",
                         strength=strength, guidance_scale=guidance, lpips_fn=lpips_fn)
            if r:
                base_results[key] = r

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # Training config
    config = {
        "num_epochs": 50,
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "gradient_accumulation": 4,
        "lora_dropout": 0.01,
        "weight_decay": 0.01,
        "patience": 6,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }

    best_path, final_path, loss_history = train(train_pairs, test_pairs, config, lpips_fn)

    # Final evaluation
    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)

    log("\n--- Base Model ---")
    for key, r in base_results.items():
        log(f"  {key}: PSNR {r['psnr_after']:.2f}, SSIM {r['ssim_after']:.4f}" +
            (f", LPIPS {r['lpips_after']:.4f}" if "lpips_after" in r else ""))

    from peft import PeftModel
    lora_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16", low_cpu_mem_usage=True
    )
    lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, best_path)
    lora_pipe.enable_model_cpu_offload()

    lora_results = {}
    for strength in [0.1, 0.2, 0.3]:
        for guidance in [3.0, 5.0]:
            key = f"s{strength}_g{guidance}"
            log(f"\n  Testing LoRA {key}...")
            r = evaluate(lora_pipe, test_pairs, f"lora_{key}",
                         strength=strength, guidance_scale=guidance, lpips_fn=lpips_fn)
            if r:
                lora_results[key] = r

    del lora_pipe
    torch.cuda.empty_cache()
    gc.collect()

    log("\n--- LoRA Model ---")
    for key, r in lora_results.items():
        base_r = base_results.get(key, {})
        log(f"  {key}: PSNR {r['psnr_after']:.2f} (base: {base_r.get('psnr_after', 'N/A')}), " +
            f"SSIM {r['ssim_after']:.4f} (base: {base_r.get('ssim_after', 'N/A')})" +
            (f", LPIPS {r['lpips_after']:.4f}" if "lpips_after" in r else ""))

    all_results = {
        "base": base_results,
        "lora": lora_results,
        "config": config,
        "loss_history": loss_history,
    }

    with open(LOG_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

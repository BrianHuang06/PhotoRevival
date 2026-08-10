# -*- coding: utf-8 -*-
"""
LoRA Training for Photo Restoration - Plan B: SwinIR + SDXL Inpainting + LoRA

Pipeline:
  1. Preprocessing: SwinIR processes degraded images -> cached outputs
  2. Training: SDXL Inpainting + LoRA, using SwinIR outputs as UNet reference
  3. Evaluation: SwinIR output -> SDXL Inpainting + LoRA -> compare with GT

Training/Inference Consistency:
  - Training: UNet reference = SwinIR output latent
  - Inference: SDXL Inpainting image = SwinIR output
  - Both use the same SwinIR output as the condition, ensuring consistency
"""

import os
import sys
import gc
import json
import random
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from photo_revival.paths import BASE_MODELS_DIR, CACHE_DIR, CHECKPOINTS_DIR, DATASETS_DIR, THIRD_PARTY_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_training_pairs(clean_dir: str, degraded_dir: str) -> List[Tuple[str, str, str]]:
    clean_files = set(f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    degraded_files = sorted(f for f in os.listdir(degraded_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    pairs = [(os.path.join(degraded_dir, df), os.path.join(clean_dir, df), df)
             for df in degraded_files if df in clean_files]
    log(f"Found {len(pairs)} training pairs")
    return pairs


def get_test_pairs(pairs: List, num: int = 10) -> List:
    step = max(1, len(pairs) // num)
    return [pairs[i] for i in range(0, len(pairs), step)][:num]


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    p.data.copy_(self.shadow[n])


def compute_snr_weight(scheduler, t, gamma):
    if gamma <= 0:
        return 1.0
    snr = (scheduler.alphas_cumprod[t] / (1 - scheduler.alphas_cumprod[t])) ** 2
    return torch.minimum(snr, torch.ones_like(snr) * gamma) / snr


# ======================================================================
# SwinIR Preprocessing
# ======================================================================

def load_swinir_model(weights_path=None, device="cuda"):
    import importlib.util
    import huggingface_hub
    
    swinir_path = os.path.join(THIRD_PARTY_DIR, "DiffBIR", "diffbir", "model", "swinir.py")
    spec = importlib.util.spec_from_file_location("swinir_module", swinir_path)
    swinir_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swinir_mod)
    SwinIR = swinir_mod.SwinIR

    model = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )

    if weights_path and os.path.exists(weights_path):
        log(f"Loading SwinIR weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        log(f"Downloading SwinIR weights from HuggingFace...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        weights_path = huggingface_hub.hf_hub_download(
            repo_id="lxq007/DiffBIR-v2",
            filename="realesrgan_s4_swinir_100k.pth",
            local_dir="weights",
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = list(state_dict.keys())[0]
        if first_key.startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    log("SwinIR model loaded successfully")
    return model


@torch.no_grad()
def run_swinir_inference(model, image_pil, target_size=512, device="cuda"):
    img = image_pil.resize((target_size, target_size), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.amp.autocast('cuda'):
        output = model(img_tensor)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def preprocess_swinir(pairs, cache_dir, weights_path=None, device="cuda"):
    os.makedirs(cache_dir, exist_ok=True)

    swinir_pairs = []
    need_process = []

    for dmg_path, clean_path, fname in pairs:
        swinir_path = os.path.join(cache_dir, fname)
        swinir_pairs.append((swinir_path, clean_path, fname))
        if not os.path.exists(swinir_path):
            need_process.append((dmg_path, swinir_path, fname))

    if not need_process:
        log(f"All {len(pairs)} SwinIR outputs already cached in {cache_dir}")
        return swinir_pairs

    log(f"Preprocessing {len(need_process)} images with SwinIR...")
    model = load_swinir_model(weights_path, device)

    for dmg_path, swinir_path, fname in tqdm(need_process, desc="SwinIR preprocessing"):
        try:
            dmg_img = Image.open(dmg_path).convert("RGB")
            result = run_swinir_inference(model, dmg_img, target_size=512, device=device)
            result.save(swinir_path, "JPEG", quality=95)
        except Exception as e:
            log(f"Error processing {fname}: {e}", "WARN")
            try:
                dmg_img = Image.open(dmg_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                dmg_img.save(swinir_path, "JPEG", quality=95)
            except:
                continue

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"SwinIR preprocessing done. {len(need_process)} images cached in {cache_dir}")

    return swinir_pairs


# ======================================================================
# Evaluation
# ======================================================================

def evaluate(pipe, swinir_test_pairs, prompt, strength_values=[0.15, 0.2, 0.25, 0.3, 0.4], seed=42):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    all_results = {}
    for strength in strength_values:
        results = {"psnr": [], "ssim": []}
        for swinir_path, clean_path, _ in swinir_test_pairs:
            try:
                swinir_img = Image.open(swinir_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                mask = Image.new("L", (512, 512), 255)

                gen = torch.Generator().manual_seed(seed)
                result = pipe(prompt=prompt, image=swinir_img, mask_image=mask,
                              strength=strength, num_inference_steps=25, generator=gen).images[0]
                result = result.resize((512, 512), Image.LANCZOS)

                clean_arr, result_arr = np.array(clean), np.array(result)
                results["psnr"].append(psnr(clean_arr, result_arr))
                results["ssim"].append(ssim(clean_arr, result_arr, channel_axis=2))
            except Exception as e:
                log(f"  Eval error (s={strength}): {e}", "WARN")
                continue

        if results["psnr"]:
            all_results[strength] = {k: float(np.mean(v)) for k, v in results.items()}
            log(f"  strength={strength}: PSNR={all_results[strength]['psnr']:.2f}, SSIM={all_results[strength]['ssim']:.4f}")

    if not all_results:
        return {"psnr": 0.0, "ssim": 0.0, "best_strength": 0.3}

    best_strength = max(all_results, key=lambda s: all_results[s]["psnr"])
    best = all_results[best_strength]
    best["best_strength"] = best_strength
    log(f"  Best strength={best_strength}: PSNR={best['psnr']:.2f}, SSIM={best['ssim']:.4f}")
    return best


# ======================================================================
# Training
# ======================================================================

def train(train_swinir_pairs, test_swinir_pairs, config):
    from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model, PeftModel

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best"), exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log("Loading SDXL Inpainting model...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        config["base_model"], torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
    )

    resume_from = config.get("resume_from", None)
    if resume_from and os.path.exists(resume_from):
        log(f"Resuming from {resume_from}...")
        unet = PeftModel.from_pretrained(pipe.unet, resume_from)
        start_epoch = config.get("start_epoch", 16)
    else:
        unet = get_peft_model(pipe.unet, LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
        ))
        start_epoch = 0

    if config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    log(f"Trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

    unet = unet.to("cuda")
    pipe.vae = pipe.vae.to("cuda").eval()
    for p in pipe.vae.parameters():
        p.requires_grad = False

    if config["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            log("Using 8-bit Adam")
        except:
            optimizer = torch.optim.AdamW(unet.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * len(train_swinir_pairs) // config["grad_acc"]
    from diffusers.optimization import get_scheduler
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=config["warmup"], num_training_steps=total_steps)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    with torch.no_grad():
        emb, _, pooled, _ = pipe.encode_prompt(config["prompt"], device="cpu",
                                                num_images_per_prompt=1, do_classifier_free_guidance=False)
        emb, pooled = emb.cuda(), pooled.cuda()

    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device="cuda")

    ema = EMAModel(unet, decay=config["ema_decay"]) if config["use_ema"] else None

    log(f"Training: {len(train_swinir_pairs)} pairs, {config['epochs']} epochs, starting from epoch {start_epoch + 1}")
    log(f"Pipeline: SwinIR (preprocessed) -> SDXL Inpainting + LoRA")

    best_psnr, patience_cnt = 0, 0
    scaler = torch.amp.GradScaler('cuda')
    global_step = 0

    for epoch in range(start_epoch, config["epochs"]):
        unet.train()
        np.random.shuffle(train_swinir_pairs)
        epoch_loss = 0

        pbar = tqdm(train_swinir_pairs, desc=f"Epoch {epoch + 1}")
        for swinir_path, clean_path, _ in pbar:
            try:
                clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                swinir_out = Image.open(swinir_path).convert("RGB").resize((512, 512), Image.LANCZOS)

                if random.random() < 0.5:
                    clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
                    swinir_out = swinir_out.transpose(Image.FLIP_LEFT_RIGHT)

                clean_t = (torch.from_numpy(np.array(clean)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).cuda().half()
                swinir_t = (torch.from_numpy(np.array(swinir_out)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).cuda().half()

                with torch.no_grad():
                    clean_lat = pipe.vae.encode(clean_t).latent_dist.sample() * 0.13025
                    swinir_lat = pipe.vae.encode(swinir_t).latent_dist.sample() * 0.13025

                noise = torch.randn_like(clean_lat)

                if config["noise_offset"] > 0:
                    noise += config["noise_offset"] * torch.randn(noise.shape[0], noise.shape[1], 1, 1, device="cuda")

                t = torch.randint(0, 1000, (1,), device="cuda")
                noisy = noise_scheduler.add_noise(clean_lat, noise, t)
                mask = torch.ones(1, 1, 64, 64, device="cuda", dtype=torch.float16)

                with torch.amp.autocast('cuda'):
                    pred = unet(torch.cat([noisy, swinir_lat, mask], 1), t,
                                encoder_hidden_states=emb,
                                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample

                    loss = F.mse_loss(pred, noise)

                    if config["snr_gamma"] > 0:
                        snr_w = compute_snr_weight(noise_scheduler, t, config["snr_gamma"])
                        loss = loss * snr_w

                scaler.scale(loss / config["grad_acc"]).backward()
                global_step += 1

                if global_step % config["grad_acc"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(unet)

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            except Exception as e:
                continue

        log(f"Epoch {epoch + 1}: loss={epoch_loss / len(train_swinir_pairs):.4f}")

        if (epoch + 1) % config["eval_every"] == 0:
            save_start_time = time.time()
            if ema:
                backup = {n: p.data.clone() for n, p in unet.named_parameters() if p.requires_grad}
                ema.apply(unet)

            unet = unet.to("cpu")
            torch.cuda.empty_cache()

            unet.save_pretrained(os.path.join(output_dir, f"epoch{epoch + 1}"))
            log(f"Model saved in {time.time() - save_start_time:.2f}s")

            unet = unet.to("cuda")

            eval_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                config["base_model"], torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
            )
            eval_pipe.unet = PeftModel.from_pretrained(eval_pipe.unet, os.path.join(output_dir, f"epoch{epoch + 1}"))
            eval_pipe.enable_model_cpu_offload()

            result = evaluate(eval_pipe, test_swinir_pairs, config["prompt"],
                              strength_values=config.get("eval_strengths", [0.15, 0.2, 0.25, 0.3, 0.4]))
            log(f"Eval: PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.4f}, best_strength={result.get('best_strength', 'N/A')}")

            if result["psnr"] > best_psnr:
                best_psnr = result["psnr"]
                unet = unet.to("cpu")
                torch.cuda.empty_cache()
                unet.save_pretrained(os.path.join(output_dir, "best"))
                unet = unet.to("cuda")
                log("New best model saved!")
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= config["patience"]:
                    log("Early stopping")
                    del eval_pipe
                    break

            del eval_pipe
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except:
                pass
            if ema:
                with torch.no_grad():
                    for n, p in unet.named_parameters():
                        if p.requires_grad and n in backup:
                            p.data.copy_(backup[n])
                del backup
            unet.train()

    unet.save_pretrained(os.path.join(output_dir, "final"))
    log(f"Training done. Best PSNR: {best_psnr:.2f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Training - Plan B: SwinIR + SDXL Inpainting + LoRA")
    parser.add_argument("--base_model", type=str,
                        default=str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1"))
    parser.add_argument("--clean_dir", type=str,
                        default=str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean"))
    parser.add_argument("--degraded_dir", type=str,
                        default=str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded"))
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINTS_DIR / "lora_inpainting_v27"))
    parser.add_argument("--swinir_cache_dir", type=str, default=str(CACHE_DIR / "swinir"))
    parser.add_argument("--swinir_weights", type=str, default=None,
                        help="Path to SwinIR weights (auto-download if not specified)")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)

    parser.add_argument("--noise_offset", type=float, default=0.1)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Starting epoch (used with --resume)")

    args = parser.parse_args()

    config = {
        "base_model": args.base_model,
        "clean_dir": args.clean_dir,
        "degraded_dir": args.degraded_dir,
        "output_dir": args.output_dir,
        "swinir_cache_dir": args.swinir_cache_dir,
        "pipeline": "swinir_sdxl_lora",
        "prompt": "restored old photo, clear, sharp, natural colors, high quality",
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "lr": args.lr,
        "weight_decay": 0.01,
        "grad_acc": args.grad_acc,
        "warmup": args.warmup,
        "patience": args.patience,
        "noise_offset": args.noise_offset,
        "snr_gamma": args.snr_gamma,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "use_8bit_adam": args.use_8bit_adam,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_every": 5,
        "eval_strengths": [0.05, 0.1, 0.15, 0.2, 0.25],
        "resume_from": args.resume,
        "start_epoch": args.start_epoch,
    }

    set_seed(args.seed)

    log("=" * 60)
    log("LoRA Training - Plan B: SwinIR + SDXL Inpainting + LoRA")
    log("=" * 60)
    for k, v in config.items():
        log(f"  {k}: {v}")

    pairs = get_training_pairs(config["clean_dir"], config["degraded_dir"])
    test = get_test_pairs(pairs, 10)
    train_pairs = [p for p in pairs if p not in test]

    log("=" * 60)
    log("Stage 1: SwinIR Preprocessing")
    log("=" * 60)
    train_swinir_pairs = preprocess_swinir(
        train_pairs,
        os.path.join(args.swinir_cache_dir, "train"),
        weights_path=args.swinir_weights,
        device="cuda"
    )
    test_swinir_pairs = preprocess_swinir(
        test,
        os.path.join(args.swinir_cache_dir, "test"),
        weights_path=args.swinir_weights,
        device="cuda"
    )

    log("=" * 60)
    log("Stage 2: SDXL Inpainting + LoRA Training")
    log("=" * 60)
    train(train_swinir_pairs, test_swinir_pairs, config)


if __name__ == "__main__":
    main()

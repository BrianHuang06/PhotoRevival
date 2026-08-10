# -*- coding: utf-8 -*-
"""
Verify trained LoRA model for old photo restoration.
Compare multiple checkpoints and parameters.
"""

import os
import gc
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compute_metrics(img1, img2):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    a1, a2 = np.array(img1), np.array(img2)
    return {
        "psnr": psnr(a1, a2),
        "ssim": ssim(a1, a2, channel_axis=2),
    }


def main():
    base_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    lora_dir = str(CHECKPOINTS_DIR / "lora_inpainting_v26")
    degraded_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    clean_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
    test_output_dir = str(EVALUATIONS_DIR / "verify_lora")
    os.makedirs(test_output_dir, exist_ok=True)

    prompt = "restored old photo, clear, sharp, natural colors, high quality"

    test_files = ["lrp_img10.jpg", "lrp_img119.jpg", "lrp_img147.jpg"]

    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel

    checkpoints = ["epoch25", "epoch35", "epoch45", "epoch50", "final"]
    strengths = [0.3, 0.5, 0.7]
    guidance_scales = [5.0, 7.5]

    all_results = []

    for ckpt in checkpoints:
        ckpt_path = os.path.join(lora_dir, ckpt)
        if not os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")):
            log(f"Skip {ckpt} (no weights)")
            continue

        log(f"=== Testing checkpoint: {ckpt} ===")

        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
        )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, ckpt_path)
        pipe.enable_model_cpu_offload()

        for s in strengths:
            for g in guidance_scales:
                log(f"  strength={s}, guidance={g}")
                scores = {"psnr": [], "ssim": []}

                for fname in test_files:
                    dmg_path = os.path.join(degraded_dir, fname)
                    clean_path = os.path.join(clean_dir, fname)
                    if not os.path.exists(dmg_path) or not os.path.exists(clean_path):
                        continue

                    dmg = Image.open(dmg_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                    clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                    mask = Image.new("L", (512, 512), 255)

                    gen = torch.Generator().manual_seed(42)
                    result = pipe(
                        prompt=prompt,
                        image=dmg,
                        mask_image=mask,
                        strength=s,
                        guidance_scale=g,
                        num_inference_steps=25,
                        generator=gen,
                    ).images[0]
                    result = result.resize((512, 512), Image.LANCZOS)

                    m = compute_metrics(clean, result)
                    scores["psnr"].append(m["psnr"])
                    scores["ssim"].append(m["ssim"])

                    tag = f"{ckpt}_s{s}_g{g}_{fname}"
                    result.save(os.path.join(test_output_dir, f"{tag}"))
                    dmg.save(os.path.join(test_output_dir, f"input_{tag}"))

                if scores["psnr"]:
                    avg_p = np.mean(scores["psnr"])
                    avg_s = np.mean(scores["ssim"])
                    log(f"  => PSNR={avg_p:.2f}, SSIM={avg_s:.4f}")
                    all_results.append({
                        "ckpt": ckpt, "strength": s, "guidance": g,
                        "psnr": avg_p, "ssim": avg_s,
                    })

        del pipe
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

    log("\n" + "=" * 70)
    log("VERIFICATION RESULTS SUMMARY")
    log("=" * 70)
    log(f"{'Checkpoint':<12} {'Strength':<10} {'Guidance':<10} {'PSNR':<10} {'SSIM':<10}")
    log("-" * 52)
    for r in sorted(all_results, key=lambda x: x["psnr"], reverse=True):
        log(f"{r['ckpt']:<12} {r['strength']:<10} {r['guidance']:<10} "
            f"{r['psnr']:<10.2f} {r['ssim']:<10.4f}")

    if all_results:
        best = max(all_results, key=lambda x: x["psnr"])
        log(f"\nBest: {best['ckpt']} s={best['strength']} g={best['guidance']} "
            f"PSNR={best['psnr']:.2f} SSIM={best['ssim']:.4f}")

    log(f"Done! Check {test_output_dir} for images.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Two-stage restoration:
  Stage 1: Repair masked regions with higher strength
  Stage 2: Fine enhancement with lower strength
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_IMAGE = str(DATASETS_DIR / "archive" / "02_Damaged_Testing_Set" / "low-resolution-photographs_img22.jpg")
OUTPUT_DIR = str(EVALUATIONS_DIR / "two_stage")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def two_stage_restoration(stage1_strength=0.6, stage2_strength=0.1):
    print(f"\n{'='*60}")
    print(f"Two-Stage Restoration")
    print(f"Stage 1 (mask repair): strength={stage1_strength}")
    print(f"Stage 2 (fine enhance): strength={stage2_strength}")
    print(f"{'='*60}")
    
    img = cv2.imread(INPUT_IMAGE)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    mask = cv2.imread(str(EVALUATIONS_DIR / "comparison" / "method1_mask.jpg"), cv2.IMREAD_GRAYSCALE)
    print(f"Mask coverage: {np.mean(mask)/255*100:.1f}%")
    
    print("\nLoading SDXL Inpainting + LoRA...")
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel
    
    sdxl_path = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    lora_path = str(CHECKPOINTS_DIR / "lora_inpainting_v28" / "best")
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        sdxl_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.enable_model_cpu_offload()
    
    target_size = 512
    img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
    mask_resized = Image.fromarray(mask).resize((target_size, target_size), Image.LANCZOS)
    
    # Stage 1: Repair masked regions
    print(f"\n[Stage 1] Repairing masked regions (strength={stage1_strength})...")
    prompt1 = "restored old photo, repaired, clean, no scratches, natural texture"
    generator = torch.Generator("cuda").manual_seed(42)
    
    stage1_result = pipe(
        prompt=prompt1,
        image=img_resized,
        mask_image=mask_resized,
        strength=stage1_strength,
        num_inference_steps=25,
        generator=generator,
    ).images[0]
    
    stage1_result.save(os.path.join(OUTPUT_DIR, f"stage1_s{stage1_strength}.jpg"), "JPEG", quality=95)
    print("  Stage 1 done!")
    
    # Stage 2: Fine enhancement on whole image
    print(f"\n[Stage 2] Fine enhancement (strength={stage2_strength})...")
    full_mask = Image.new("L", (target_size, target_size), 255)
    prompt2 = "high quality photo, clear, sharp, natural colors"
    generator = torch.Generator("cuda").manual_seed(42)
    
    stage2_result = pipe(
        prompt=prompt2,
        image=stage1_result,
        mask_image=full_mask,
        strength=stage2_strength,
        num_inference_steps=25,
        generator=generator,
    ).images[0]
    
    stage2_result = stage2_result.resize((w, h), Image.LANCZOS)
    output_name = f"final_s1_{stage1_strength}_s2_{stage2_strength}.jpg"
    stage2_result.save(os.path.join(OUTPUT_DIR, output_name), "JPEG", quality=95)
    print(f"  Saved: {output_name}")
    
    return stage2_result

if __name__ == "__main__":
    configs = [
        (0.5, 0.1),
        (0.6, 0.1),
        (0.7, 0.1),
        (0.6, 0.05),
        (0.6, 0.15),
    ]
    
    for s1, s2 in configs:
        two_stage_restoration(s1, s2)
    
    print("\nDone!")

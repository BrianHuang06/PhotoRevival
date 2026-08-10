# -*- coding: utf-8 -*-
"""
Method 1 with lower strength
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_IMAGE = str(DATASETS_DIR / "archive" / "02_Damaged_Testing_Set" / "low-resolution-photographs_img22.jpg")
OUTPUT_DIR = str(EVALUATIONS_DIR / "comparison")

def run_with_strength(strength):
    print(f"\n{'='*60}")
    print(f"Running with strength={strength}")
    print(f"{'='*60}")
    
    img = cv2.imread(INPUT_IMAGE)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    mask = cv2.imread(os.path.join(OUTPUT_DIR, "method1_mask.jpg"), cv2.IMREAD_GRAYSCALE)
    
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
    
    prompt = "high quality restored old photo, clear, sharp, natural colors"
    generator = torch.Generator("cuda").manual_seed(42)
    
    result = pipe(
        prompt=prompt,
        image=img_resized,
        mask_image=mask_resized,
        strength=strength,
        num_inference_steps=25,
        generator=generator,
    ).images[0]
    
    result = result.resize((w, h), Image.LANCZOS)
    output_path = os.path.join(OUTPUT_DIR, f"method1_strength_{strength}.jpg")
    result.save(output_path, "JPEG", quality=95)
    print(f"Saved: {output_path}")
    
    return result

if __name__ == "__main__":
    for strength in [0.1, 0.15, 0.2, 0.3]:
        run_with_strength(strength)
    
    print("\nDone!")

# -*- coding: utf-8 -*-
"""
Compare two restoration methods
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import shutil
import subprocess

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR, THIRD_PARTY_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_IMAGE = str(DATASETS_DIR / "archive" / "02_Damaged_Testing_Set" / "low-resolution-photographs_img22.jpg")
OUTPUT_DIR = str(EVALUATIONS_DIR / "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def method1_bopbtl_mask_sdxl_lora():
    """
    Method 1: BOPBTL mask detection + SDXL Inpainting + LoRA
    """
    print("\n" + "="*60)
    print("METHOD 1: BOPBTL Mask + SDXL LoRA")
    print("="*60)
    
    img = cv2.imread(INPUT_IMAGE)
    h, w = img.shape[:2]
    print(f"Input size: {w}x{h}")
    
    print("\n[1/3] Detecting scratches with BOPBTL...")
    
    temp_dir = os.path.join(OUTPUT_DIR, "temp_detect_input")
    os.makedirs(temp_dir, exist_ok=True)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.save(os.path.join(temp_dir, "input.png"))
    
    detection_script = THIRD_PARTY_DIR / "Bringing-Old-Photos-Back-to-Life" / "Global" / "detection.py"
    subprocess.run(
        [sys.executable, str(detection_script), "--test_path", temp_dir,
         "--output_dir", os.path.join(OUTPUT_DIR, "method1_detect"),
         "--input_size", "full_size", "--GPU", "0"],
        check=True,
    )
    
    mask_path = os.path.join(OUTPUT_DIR, "method1_detect", "mask", "input.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h))
        cv2.imwrite(os.path.join(OUTPUT_DIR, "method1_mask.jpg"), mask)
        print(f"  Mask coverage: {np.mean(mask)/255*100:.1f}%")
    else:
        print("  Warning: Mask detection failed, using full mask")
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.imwrite(os.path.join(OUTPUT_DIR, "method1_mask.jpg"), mask)
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n[2/3] Loading SDXL Inpainting + LoRA...")
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel
    
    sdxl_path = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    lora_path = str(CHECKPOINTS_DIR / "lora_inpainting_v28" / "best")
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        sdxl_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.enable_model_cpu_offload()
    print("  SDXL loaded!")
    
    print("\n[3/3] Restoring with SDXL...")
    
    target_size = 512
    img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
    mask_resized = Image.fromarray(mask).resize((target_size, target_size), Image.LANCZOS)
    
    prompt = "high quality restored old photo, clear, sharp, natural colors, no scratches"
    generator = torch.Generator("cuda").manual_seed(42)
    
    result = pipe(
        prompt=prompt,
        image=img_resized,
        mask_image=mask_resized,
        strength=0.6,
        num_inference_steps=25,
        generator=generator,
    ).images[0]
    
    result = result.resize((w, h), Image.LANCZOS)
    result.save(os.path.join(OUTPUT_DIR, "method1_result.jpg"), "JPEG", quality=95)
    
    print("  Method 1 done!")
    return result


def method2_bopbtl_full():
    """
    Method 2: BOPBTL full restoration pipeline
    """
    print("\n" + "="*60)
    print("METHOD 2: BOPBTL Full Restoration")
    print("="*60)
    
    temp_input = os.path.join(OUTPUT_DIR, "temp_input")
    os.makedirs(temp_input, exist_ok=True)
    
    img = cv2.imread(INPUT_IMAGE)
    cv2.imwrite(os.path.join(temp_input, "input.jpg"), img)
    
    print("\n[1/2] Running BOPBTL full restoration...")
    
    bopbtl_dir = THIRD_PARTY_DIR / "Bringing-Old-Photos-Back-to-Life"
    subprocess.run(
        [sys.executable, "run.py", "--input_folder", os.path.abspath(temp_input),
         "--output_folder", os.path.join(OUTPUT_DIR, "method2_output"), "--GPU", "0"],
        cwd=bopbtl_dir,
        check=True,
    )
    
    print("\n[2/2] Collecting results...")
    
    result_path = os.path.join(OUTPUT_DIR, "method2_output", "final_output", "input.png")
    if os.path.exists(result_path):
        result = Image.open(result_path)
        result.save(os.path.join(OUTPUT_DIR, "method2_result.jpg"), "JPEG", quality=95)
        print("  Method 2 done!")
    else:
        print("  Warning: BOPBTL full restoration may have failed")
        for root, dirs, files in os.walk(os.path.join(OUTPUT_DIR, "method2_output")):
            for f in files:
                print(f"    Found: {os.path.join(root, f)}")
    
    shutil.rmtree(temp_input, ignore_errors=True)


def create_comparison():
    """Create side-by-side comparison"""
    print("\n" + "="*60)
    print("Creating comparison...")
    print("="*60)
    
    orig = cv2.imread(INPUT_IMAGE)
    h, w = orig.shape[:2]
    
    method1 = cv2.imread(os.path.join(OUTPUT_DIR, "method1_result.jpg"))
    method2 = cv2.imread(os.path.join(OUTPUT_DIR, "method2_result.jpg"))
    
    results = []
    labels = ["Original"]
    results.append(orig)
    
    if method1 is not None:
        method1 = cv2.resize(method1, (w, h))
        results.append(method1)
        labels.append("Method 1: BOPBTL+SDXL")
    
    if method2 is not None:
        method2 = cv2.resize(method2, (w, h))
        results.append(method2)
        labels.append("Method 2: BOPBTL Full")
    
    if len(results) > 1:
        comparison = np.hstack(results)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset = 0
        for i, label in enumerate(labels):
            cv2.putText(comparison, label, (x_offset + 10, 30), font, 0.8, (255, 255, 255), 2)
            x_offset += w
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, "comparison.jpg"), comparison)
        print(f"\nComparison saved to: {OUTPUT_DIR}/comparison.jpg")
    
    mask = cv2.imread(os.path.join(OUTPUT_DIR, "method1_mask.jpg"))
    if mask is not None:
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_comparison = np.hstack([orig, mask_color])
        
        cv2.putText(mask_comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(mask_comparison, "Detected Mask", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_comparison.jpg"), mask_comparison)
        print(f"Mask comparison saved to: {OUTPUT_DIR}/mask_comparison.jpg")


if __name__ == "__main__":
    print("="*60)
    print("Two Methods Comparison")
    print(f"Input: {INPUT_IMAGE}")
    print("="*60)
    
    try:
        method1_bopbtl_mask_sdxl_lora()
    except Exception as e:
        print(f"Method 1 error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        method2_bopbtl_full()
    except Exception as e:
        print(f"Method 2 error: {e}")
        import traceback
        traceback.print_exc()
    
    create_comparison()
    
    print("\n" + "="*60)
    print("Done!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)

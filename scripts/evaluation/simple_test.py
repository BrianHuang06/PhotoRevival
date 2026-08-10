# -*- coding: utf-8 -*-
"""
Simple test for two-stage restoration with visual outputs
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    from photo_revival.two_stage_restoration import TwoStageRestorationPipeline
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    sdxl_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    lora_path = str(CHECKPOINTS_DIR / "lora_inpainting_v28" / "best")
    
    degraded_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    clean_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
    output_dir = str(EVALUATIONS_DIR / "simple_test")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = TwoStageRestorationPipeline(
        sdxl_model_path=sdxl_model,
        lora_path=lora_path,
    )
    pipeline.load()
    
    files = sorted([f for f in os.listdir(degraded_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:3]
    
    print(f"Testing {len(files)} images")
    
    configs = [
        (0.5, 0.05),
        (0.6, 0.05),
        (0.6, 0.1),
    ]
    
    results = {}
    
    for fname in files:
        degraded_path = os.path.join(degraded_dir, fname)
        clean_path = os.path.join(clean_dir, fname)
        
        if not os.path.exists(clean_path):
            continue
        
        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        
        base_name = os.path.splitext(fname)[0]
        
        degraded.save(os.path.join(output_dir, f"{base_name}_input.jpg"), "JPEG", quality=95)
        clean.save(os.path.join(output_dir, f"{base_name}_gt.jpg"), "JPEG", quality=95)
        
        print(f"\nProcessing: {fname}")
        
        for s1, s2 in configs:
            key = f"s1={s1}_s2={s2}"
            if key not in results:
                results[key] = {"psnr": [], "ssim": []}
            
            try:
                output = pipeline.run(
                    degraded,
                    stage1_strength=s1,
                    stage2_strength=s2,
                    use_swinir=True,
                    seed=42,
                )
                
                final = output["final"].resize((512, 512), Image.LANCZOS)
                swinir = output["swinir"]
                mask = output["scratch_mask"]
                stage1 = output["stage1_repaired"]
                
                clean_arr = np.array(clean)
                final_arr = np.array(final)
                
                p = psnr(clean_arr, final_arr)
                s = ssim(clean_arr, final_arr, channel_axis=2)
                
                results[key]["psnr"].append(p)
                results[key]["ssim"].append(s)
                
                print(f"  {key}: PSNR={p:.2f}, SSIM={s:.4f}")
                
                final.save(os.path.join(output_dir, f"{base_name}_{key}_final.jpg"), "JPEG", quality=95)
                
                if s1 == configs[0][0] and s2 == configs[0][1]:
                    swinir.save(os.path.join(output_dir, f"{base_name}_swinir.jpg"), "JPEG", quality=95)
                    mask.save(os.path.join(output_dir, f"{base_name}_mask.jpg"))
                    stage1.save(os.path.join(output_dir, f"{base_name}_stage1.jpg"), "JPEG", quality=95)
                
            except Exception as e:
                print(f"  Error for {key}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for key, metrics in results.items():
        if metrics["psnr"]:
            avg_psnr = np.mean(metrics["psnr"])
            avg_ssim = np.mean(metrics["ssim"])
            print(f"{key}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    if results:
        best_key = max(results, key=lambda k: np.mean(results[k]["psnr"]) if results[k]["psnr"] else 0)
        print(f"\nBest config: {best_key}")
    
    pipeline.cleanup()
    print(f"\nOutput saved to: {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()

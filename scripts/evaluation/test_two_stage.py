# -*- coding: utf-8 -*-
"""
Test Two-Stage Restoration Pipeline

Compare different strength combinations for Stage 1 and Stage 2
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

from photo_revival.paths import BASE_MODELS_DIR, EVALUATIONS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_two_stage(
    input_image: str,
    output_dir: str = str(EVALUATIONS_DIR / "two_stage_test"),
    lora_path: str = None,
    stage1_strengths: list = [0.5, 0.6, 0.7],
    stage2_strengths: list = [0.05, 0.1, 0.15],
):
    """
    Test two-stage restoration with different strength combinations
    """
    from photo_revival.two_stage_restoration import TwoStageRestorationPipeline
    
    os.makedirs(output_dir, exist_ok=True)
    
    sdxl_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    
    pipeline = TwoStageRestorationPipeline(
        sdxl_model_path=sdxl_model,
        lora_path=lora_path,
    )
    pipeline.load()
    
    image = Image.open(input_image).convert("RGB")
    base_name = Path(input_image).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nTesting image: {input_image}")
    print(f"Stage 1 strengths: {stage1_strengths}")
    print(f"Stage 2 strengths: {stage2_strengths}")
    print(f"Total combinations: {len(stage1_strengths) * len(stage2_strengths)}")
    
    results_summary = []
    
    for s1 in stage1_strengths:
        for s2 in stage2_strengths:
            print(f"\n{'='*50}")
            print(f"Testing: Stage1={s1}, Stage2={s2}")
            print(f"{'='*50}")
            
            try:
                results = pipeline.run(
                    image,
                    stage1_strength=s1,
                    stage2_strength=s2,
                    use_swinir=True,
                    seed=42,
                )
                
                output_name = f"{base_name}_s1_{s1}_s2_{s2}.jpg"
                output_path = os.path.join(output_dir, output_name)
                results["final"].save(output_path, "JPEG", quality=95)
                
                mask_name = f"{base_name}_s1_{s1}_s2_{s2}_mask.jpg"
                results["scratch_mask"].save(os.path.join(output_dir, mask_name))
                
                results_summary.append({
                    "stage1": s1,
                    "stage2": s2,
                    "output": output_name,
                })
                
                print(f"Saved: {output_path}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    results["swinir"].save(os.path.join(output_dir, f"{base_name}_swinir.jpg"), "JPEG", quality=95)
    
    print(f"\n{'='*50}")
    print("Test completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")
    
    pipeline.cleanup()
    
    return results_summary


def test_with_gt(
    degraded_dir: str,
    clean_dir: str,
    output_dir: str = str(EVALUATIONS_DIR / "two_stage_test"),
    lora_path: str = None,
    num_images: int = 5,
    stage1_strengths: list = [0.5, 0.6, 0.7],
    stage2_strengths: list = [0.05, 0.1, 0.15],
):
    """
    Test two-stage restoration with ground truth comparison
    """
    from photo_revival.two_stage_restoration import TwoStageRestorationPipeline
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    os.makedirs(output_dir, exist_ok=True)
    
    sdxl_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    
    degraded_files = sorted([f for f in os.listdir(degraded_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:num_images]
    
    print(f"Found {len(degraded_files)} test images")
    
    pipeline = TwoStageRestorationPipeline(
        sdxl_model_path=sdxl_model,
        lora_path=lora_path,
    )
    pipeline.load()
    
    all_results = {}
    
    for s1 in stage1_strengths:
        for s2 in stage2_strengths:
            key = f"s1={s1}_s2={s2}"
            all_results[key] = {"psnr": [], "ssim": []}
    
    for idx, fname in enumerate(degraded_files):
        print(f"\n[{idx+1}/{len(degraded_files)}] Processing: {fname}")
        
        degraded_path = os.path.join(degraded_dir, fname)
        clean_path = os.path.join(clean_dir, fname)
        
        if not os.path.exists(clean_path):
            print(f"  Skipping: no ground truth")
            continue
        
        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        
        for s1 in stage1_strengths:
            for s2 in stage2_strengths:
                key = f"s1={s1}_s2={s2}"
                
                try:
                    results = pipeline.run(
                        degraded,
                        stage1_strength=s1,
                        stage2_strength=s2,
                        use_swinir=True,
                        seed=42,
                    )
                    
                    final = results["final"].resize((512, 512), Image.LANCZOS)
                    
                    clean_arr = np.array(clean)
                    final_arr = np.array(final)
                    
                    p = psnr(clean_arr, final_arr)
                    s = ssim(clean_arr, final_arr, channel_axis=2)
                    
                    all_results[key]["psnr"].append(p)
                    all_results[key]["ssim"].append(s)
                    
                except Exception as e:
                    print(f"  Error for {key}: {e}")
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    summary = []
    for key, metrics in all_results.items():
        if metrics["psnr"]:
            avg_psnr = np.mean(metrics["psnr"])
            avg_ssim = np.mean(metrics["ssim"])
            summary.append((key, avg_psnr, avg_ssim))
            print(f"{key}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    summary.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Best: {summary[0][0]}")
    print(f"PSNR: {summary[0][1]:.2f}")
    print(f"SSIM: {summary[0][2]:.4f}")
    
    pipeline.cleanup()
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Two-Stage Restoration")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch"],
                        help="Test mode: single image or batch with GT")
    parser.add_argument("--input", type=str, help="Input image (single mode) or directory (batch mode)")
    parser.add_argument("--clean_dir", type=str, help="Clean images directory (batch mode)")
    parser.add_argument("--output", type=str, default=str(EVALUATIONS_DIR / "two_stage_test"), help="Output directory")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to test (batch mode)")
    parser.add_argument("--stage1_strengths", type=str, default="0.5,0.6,0.7",
                        help="Comma-separated Stage 1 strengths")
    parser.add_argument("--stage2_strengths", type=str, default="0.05,0.1,0.15",
                        help="Comma-separated Stage 2 strengths")
    
    args = parser.parse_args()
    
    stage1_strengths = [float(x) for x in args.stage1_strengths.split(",")]
    stage2_strengths = [float(x) for x in args.stage2_strengths.split(",")]
    
    if args.mode == "single":
        if not args.input:
            print("Error: --input is required for single mode")
            sys.exit(1)
        test_two_stage(
            args.input,
            args.output,
            args.lora,
            stage1_strengths,
            stage2_strengths,
        )
    
    elif args.mode == "batch":
        if not args.input or not args.clean_dir:
            print("Error: --input and --clean_dir are required for batch mode")
            sys.exit(1)
        test_with_gt(
            args.input,
            args.clean_dir,
            args.output,
            args.lora,
            args.num_images,
            stage1_strengths,
            stage2_strengths,
        )


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from peft import PeftModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, EVALUATIONS_DIR

def main():
    degraded_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    clean_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
    lora_path = str(CHECKPOINTS_DIR / "lora_inpainting_v27" / "best")
    base_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    output_dir = str(EVALUATIONS_DIR / "train_data")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SDXL Inpainting model...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16"
    )
    
    if os.path.exists(lora_path):
        print(f"Loading LoRA from {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    else:
        print(f"Warning: LoRA path not found: {lora_path}")
        print("Using base model without LoRA")
    
    pipe.to("cpu")

    test_images = ["lrp_img10.jpg"]
    
    prompt = "restored old photo, clear, sharp, natural colors, high quality"
    
    results = []
    
    for img_name in test_images:
        degraded_path = os.path.join(degraded_dir, img_name)
        clean_path = os.path.join(clean_dir, img_name)
        
        if not os.path.exists(degraded_path) or not os.path.exists(clean_path):
            print(f"Skipping {img_name} - file not found")
            continue
        
        print(f"\nProcessing {img_name}...")
        
        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        
        orig_w, orig_h = degraded.size
        target_size = 512
        
        degraded_resized = degraded.resize((target_size, target_size), Image.LANCZOS)
        clean_resized = clean.resize((target_size, target_size), Image.LANCZOS)
        
        full_mask = Image.new("L", (target_size, target_size), 255)
        
        strength_values = [0.2]
        
        for strength in strength_values:
            print(f"  Testing strength={strength}...")
            
            generator = torch.Generator().manual_seed(42)
            result = pipe(
                prompt=prompt,
                image=degraded_resized,
                mask_image=full_mask,
                strength=strength,
                num_inference_steps=25,
                generator=generator,
            ).images[0]
            
            result_resized = result.resize((orig_w, orig_h), Image.LANCZOS)
            clean_orig = clean.resize((orig_w, orig_h), Image.LANCZOS)
            
            result_arr = np.array(result_resized)
            clean_arr = np.array(clean_orig)
            
            p = psnr(clean_arr, result_arr)
            s = ssim(clean_arr, result_arr, channel_axis=2)
            
            results.append({
                "image": img_name,
                "strength": strength,
                "psnr": p,
                "ssim": s
            })
            
            print(f"    PSNR: {p:.2f}, SSIM: {s:.4f}")
            
            comparison = Image.new("RGB", (orig_w * 3, orig_h))
            comparison.paste(degraded, (0, 0))
            comparison.paste(result_resized, (orig_w, 0))
            comparison.paste(clean_orig, (orig_w * 2, 0))
            
            comparison.save(os.path.join(output_dir, f"{img_name[:-4]}_s{strength}.jpg"), quality=95)
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"{'Image':<20} {'Strength':<10} {'PSNR':<10} {'SSIM':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['image']:<20} {r['strength']:<10} {r['psnr']:<10.2f} {r['ssim']:<10.4f}")
    
    if results:
        avg_psnr = np.mean([r["psnr"] for r in results])
        avg_ssim = np.mean([r["ssim"] for r in results])
        print("-" * 60)
        print(f"{'Average':<20} {'':<10} {avg_psnr:<10.2f} {avg_ssim:<10.4f}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()

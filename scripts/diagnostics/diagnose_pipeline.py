# -*- coding: utf-8 -*-
"""
Diagnostic script to identify where the problem is in the restoration pipeline.

Tests:
1. SwinIR output vs GT (baseline)
2. SDXL Inpainting (no LoRA) + SwinIR output vs GT
3. SDXL Inpainting + LoRA + SwinIR output vs GT
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from photo_revival.paths import BASE_MODELS_DIR, CACHE_DIR, CHECKPOINTS_DIR, DATASETS_DIR, THIRD_PARTY_DIR, WEIGHTS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_swinir_model(weights_path, device="cuda"):
    import importlib.util
    swinir_path = os.path.join(THIRD_PARTY_DIR, "DiffBIR", "diffbir", "model", "swinir.py")
    spec = importlib.util.spec_from_file_location("swinir_module", swinir_path)
    swinir_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swinir_mod)
    SwinIR = swinir_mod.SwinIR

    model = SwinIR(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8, mlp_ratio=2, sf=8, img_range=1.0,
        upsampler="nearest+conv", resi_connection="1conv",
        unshuffle=True, unshuffle_scale=8,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def run_swinir(model, image, device="cuda"):
    img = image.resize((512, 512), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.amp.autocast('cuda'):
        output = model(img_tensor)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def get_test_pairs(clean_dir, degraded_dir, swinir_cache_dir, num=10):
    clean_files = set(f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    degraded_files = sorted(f for f in os.listdir(degraded_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    pairs = [(os.path.join(degraded_dir, df), os.path.join(clean_dir, df), df)
             for df in degraded_files if df in clean_files]
    step = max(1, len(pairs) // num)
    test_pairs = [pairs[i] for i in range(0, len(pairs), step)][:num]
    
    swinir_pairs = []
    for dmg_path, clean_path, fname in test_pairs:
        swinir_path = os.path.join(swinir_cache_dir, fname)
        if os.path.exists(swinir_path):
            swinir_pairs.append((swinir_path, clean_path, fname))
    return swinir_pairs


def evaluate_swinir_only(swinir_pairs):
    print("\n" + "=" * 60)
    print("Test 1: SwinIR output vs GT (baseline)")
    print("=" * 60)
    
    results = {"psnr": [], "ssim": []}
    for swinir_path, clean_path, fname in tqdm(swinir_pairs, desc="Evaluating SwinIR"):
        swinir_img = Image.open(swinir_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        
        swinir_arr = np.array(swinir_img)
        clean_arr = np.array(clean)
        
        results["psnr"].append(psnr(clean_arr, swinir_arr))
        results["ssim"].append(ssim(clean_arr, swinir_arr, channel_axis=2))
    
    avg_psnr = np.mean(results["psnr"])
    avg_ssim = np.mean(results["ssim"])
    print(f"SwinIR vs GT: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def evaluate_sdxl_no_lora(swinir_pairs, base_model, prompt, strength_values=[0.2, 0.3, 0.4, 0.5]):
    print("\n" + "=" * 60)
    print("Test 2: SDXL Inpainting (no LoRA) + SwinIR vs GT")
    print("=" * 60)
    
    from diffusers import StableDiffusionXLInpaintPipeline
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
    )
    pipe.enable_model_cpu_offload()
    
    results_by_strength = {}
    
    for strength in strength_values:
        print(f"\nTesting strength={strength}...")
        results = {"psnr": [], "ssim": []}
        
        for swinir_path, clean_path, fname in tqdm(swinir_pairs, desc=f"strength={strength}"):
            swinir_img = Image.open(swinir_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            mask = Image.new("L", (512, 512), 255)
            
            gen = torch.Generator().manual_seed(42)
            result = pipe(prompt=prompt, image=swinir_img, mask_image=mask,
                          strength=strength, num_inference_steps=25, generator=gen).images[0]
            result = result.resize((512, 512), Image.LANCZOS)
            
            clean_arr = np.array(clean)
            result_arr = np.array(result)
            
            results["psnr"].append(psnr(clean_arr, result_arr))
            results["ssim"].append(ssim(clean_arr, result_arr, channel_axis=2))
        
        avg_psnr = np.mean(results["psnr"])
        avg_ssim = np.mean(results["ssim"])
        results_by_strength[strength] = (avg_psnr, avg_ssim)
        print(f"  strength={strength}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    del pipe
    torch.cuda.empty_cache()
    return results_by_strength


def evaluate_sdxl_with_lora(swinir_pairs, base_model, lora_path, prompt, strength_values=[0.2, 0.3, 0.4, 0.5]):
    print("\n" + "=" * 60)
    print(f"Test 3: SDXL Inpainting + LoRA ({lora_path}) vs GT")
    print("=" * 60)
    
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.enable_model_cpu_offload()
    
    results_by_strength = {}
    
    for strength in strength_values:
        print(f"\nTesting strength={strength}...")
        results = {"psnr": [], "ssim": []}
        
        for swinir_path, clean_path, fname in tqdm(swinir_pairs, desc=f"strength={strength}"):
            swinir_img = Image.open(swinir_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            mask = Image.new("L", (512, 512), 255)
            
            gen = torch.Generator().manual_seed(42)
            result = pipe(prompt=prompt, image=swinir_img, mask_image=mask,
                          strength=strength, num_inference_steps=25, generator=gen).images[0]
            result = result.resize((512, 512), Image.LANCZOS)
            
            clean_arr = np.array(clean)
            result_arr = np.array(result)
            
            results["psnr"].append(psnr(clean_arr, result_arr))
            results["ssim"].append(ssim(clean_arr, result_arr, channel_axis=2))
        
        avg_psnr = np.mean(results["psnr"])
        avg_ssim = np.mean(results["ssim"])
        results_by_strength[strength] = (avg_psnr, avg_ssim)
        print(f"  strength={strength}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    del pipe
    torch.cuda.empty_cache()
    return results_by_strength


def main():
    base_model = str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1")
    clean_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
    degraded_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    swinir_cache_dir = str(CACHE_DIR / "swinir" / "test")
    swinir_weights = str(WEIGHTS_DIR / "realesrgan_s4_swinir_100k.pth")
    lora_path = str(CHECKPOINTS_DIR / "lora_inpainting_v27" / "epoch5")
    prompt = "restored old photo, clear, sharp, natural colors, high quality"
    
    print("=" * 60)
    print("Diagnostic: Where is the problem?")
    print("=" * 60)
    
    swinir_pairs = get_test_pairs(clean_dir, degraded_dir, swinir_cache_dir, num=10)
    print(f"Found {len(swinir_pairs)} test pairs")
    
    swinir_psnr, swinir_ssim = evaluate_swinir_only(swinir_pairs)
    
    sdxl_results = evaluate_sdxl_no_lora(swinir_pairs, base_model, prompt)
    
    if os.path.exists(lora_path):
        lora_results = evaluate_sdxl_with_lora(swinir_pairs, base_model, lora_path, prompt)
    else:
        print(f"\nLoRA path not found: {lora_path}")
        lora_results = {}
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1. SwinIR vs GT:           PSNR={swinir_psnr:.2f}, SSIM={swinir_ssim:.4f}")
    print("\n2. SDXL (no LoRA) vs GT:")
    for strength, (p, s) in sorted(sdxl_results.items()):
        print(f"   strength={strength}: PSNR={p:.2f}, SSIM={s:.4f}")
    print("\n3. SDXL + LoRA vs GT:")
    for strength, (p, s) in sorted(lora_results.items()):
        print(f"   strength={strength}: PSNR={p:.2f}, SSIM={s:.4f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    best_no_lora = max(sdxl_results.items(), key=lambda x: x[1][0])
    print(f"Best SDXL (no LoRA): strength={best_no_lora[0]}, PSNR={best_no_lora[1][0]:.2f}")
    
    if lora_results:
        best_with_lora = max(lora_results.items(), key=lambda x: x[1][0])
        print(f"Best SDXL + LoRA:   strength={best_with_lora[0]}, PSNR={best_with_lora[1][0]:.2f}")
        
        if best_with_lora[1][0] > best_no_lora[1][0]:
            print("\n=> LoRA is helping! PSNR improved by {:.2f}".format(best_with_lora[1][0] - best_no_lora[1][0]))
        else:
            print("\n=> LoRA is NOT helping. Consider:")
            print("   - More training epochs")
            print("   - Different learning rate")
            print("   - Check if training is actually working")
    
    if swinir_psnr < 22:
        print("\n=> SwinIR output quality is LOW. Consider:")
        print("   - Check if SwinIR weights are correct")
        print("   - Check if input images are properly preprocessed")


if __name__ == "__main__":
    main()

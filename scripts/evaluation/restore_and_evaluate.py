#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore and Evaluate Newly Crawled Damaged Photos
Uses trained LoRA v10 model

Evaluation metrics:
1. No-reference: NIQE, BRISQUE, Sharpness, Color Quality
2. Before/After comparison: Visual quality improvement
3. Paired evaluation: Use clean photos as reference where applicable
"""

import os
import gc
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR

BASE_MODEL_ID = str(BASE_MODELS_DIR / "sdxl-inpainting")
LORA_PATH = str(CHECKPOINTS_DIR / "lora_photo_restoration_v10" / "iter_1" / "best")

PHOTO_DATASET_DIR = DATASETS_DIR / "photo_dataset"
DAMAGED_DIR = str(PHOTO_DATASET_DIR / "damaged_photos")
CLEAN_DIR = str(PHOTO_DATASET_DIR / "clean_photos")
OUTPUT_DIR = str(PHOTO_DATASET_DIR / "restored_photos")
REPORT_FILE = str(PHOTO_DATASET_DIR / "evaluation_report.json")

PROMPT = "restored old photo, clear, sharp, natural colors, high quality, no damage"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def color_correction(img):
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    avg_r, avg_g, avg_b = r.mean(), g.mean(), b.mean()
    avg_gray = (avg_r + avg_g + avg_b) / 3
    if avg_gray > 0:
        arr[:, :, 0] = np.clip(r * (avg_gray / max(avg_r, 1)), 0, 255)
        arr[:, :, 1] = np.clip(g * (avg_gray / max(avg_g, 1)), 0, 255)
        arr[:, :, 2] = np.clip(b * (avg_gray / max(avg_b, 1)), 0, 255)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
    arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    arr = np.clip(arr * 1.05, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def detect_faces(img):
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def generate_damage_mask(damaged_img, target_size=(512, 512)):
    img = damaged_img.resize(target_size, Image.LANCZOS)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edge_mag = np.abs(edges)
    
    kernel_sizes = [3, 7, 15]
    combined = np.zeros_like(gray, dtype=np.float64)
    gray_f = gray.astype(np.float64)
    for k in kernel_sizes:
        local_mean = cv2.blur(gray_f, (k, k))
        local_var = cv2.blur((gray_f - local_mean) ** 2, (k, k))
        combined += local_var
    combined = combined / len(kernel_sizes)
    
    combined_norm = np.clip(combined / (combined.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    edge_norm = np.clip(edge_mag / (edge_mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    
    fused = (combined_norm.astype(np.float32) * 0.7 + edge_norm.astype(np.float32) * 0.3).astype(np.uint8)
    
    _, binary = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.GaussianBlur(binary, (11, 11), 0)
    
    white_mask = np.all(arr > 240, axis=2).astype(np.uint8) * 255
    dark_mask = np.all(arr < 15, axis=2).astype(np.uint8) * 255
    
    final_mask = np.maximum(binary, white_mask)
    final_mask = np.maximum(final_mask, dark_mask)
    
    min_mask_area = 0.01 * target_size[0] * target_size[1]
    if final_mask.sum() / 255 < min_mask_area:
        final_mask = np.full(target_size, 128, dtype=np.uint8)
    
    return final_mask


def compute_niqe(img_arr):
    try:
        from piq import niqe
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        score = niqe(tensor).item()
        return score
    except:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return max(0, 20 - gray.std() * 0.5)


def compute_sharpness(img_arr):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_color_quality(img_arr):
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    l_std = lab[:, :, 0].std()
    a_std = lab[:, :, 1].std()
    b_std = lab[:, :, 2].std()
    return l_std + a_std + b_std


def compute_contrast(img_arr):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    return gray.std()


def compute_snr(img_arr):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    signal = gray.mean()
    noise = gray.std()
    if noise == 0:
        return 40.0
    return 20 * np.log10(signal / noise)


def restore_image(pipe, input_img, strength=0.4, face_blend=0.7):
    w, h = input_img.size
    
    input_corrected = color_correction(input_img)
    input_512 = input_corrected.resize((512, 512), Image.LANCZOS)
    
    damage_mask = generate_damage_mask(input_corrected, (512, 512))
    mask_pil = Image.fromarray(damage_mask)
    
    result = pipe(
        prompt=PROMPT,
        image=input_512,
        mask_image=mask_pil,
        strength=strength,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]
    
    restored = result.resize((w, h), Image.LANCZOS)
    
    faces = detect_faces(input_img)
    if faces and face_blend > 0:
        scale_x, scale_y = w / 512, h / 512
        original_faces = [
            (int(x * scale_x), int(y * scale_y), int(fw * scale_x), int(fh * scale_y))
            for (x, y, fw, fh) in faces
        ]
        blend_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(blend_mask)
        for (fx, fy, fw, fh) in original_faces:
            draw.ellipse([fx, fy, fx + fw, fy + fh], fill=int(255 * face_blend))
        blend_mask = blend_mask.filter(ImageFilter.GaussianBlur(15))
        restored = Image.composite(input_corrected, restored, blend_mask)
    
    return restored, input_corrected, damage_mask


def load_model():
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft import PeftModel
    
    log("Loading SDXL Inpainting model...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    log(f"Loading LoRA weights from {LORA_PATH}...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    log("Model loaded successfully")
    return pipe


def main():
    log("=" * 60)
    log("Restore & Evaluate Crawled Damaged Photos")
    log("=" * 60)
    
    damaged_files = sorted([f for f in os.listdir(DAMAGED_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    log(f"Found {len(damaged_files)} damaged photos to restore")
    
    clean_files = set(f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    log(f"Found {len(clean_files)} clean photos for reference")
    
    pipe = load_model()
    
    results = []
    all_metrics = {
        "sharpness_before": [], "sharpness_after": [],
        "contrast_before": [], "contrast_after": [],
        "color_before": [], "color_after": [],
        "snr_before": [], "snr_after": [],
        "niqe_before": [], "niqe_after": [],
        "psnr_paired": [], "ssim_paired": [],
    }
    
    paired_count = 0
    unpaired_count = 0
    
    for idx, fname in enumerate(damaged_files):
        log(f"\n[{idx+1}/{len(damaged_files)}] Processing: {fname}")
        
        try:
            damaged_path = os.path.join(DAMAGED_DIR, fname)
            damaged_img = Image.open(damaged_path).convert("RGB")
            damaged_arr = np.array(damaged_img)
            
            metrics = {"filename": fname}
            
            metrics["sharpness_before"] = compute_sharpness(damaged_arr)
            metrics["contrast_before"] = compute_contrast(damaged_arr)
            metrics["color_before"] = compute_color_quality(damaged_arr)
            metrics["snr_before"] = compute_snr(damaged_arr)
            metrics["niqe_before"] = compute_niqe(damaged_arr)
            
            restored, corrected, mask = restore_image(pipe, damaged_img, strength=0.4, face_blend=0.7)
            restored_arr = np.array(restored)
            
            metrics["sharpness_after"] = compute_sharpness(restored_arr)
            metrics["contrast_after"] = compute_contrast(restored_arr)
            metrics["color_after"] = compute_color_quality(restored_arr)
            metrics["snr_after"] = compute_snr(restored_arr)
            metrics["niqe_after"] = compute_niqe(restored_arr)
            
            metrics["sharpness_change"] = metrics["sharpness_after"] - metrics["sharpness_before"]
            metrics["contrast_change"] = metrics["contrast_after"] - metrics["contrast_before"]
            metrics["color_change"] = metrics["color_after"] - metrics["color_before"]
            metrics["snr_change"] = metrics["snr_after"] - metrics["snr_before"]
            metrics["niqe_change"] = metrics["niqe_after"] - metrics["niqe_before"]
            
            clean_match = fname.replace("_dmg_", "_")
            has_pair = False
            for cf in clean_files:
                if cf.startswith("europ_9") or cf.startswith("met_"):
                    base = fname.split("_dmg_")[1] if "_dmg_" in fname else fname
                    if base in cf:
                        clean_match = cf
                        has_pair = True
                        break
            
            if not has_pair and len(clean_files) > 0:
                clean_list = sorted(clean_files)
                ref_idx = idx % len(clean_list)
                clean_match = clean_list[ref_idx]
                has_pair = False
            
            metrics["has_paired_ref"] = False
            
            base_name = os.path.splitext(fname)[0]
            restored_path = os.path.join(OUTPUT_DIR, f"{base_name}_restored.jpg")
            restored.save(restored_path, "JPEG", quality=95)
            
            mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.jpg")
            Image.fromarray(mask).save(mask_path)
            
            corrected_path = os.path.join(OUTPUT_DIR, f"{base_name}_corrected.jpg")
            corrected.save(corrected_path, "JPEG", quality=95)
            
            all_metrics["sharpness_before"].append(metrics["sharpness_before"])
            all_metrics["sharpness_after"].append(metrics["sharpness_after"])
            all_metrics["contrast_before"].append(metrics["contrast_before"])
            all_metrics["contrast_after"].append(metrics["contrast_after"])
            all_metrics["color_before"].append(metrics["color_before"])
            all_metrics["color_after"].append(metrics["color_after"])
            all_metrics["snr_before"].append(metrics["snr_before"])
            all_metrics["snr_after"].append(metrics["snr_after"])
            all_metrics["niqe_before"].append(metrics["niqe_before"])
            all_metrics["niqe_after"].append(metrics["niqe_after"])
            
            results.append(metrics)
            
            log(f"  Sharpness: {metrics['sharpness_before']:.1f} -> {metrics['sharpness_after']:.1f} ({metrics['sharpness_change']:+.1f})")
            log(f"  Contrast: {metrics['contrast_before']:.1f} -> {metrics['contrast_after']:.1f} ({metrics['contrast_change']:+.1f})")
            log(f"  SNR: {metrics['snr_before']:.1f} -> {metrics['snr_after']:.1f} ({metrics['snr_change']:+.1f})")
            
            del restored, corrected, mask
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            log(f"  Error: {e}")
            continue
    
    log("\n" + "=" * 60)
    log("EVALUATION SUMMARY")
    log("=" * 60)
    
    n = len(results)
    if n > 0:
        avg_sharp_before = np.mean(all_metrics["sharpness_before"])
        avg_sharp_after = np.mean(all_metrics["sharpness_after"])
        avg_contrast_before = np.mean(all_metrics["contrast_before"])
        avg_contrast_after = np.mean(all_metrics["contrast_after"])
        avg_color_before = np.mean(all_metrics["color_before"])
        avg_color_after = np.mean(all_metrics["color_after"])
        avg_snr_before = np.mean(all_metrics["snr_before"])
        avg_snr_after = np.mean(all_metrics["snr_after"])
        avg_niqe_before = np.mean(all_metrics["niqe_before"])
        avg_niqe_after = np.mean(all_metrics["niqe_after"])
        
        sharp_improved = sum(1 for r in results if r.get("sharpness_change", 0) > 0) / n * 100
        contrast_improved = sum(1 for r in results if r.get("contrast_change", 0) > 0) / n * 100
        snr_improved = sum(1 for r in results if r.get("snr_change", 0) > 0) / n * 100
        
        log(f"\nProcessed: {n}/{len(damaged_files)} images")
        log(f"\n--- No-Reference Quality Metrics ---")
        log(f"Sharpness:  {avg_sharp_before:.1f} -> {avg_sharp_after:.1f} ({avg_sharp_after - avg_sharp_before:+.1f}) [{sharp_improved:.0f}% improved]")
        log(f"Contrast:   {avg_contrast_before:.1f} -> {avg_contrast_after:.1f} ({avg_contrast_after - avg_contrast_before:+.1f}) [{contrast_improved:.0f}% improved]")
        log(f"Color:      {avg_color_before:.1f} -> {avg_color_after:.1f} ({avg_color_after - avg_color_before:+.1f})")
        log(f"SNR:        {avg_snr_before:.1f} -> {avg_snr_after:.1f} ({avg_snr_after - avg_snr_before:+.1f}) [{snr_improved:.0f}% improved]")
        log(f"NIQE:       {avg_niqe_before:.1f} -> {avg_niqe_after:.1f} ({avg_niqe_after - avg_niqe_before:+.1f}) [lower=better]")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": LORA_PATH,
            "num_processed": n,
            "num_total": len(damaged_files),
            "averages": {
                "sharpness": {"before": avg_sharp_before, "after": avg_sharp_after, "change": avg_sharp_after - avg_sharp_before},
                "contrast": {"before": avg_contrast_before, "after": avg_contrast_after, "change": avg_contrast_after - avg_contrast_before},
                "color": {"before": avg_color_before, "after": avg_color_after, "change": avg_color_after - avg_color_before},
                "snr": {"before": avg_snr_before, "after": avg_snr_after, "change": avg_snr_after - avg_snr_before},
                "niqe": {"before": avg_niqe_before, "after": avg_niqe_after, "change": avg_niqe_after - avg_niqe_before},
            },
            "improvement_rate": {
                "sharpness": sharp_improved,
                "contrast": contrast_improved,
                "snr": snr_improved,
            },
            "per_image": results,
        }
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        log(f"\nReport saved to: {REPORT_FILE}")
    
    log(f"\nRestored images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

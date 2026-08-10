# -*- coding: utf-8 -*-
"""
Hybrid Restoration Pipeline: DiffBIR Stage 1 + SDXL Inpainting + LoRA

Pipeline Flow:
  Input Image -> Stage 1 (SwinIR/BSRNet) -> Stage 2 (SDXL Inpainting + LoRA) -> Output

Advantages:
  1. Stage 1 removes degradation (blur, noise, compression artifacts)
  2. Stage 2 adds realistic details via LoRA fine-tuned model
  3. More controllable than full DiffBIR pipeline
  4. Faster inference (single forward pass for Stage 1)
"""

import os
import sys
import gc
import json
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR, THIRD_PARTY_DIR

if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

try:
    import cv2
except ImportError:
    print("Warning: cv2 not found, some features may be disabled")
    cv2 = None

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Warning: skimage not found, evaluation metrics disabled")
    psnr, ssim = None, None


class HybridRestorationPipeline:
    """
    Hybrid pipeline combining DiffBIR Stage 1 and SDXL Inpainting + LoRA
    """
    
    def __init__(
        self,
        sdxl_model_path: str,
        lora_path: Optional[str] = None,
        swinir_model_path: Optional[str] = None,
        device: str = "cuda",
        use_cpu_offload: bool = True,
    ):
        self.device = device
        self.use_cpu_offload = use_cpu_offload
        self.sdxl_model_path = sdxl_model_path
        self.lora_path = lora_path
        self.swinir_model_path = swinir_model_path
        
        self.stage1_model = None
        self.stage2_pipe = None
        
        print(f"[HybridPipeline] Initializing on {device}...")
        
    def load_stage1(self, model_type: str = "swinir"):
        """Load Stage 1: SwinIR or BSRNet for degradation removal"""
        print(f"[Stage 1] Loading {model_type} model...")
        
        if model_type == "swinir":
            from DiffBIR.diffbir.model import SwinIR
            from DiffBIR.diffbir.utils.common import load_model_from_url
            
            SWINIR_MODEL_URL = "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth"

            self.stage1_model = SwinIR(
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
            
            if self.swinir_model_path and os.path.exists(self.swinir_model_path):
                state_dict = torch.load(self.swinir_model_path, map_location="cpu")
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    SWINIR_MODEL_URL, 
                    map_location="cpu",
                    progress=True
                )
            
            self.stage1_model.load_state_dict(state_dict, strict=True)
            self.stage1_model.eval().to(self.device)
            print(f"[Stage 1] SwinIR loaded successfully")
            
        elif model_type == "bsrnet":
            try:
                from DiffBIR.diffbir.model import RRDBNet
                from basicsr.archs.rrdbnet_arch import RRDBNet as BSRNet
                
                self.stage1_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                self.stage1_model.eval().to(self.device)
                print(f"[Stage 1] BSRNet loaded successfully")
            except Exception as e:
                print(f"[Stage 1] Failed to load BSRNet: {e}")
                print("[Stage 1] Falling back to SwinIR...")
                return self.load_stage1("swinir")
        
        return self
    
    def load_stage2(self):
        """Load Stage 2: SDXL Inpainting + LoRA"""
        print(f"[Stage 2] Loading SDXL Inpainting model from {self.sdxl_model_path}...")
        
        from diffusers import StableDiffusionXLInpaintPipeline
        
        self.stage2_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.sdxl_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"[Stage 2] Loading LoRA weights from {self.lora_path}...")
            try:
                from peft import PeftModel
                self.stage2_pipe.unet = PeftModel.from_pretrained(
                    self.stage2_pipe.unet, 
                    self.lora_path
                )
                print(f"[Stage 2] LoRA weights loaded successfully")
            except Exception as e:
                print(f"[Stage 2] Warning: Failed to load LoRA: {e}")
        
        if self.use_cpu_offload:
            self.stage2_pipe.enable_model_cpu_offload()
        else:
            self.stage2_pipe.to(self.device)
        
        print(f"[Stage 2] SDXL Inpainting loaded successfully")
        return self
    
    def load(self, stage1_type: str = "swinir"):
        """Load both stages"""
        self.load_stage1(stage1_type)
        self.load_stage2()
        print("[HybridPipeline] All models loaded!")
        return self
    
    @torch.no_grad()
    def run_stage1(
        self,
        image: Image.Image,
        tile_size: int = 512,
        tile_stride: int = 256,
    ) -> Image.Image:
        """
        Run Stage 1: Degradation removal using SwinIR/BSRNet
        
        Args:
            image: Input PIL Image
            tile_size: Tile size for tiled inference (reduces VRAM)
            tile_stride: Stride between tiles
            
        Returns:
            Cleaned PIL Image (4x upsampled)
        """
        if self.stage1_model is None:
            raise RuntimeError("Stage 1 model not loaded. Call load_stage1() first.")
        
        w, h = image.size
        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        min_size = min(w, h)
        if min_size < 512:
            scale = 512 / min_size
            new_w, new_h = int(w * scale), int(h * scale)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(new_h, new_w), mode="bicubic", antialias=True
            )
        
        _, _, h, w = img_tensor.shape
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        
        print(f"[Stage 1] Processing {w}x{h} image...")
        
        with torch.cuda.amp.autocast():
            output = self.stage1_model(img_tensor)
        
        output = output[:, :, :h, :w]
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(output)
    
    def generate_adaptive_mask(
        self,
        image: Image.Image,
        mask_type: str = "auto",
        threshold: float = 0.3,
    ) -> Image.Image:
        """
        Generate mask for Stage 2 inpainting
        
        Args:
            image: Input image (Stage 1 output)
            mask_type: "auto" (detect damage), "full" (full image), "edges" (edge regions)
            threshold: Threshold for auto detection
            
        Returns:
            Mask PIL Image (L mode, 255=inpainted, 0=kept)
        """
        w, h = image.size
        target_size = (512, 512)
        
        if mask_type == "full":
            mask = Image.new("L", target_size, 255)
            return mask
        
        img_resized = image.resize(target_size, Image.LANCZOS)
        arr = np.array(img_resized)
        
        if mask_type == "auto":
            if cv2 is None:
                print("[Warning] cv2 not available, using full mask")
                return Image.new("L", target_size, 255)
            
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Laplacian(blurred, cv2.CV_64F)
            edge_mag = np.abs(edges)
            
            local_var = np.zeros_like(gray, dtype=np.float64)
            gray_f = gray.astype(np.float64)
            for k in [3, 7, 15]:
                local_mean = cv2.blur(gray_f, (k, k))
                local_var += cv2.blur((gray_f - local_mean) ** 2, (k, k))
            local_var /= 3
            
            fused = local_var * 0.7 + edge_mag * 0.3
            fused_norm = (fused / (fused.max() + 1e-6) * 255).astype(np.uint8)
            
            _, binary = cv2.threshold(fused_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.dilate(binary, kernel, iterations=2)
            binary = cv2.GaussianBlur(binary, (11, 11), 0)
            
            white_mask = np.all(arr > 240, axis=2).astype(np.uint8) * 255
            dark_mask = np.all(arr < 15, axis=2).astype(np.uint8) * 255
            final_mask = np.maximum(binary, white_mask)
            final_mask = np.maximum(final_mask, dark_mask)
            
            min_mask_area = 0.01 * target_size[0] * target_size[1]
            if final_mask.sum() / 255 < min_mask_area:
                final_mask = np.full(target_size, int(128 * threshold), dtype=np.uint8)
            
            return Image.fromarray(final_mask)
        
        elif mask_type == "edges":
            if cv2 is None:
                return Image.new("L", target_size, 128)
            
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=3)
            edges = cv2.GaussianBlur(edges, (11, 11), 0)
            return Image.fromarray(edges)
        
        return Image.new("L", target_size, 128)
    
    def run_stage2(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        prompt: str = "high quality restored photo, clear, sharp, natural colors",
        negative_prompt: str = "blurry, low quality, artifacts, noise, damage",
        strength: float = 0.35,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: int = 42,
    ) -> Image.Image:
        """
        Run Stage 2: SDXL Inpainting + LoRA for detail enhancement
        
        Args:
            image: Input image (Stage 1 output)
            mask: Inpainting mask (auto-generated if None)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            strength: Inpainting strength (0.0-1.0, lower = more faithful to input)
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            
        Returns:
            Restored PIL Image
        """
        if self.stage2_pipe is None:
            raise RuntimeError("Stage 2 model not loaded. Call load_stage2() first.")
        
        original_size = image.size
        input_512 = image.resize((512, 512), Image.LANCZOS)
        
        if mask is None:
            mask = self.generate_adaptive_mask(input_512, mask_type="auto")
        elif mask.size != (512, 512):
            mask = mask.resize((512, 512), Image.LANCZOS)
        
        print(f"[Stage 2] Running inpainting (strength={strength}, steps={num_inference_steps})...")
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        result = self.stage2_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_512,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        result = result.resize(original_size, Image.LANCZOS)
        
        return result
    
    def blend_with_original(
        self,
        original: Image.Image,
        restored: Image.Image,
        blend_factor: float = 0.7,
    ) -> Image.Image:
        """
        Blend restored result with original for more natural output
        
        Args:
            original: Original input image
            restored: Restored image from pipeline
            blend_factor: 0.0 = all original, 1.0 = all restored
            
        Returns:
            Blended PIL Image
        """
        if blend_factor >= 1.0:
            return restored
        if blend_factor <= 0.0:
            return original
        
        original = original.resize(restored.size, Image.LANCZOS)
        return Image.blend(original, restored, blend_factor)
    
    def run(
        self,
        image: Image.Image,
        stage1_enabled: bool = True,
        mask_type: str = "auto",
        strength: float = 0.35,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        blend_factor: float = 0.8,
        prompt: str = "high quality restored photo, clear, sharp, natural colors",
        negative_prompt: str = "blurry, low quality, artifacts, noise, damage",
        seed: int = 42,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Run full hybrid pipeline
        
        Args:
            image: Input PIL Image
            stage1_enabled: Whether to run Stage 1 (disable for direct SDXL)
            mask_type: "auto", "full", or "edges"
            strength: Inpainting strength
            guidance_scale: CFG scale
            num_inference_steps: Denoising steps
            blend_factor: Blend factor with original
            prompt: Positive prompt
            negative_prompt: Negative prompt
            seed: Random seed
            
        Returns:
            Tuple of (final_result, stage1_output, stage2_mask)
        """
        print(f"\n[HybridPipeline] Processing image: {image.size}")
        
        original = image.copy()
        
        if stage1_enabled and self.stage1_model is not None:
            stage1_output = self.run_stage1(image)
            print(f"[Stage 1] Output size: {stage1_output.size}")
        else:
            stage1_output = image.copy()
            print("[Stage 1] Skipped")
        
        mask = self.generate_adaptive_mask(stage1_output, mask_type=mask_type)
        
        stage2_output = self.run_stage2(
            stage1_output,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        if blend_factor < 1.0:
            final_output = self.blend_with_original(original, stage2_output, blend_factor)
        else:
            final_output = stage2_output
        
        print(f"[HybridPipeline] Done!")
        
        return final_output, stage1_output, mask
    
    def cleanup(self):
        """Free GPU memory"""
        if self.stage1_model is not None:
            del self.stage1_model
            self.stage1_model = None
        if self.stage2_pipe is not None:
            del self.stage2_pipe
            self.stage2_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[HybridPipeline] Memory cleaned up")


def batch_process(
    input_dir: str,
    output_dir: str,
    pipeline: HybridRestorationPipeline,
    strength: float = 0.35,
    mask_type: str = "auto",
    extensions: Tuple[str] = (".jpg", ".jpeg", ".png", ".bmp"),
):
    """
    Batch process images in a directory
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        pipeline: HybridRestorationPipeline instance
        strength: Inpainting strength
        mask_type: Mask type for inpainting
        extensions: Supported file extensions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = []
    for ext in extensions:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    files = sorted(set(files))
    print(f"[Batch] Found {len(files)} images to process")
    
    results = []
    
    for idx, file_path in enumerate(files):
        print(f"\n[Batch] [{idx+1}/{len(files)}] Processing: {file_path.name}")
        
        try:
            image = Image.open(file_path).convert("RGB")
            
            final, stage1, mask = pipeline.run(
                image,
                strength=strength,
                mask_type=mask_type,
            )
            
            base_name = file_path.stem
            final_path = os.path.join(output_dir, f"{base_name}_restored.jpg")
            stage1_path = os.path.join(output_dir, f"{base_name}_stage1.jpg")
            mask_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
            
            final.save(final_path, "JPEG", quality=95)
            stage1.save(stage1_path, "JPEG", quality=95)
            mask.save(mask_path)
            
            results.append({
                "input": str(file_path),
                "output": final_path,
                "stage1": stage1_path,
                "mask": mask_path,
                "status": "success",
            })
            
        except Exception as e:
            print(f"[Batch] Error processing {file_path.name}: {e}")
            results.append({
                "input": str(file_path),
                "status": "error",
                "error": str(e),
            })
    
    report_path = os.path.join(output_dir, "batch_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(files),
            "success": sum(1 for r in results if r["status"] == "success"),
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n[Batch] Completed! Report saved to: {report_path}")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Restoration Pipeline: DiffBIR Stage 1 + SDXL Inpainting + LoRA")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument("--sdxl_model", type=str, default=str(BASE_MODELS_DIR / "sdxl-inpainting"), help="Path to SDXL Inpainting model")
    parser.add_argument("--lora", type=str, default=str(CHECKPOINTS_DIR / "lora_inpainting_v25" / "final"), help="Path to LoRA weights")
    parser.add_argument("--swinir", type=str, default=None, help="Path to SwinIR weights (optional)")
    parser.add_argument("--stage1_type", type=str, default="swinir", choices=["swinir", "bsrnet"], help="Stage 1 model type")
    parser.add_argument("--strength", type=float, default=0.35, help="Inpainting strength (0.0-1.0)")
    parser.add_argument("--mask_type", type=str, default="auto", choices=["auto", "full", "edges"], help="Mask generation type")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--blend", type=float, default=0.8, help="Blend factor with original (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 (use SDXL only)")
    
    args = parser.parse_args()
    
    pipeline = HybridRestorationPipeline(
        sdxl_model_path=args.sdxl_model,
        lora_path=args.lora,
        swinir_model_path=args.swinir,
        device=args.device,
    )
    
    if not args.skip_stage1:
        pipeline.load(args.stage1_type)
    else:
        pipeline.load_stage2()
    
    if os.path.isfile(args.input):
        image = Image.open(args.input).convert("RGB")
        
        final, stage1, mask = pipeline.run(
            image,
            stage1_enabled=not args.skip_stage1,
            mask_type=args.mask_type,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            blend_factor=args.blend,
            seed=args.seed,
        )
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        final.save(args.output, "JPEG", quality=95)
        
        base, ext = os.path.splitext(args.output)
        stage1.save(f"{base}_stage1.jpg", "JPEG", quality=95)
        mask.save(f"{base}_mask.jpg")
        
        print(f"\nResults saved to:")
        print(f"  Final: {args.output}")
        print(f"  Stage 1: {base}_stage1.jpg")
        print(f"  Mask: {base}_mask.jpg")
        
    elif os.path.isdir(args.input):
        batch_process(
            args.input,
            args.output,
            pipeline,
            strength=args.strength,
            mask_type=args.mask_type,
        )
    
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    pipeline.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()

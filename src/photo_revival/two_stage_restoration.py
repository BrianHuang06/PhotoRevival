# -*- coding: utf-8 -*-
"""
Two-Stage Photo Restoration Pipeline

Stage 1: Scratch/Damage Repair
  - Auto-detect scratches, cracks, and damaged regions
  - Generate precise mask for damaged areas
  - Use higher strength (0.5-0.7) to repair only masked regions

Stage 2: Fine Enhancement
  - Use full mask for global processing
  - Use lower strength (0.05-0.15) for subtle enhancement
  - Improve clarity and sharpness while preserving details
"""

import os
import sys
import gc
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from photo_revival.paths import BASE_MODELS_DIR, THIRD_PARTY_DIR

try:
    import cv2
except ImportError:
    print("Warning: cv2 not found, install with: pip install opencv-python")
    cv2 = None


class ScratchDetector:
    """
    Detect scratches, cracks, and damaged regions in old photos
    Improved version with multi-scale detection
    """
    
    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect scratches and damage in image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Binary mask (H, W), 255 = damaged, 0 = clean
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required for scratch detection")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        mask = np.maximum(mask, self._detect_thin_scratches(gray))
        mask = np.maximum(mask, self._detect_local_defects(gray))
        mask = np.maximum(mask, self._detect_bright_spots(gray))
        mask = np.maximum(mask, self._detect_dark_spots(gray))
        
        mask = self._clean_mask(mask)
        
        return mask
    
    def _detect_thin_scratches(self, gray: np.ndarray) -> np.ndarray:
        """Detect thin linear scratches using multi-scale morphological operations"""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        scratches = np.zeros_like(gray)
        
        for kernel_len in [15, 21, 31]:
            for angle in [0, 90]:
                if angle == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
                
                blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
                
                thresh = 8 + int(15 * self.sensitivity)
                _, binary = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
                
                scratches = np.maximum(scratches, binary)
        
        kernel_diag = np.eye(11, dtype=np.uint8)
        kernel_anti = np.fliplr(kernel_diag)
        
        for k in [kernel_diag, kernel_anti]:
            blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, k)
            _, binary = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
            scratches = np.maximum(scratches, binary)
        
        return scratches
    
    def _detect_local_defects(self, gray: np.ndarray) -> np.ndarray:
        """Detect local defects using adaptive thresholding"""
        h, w = gray.shape
        block_size = 31
        
        mean = cv2.blur(gray.astype(np.float32), (block_size, block_size))
        diff = np.abs(gray.astype(np.float32) - mean)
        
        local_std = np.sqrt(cv2.blur((gray.astype(np.float32) - mean)**2, (block_size, block_size)))
        
        anomaly = np.zeros_like(gray, dtype=np.float32)
        
        high_contrast = (diff > 30) & (local_std < 20)
        anomaly[high_contrast] = 255
        
        return anomaly.astype(np.uint8)
    
    def _detect_bright_spots(self, gray: np.ndarray) -> np.ndarray:
        """Detect bright spots and white scratches"""
        bright = (gray > 250).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 500:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        
        return result
    
    def _detect_dark_spots(self, gray: np.ndarray) -> np.ndarray:
        """Detect dark spots and black scratches"""
        dark = (gray < 5).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        
        return dark
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up the mask"""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(result, kernel_dilate, iterations=1)
        
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result


class TwoStageRestorationPipeline:
    """
    Two-stage photo restoration pipeline
    
    Stage 1: Repair scratches and damaged regions
    Stage 2: Enhance overall image quality
    """
    
    def __init__(
        self,
        sdxl_model_path: str,
        lora_path: Optional[str] = None,
        swinir_model_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("[WARNING] CUDA not available, using CPU (will be slow)")
        
        self.device = device
        self.sdxl_model_path = sdxl_model_path
        self.lora_path = lora_path
        self.swinir_model_path = swinir_model_path
        
        self.swinir_model = None
        self.sdxl_pipe = None
        self.scratch_detector = ScratchDetector()
        
        print(f"[TwoStagePipeline] Initializing on {device}...")
    
    def load_swinir(self):
        """Load SwinIR model for degradation removal"""
        import importlib.util
        
        swinir_path = os.path.join(
        THIRD_PARTY_DIR,
        "DiffBIR",
            "diffbir",
            "model",
            "swinir.py",
        )
        spec = importlib.util.spec_from_file_location("swinir_module", swinir_path)
        swinir_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(swinir_mod)
        SwinIR = swinir_mod.SwinIR
        
        self.swinir_model = SwinIR(
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
            state_dict = torch.load(self.swinir_model_path, map_location="cpu", weights_only=True)
        else:
            import huggingface_hub
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            weights_path = huggingface_hub.hf_hub_download(
                repo_id="lxq007/DiffBIR-v2",
                filename="realesrgan_s4_swinir_100k.pth",
                local_dir="weights",
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and len(state_dict) > 0:
            first_key = list(state_dict.keys())[0]
            if first_key.startswith("module."):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
        self.swinir_model.load_state_dict(state_dict, strict=True)
        self.swinir_model.eval().to(self.device)
        print("[TwoStagePipeline] SwinIR loaded")
    
    def load_sdxl(self):
        """Load SDXL Inpainting + LoRA"""
        from diffusers import StableDiffusionXLInpaintPipeline
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.sdxl_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.sdxl_model_path,
            torch_dtype=torch_dtype,
            variant="fp16" if self.device == "cuda" else None,
        )
        
        if self.lora_path and os.path.exists(self.lora_path):
            from peft import PeftModel
            self.sdxl_pipe.unet = PeftModel.from_pretrained(
                self.sdxl_pipe.unet,
                self.lora_path
            )
            print(f"[TwoStagePipeline] LoRA loaded from {self.lora_path}")
        
        if self.device == "cuda":
            self.sdxl_pipe.enable_model_cpu_offload()
        else:
            self.sdxl_pipe.to(self.device)
        print("[TwoStagePipeline] SDXL Inpainting loaded")
    
    def load(self):
        """Load all models"""
        self.load_swinir()
        self.load_sdxl()
        print("[TwoStagePipeline] All models loaded!")
    
    @torch.no_grad()
    def run_swinir(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """Run SwinIR preprocessing"""
        if self.swinir_model is None:
            raise RuntimeError("SwinIR not loaded")
        
        img = image.resize((target_size, target_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast('cuda'):
            output = self.swinir_model(img_tensor)
        
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(output)
    
    def detect_scratches(self, image: Image.Image) -> Image.Image:
        """Detect scratches and generate mask"""
        arr = np.array(image)
        mask = self.scratch_detector.detect(arr)
        return Image.fromarray(mask)
    
    def stage1_repair(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "restored old photo, repaired, clean, no scratches",
        strength: float = 0.6,
        seed: int = 42,
    ) -> Image.Image:
        """
        Stage 1: Repair scratches and damaged regions
        
        Uses higher strength to effectively repair damaged areas
        Only processes masked regions
        """
        print(f"[Stage 1] Repairing scratches (strength={strength})...")
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        result = self.sdxl_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        
        return result
    
    def stage2_enhance(
        self,
        image: Image.Image,
        prompt: str = "high quality restored photo, clear, sharp, natural colors",
        strength: float = 0.1,
        seed: int = 42,
    ) -> Image.Image:
        """
        Stage 2: Fine enhancement for overall quality
        
        Uses lower strength for subtle, natural enhancement
        Processes entire image with full mask
        """
        print(f"[Stage 2] Enhancing quality (strength={strength})...")
        
        full_mask = Image.new("L", image.size, 255)
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        result = self.sdxl_pipe(
            prompt=prompt,
            image=image,
            mask_image=full_mask,
            strength=strength,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        
        return result
    
    def run(
        self,
        image: Image.Image,
        stage1_strength: float = 0.6,
        stage2_strength: float = 0.1,
        use_swinir: bool = True,
        prompt_stage1: str = "restored old photo, repaired, clean, no scratches",
        prompt_stage2: str = "high quality restored photo, clear, sharp, natural colors",
        seed: int = 42,
        save_intermediates: bool = False,
        output_dir: str = "output",
    ) -> Dict[str, Image.Image]:
        """
        Run full two-stage restoration pipeline
        
        Args:
            image: Input PIL Image
            stage1_strength: Strength for scratch repair (0.5-0.7 recommended)
            stage2_strength: Strength for fine enhancement (0.05-0.15 recommended)
            use_swinir: Whether to use SwinIR preprocessing
            prompt_stage1: Prompt for Stage 1
            prompt_stage2: Prompt for Stage 2
            seed: Random seed
            save_intermediates: Save intermediate results
            output_dir: Directory for intermediate results
            
        Returns:
            Dictionary with all intermediate and final images
        """
        print(f"\n{'='*60}")
        print(f"[TwoStagePipeline] Processing image: {image.size}")
        print(f"{'='*60}")
        
        results = {}
        original_size = image.size
        
        if use_swinir and self.swinir_model is not None:
            print("\n[Preprocessing] Running SwinIR...")
            swinir_output = self.run_swinir(image)
            results["swinir"] = swinir_output
            working_image = swinir_output
        else:
            working_image = image.resize((512, 512), Image.LANCZOS)
            results["swinir"] = working_image
        
        print("\n[Detection] Detecting scratches and damage...")
        scratch_mask = self.detect_scratches(working_image)
        results["scratch_mask"] = scratch_mask
        
        mask_coverage = np.mean(np.array(scratch_mask)) / 255 * 100
        print(f"[Detection] Mask coverage: {mask_coverage:.1f}%")
        
        if mask_coverage > 1:
            print("\n[Stage 1] Repairing scratches...")
            repaired = self.stage1_repair(
                working_image,
                scratch_mask,
                prompt=prompt_stage1,
                strength=stage1_strength,
                seed=seed,
            )
            results["stage1_repaired"] = repaired
        else:
            print("\n[Stage 1] No significant damage detected, skipping repair...")
            repaired = working_image
            results["stage1_repaired"] = repaired
        
        print("\n[Stage 2] Fine enhancement...")
        enhanced = self.stage2_enhance(
            repaired,
            prompt=prompt_stage2,
            strength=stage2_strength,
            seed=seed,
        )
        results["stage2_enhanced"] = enhanced
        
        final = enhanced.resize(original_size, Image.LANCZOS)
        results["final"] = final
        
        if save_intermediates:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for name, img in results.items():
                path = os.path.join(output_dir, f"{timestamp}_{name}.jpg")
                img.save(path, "JPEG", quality=95)
            print(f"\n[Output] Results saved to {output_dir}")
        
        print(f"\n{'='*60}")
        print("[TwoStagePipeline] Done!")
        print(f"{'='*60}")
        
        return results
    
    def cleanup(self):
        """Free GPU memory"""
        if self.swinir_model is not None:
            del self.swinir_model
            self.swinir_model = None
        if self.sdxl_pipe is not None:
            del self.sdxl_pipe
            self.sdxl_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[TwoStagePipeline] Memory cleaned up")


def evaluate_two_stage(
    pipeline: TwoStageRestorationPipeline,
    test_pairs: List[Tuple[str, str]],
    stage1_strengths: List[float] = [0.5, 0.6, 0.7],
    stage2_strengths: List[float] = [0.05, 0.1, 0.15],
    seed: int = 42,
) -> Dict:
    """
    Evaluate two-stage pipeline with different strength combinations
    
    Args:
        pipeline: TwoStageRestorationPipeline instance
        test_pairs: List of (degraded_path, clean_path) tuples
        stage1_strengths: List of strengths to try for Stage 1
        stage2_strengths: List of strengths to try for Stage 2
        seed: Random seed
        
    Returns:
        Dictionary with evaluation results
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    results = {}
    
    for s1 in stage1_strengths:
        for s2 in stage2_strengths:
            key = f"s1={s1}_s2={s2}"
            metrics = {"psnr": [], "ssim": []}
            
            print(f"\n[Evaluating] Stage1={s1}, Stage2={s2}")
            
            for degraded_path, clean_path in test_pairs:
                try:
                    degraded = Image.open(degraded_path).convert("RGB")
                    clean = Image.open(clean_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                    
                    result = pipeline.run(
                        degraded,
                        stage1_strength=s1,
                        stage2_strength=s2,
                        use_swinir=True,
                        seed=seed,
                    )
                    
                    final = result["final"].resize((512, 512), Image.LANCZOS)
                    
                    clean_arr = np.array(clean)
                    final_arr = np.array(final)
                    
                    metrics["psnr"].append(psnr(clean_arr, final_arr))
                    metrics["ssim"].append(ssim(clean_arr, final_arr, channel_axis=2))
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
            
            if metrics["psnr"]:
                results[key] = {
                    "stage1_strength": s1,
                    "stage2_strength": s2,
                    "psnr": float(np.mean(metrics["psnr"])),
                    "ssim": float(np.mean(metrics["ssim"])),
                }
                print(f"  PSNR={results[key]['psnr']:.2f}, SSIM={results[key]['ssim']:.4f}")
    
    if results:
        best_key = max(results, key=lambda k: results[k]["psnr"])
        print(f"\n[Best] {best_key}: PSNR={results[best_key]['psnr']:.2f}, SSIM={results[best_key]['ssim']:.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Stage Photo Restoration Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument("--sdxl_model", type=str, 
                        default=str(BASE_MODELS_DIR / "AI-ModelScope" / "stable-diffusion-xl-1___0-inpainting-0___1"),
                        help="Path to SDXL Inpainting model")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--swinir", type=str, default=None, help="Path to SwinIR weights")
    parser.add_argument("--stage1_strength", type=float, default=0.6, 
                        help="Stage 1 strength (scratch repair, 0.5-0.7)")
    parser.add_argument("--stage2_strength", type=float, default=0.1,
                        help="Stage 2 strength (fine enhancement, 0.05-0.15)")
    parser.add_argument("--skip_swinir", action="store_true", help="Skip SwinIR preprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_intermediates", action="store_true", help="Save intermediate results")
    
    args = parser.parse_args()
    
    pipeline = TwoStageRestorationPipeline(
        sdxl_model_path=args.sdxl_model,
        lora_path=args.lora,
        swinir_model_path=args.swinir,
    )
    pipeline.load()
    
    if os.path.isfile(args.input):
        image = Image.open(args.input).convert("RGB")
        
        results = pipeline.run(
            image,
            stage1_strength=args.stage1_strength,
            stage2_strength=args.stage2_strength,
            use_swinir=not args.skip_swinir,
            seed=args.seed,
            save_intermediates=args.save_intermediates,
            output_dir=os.path.dirname(args.output) or "output",
        )
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results["final"].save(args.output, "JPEG", quality=95)
        
        if args.save_intermediates:
            base = os.path.splitext(args.output)[0]
            results["swinir"].save(f"{base}_swinir.jpg", "JPEG", quality=95)
            results["scratch_mask"].save(f"{base}_mask.jpg")
            results["stage1_repaired"].save(f"{base}_stage1.jpg", "JPEG", quality=95)
        
        print(f"\nResults saved to {args.output}")
    
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        
        files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            files.extend(Path(args.input).glob(f"*{ext}"))
            files.extend(Path(args.input).glob(f"*{ext.upper()}"))
        files = sorted(set(files))
        
        print(f"Found {len(files)} images to process")
        
        for idx, file_path in enumerate(files):
            print(f"\n[{idx+1}/{len(files)}] Processing: {file_path.name}")
            
            try:
                image = Image.open(file_path).convert("RGB")
                
                results = pipeline.run(
                    image,
                    stage1_strength=args.stage1_strength,
                    stage2_strength=args.stage2_strength,
                    use_swinir=not args.skip_swinir,
                    seed=args.seed,
                )
                
                output_path = os.path.join(args.output, f"{file_path.stem}_restored.jpg")
                results["final"].save(output_path, "JPEG", quality=95)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print(f"\nBatch processing done! Results saved to {args.output}")
    
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    pipeline.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()

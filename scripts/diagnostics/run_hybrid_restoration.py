# -*- coding: utf-8 -*-
"""
Quick Start Example for Hybrid Restoration Pipeline

Usage:
  python run_hybrid_restoration.py --input path/to/image.jpg --output path/to/output.jpg
"""

import os
import sys

from photo_revival.paths import BASE_MODELS_DIR, CHECKPOINTS_DIR

SDXL_MODEL_PATH = str(BASE_MODELS_DIR / "sdxl-inpainting")
LORA_PATH = str(CHECKPOINTS_DIR / "lora_inpainting_v25" / "final")

DEFAULT_PROMPT = "high quality restored photo, clear, sharp, natural colors, no damage"
DEFAULT_NEG_PROMPT = "blurry, low quality, artifacts, noise, damage, scratches"


def restore_single_image(
    input_path: str,
    output_path: str,
    strength: float = 0.35,
    mask_type: str = "auto",
    use_stage1: bool = True,
):
    """Restore a single image using hybrid pipeline"""
    from PIL import Image
    from photo_revival.hybrid_restoration_pipeline import HybridRestorationPipeline
    
    print(f"Loading models...")
    pipeline = HybridRestorationPipeline(
        sdxl_model_path=SDXL_MODEL_PATH,
        lora_path=LORA_PATH,
    )
    
    if use_stage1:
        pipeline.load("swinir")
    else:
        pipeline.load_stage2()
    
    print(f"Processing: {input_path}")
    image = Image.open(input_path).convert("RGB")
    
    final, stage1, mask = pipeline.run(
        image,
        stage1_enabled=use_stage1,
        mask_type=mask_type,
        strength=strength,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEG_PROMPT,
    )
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.save(output_path, "JPEG", quality=95)
    
    base, _ = os.path.splitext(output_path)
    stage1.save(f"{base}_stage1.jpg", "JPEG", quality=95)
    mask.save(f"{base}_mask.jpg")
    
    print(f"\nSaved:")
    print(f"  Final: {output_path}")
    print(f"  Stage 1: {base}_stage1.jpg")
    print(f"  Mask: {base}_mask.jpg")
    
    pipeline.cleanup()
    return final


def restore_directory(
    input_dir: str,
    output_dir: str,
    strength: float = 0.35,
    mask_type: str = "auto",
    use_stage1: bool = True,
):
    """Restore all images in a directory"""
    from photo_revival.hybrid_restoration_pipeline import HybridRestorationPipeline, batch_process
    
    print(f"Loading models...")
    pipeline = HybridRestorationPipeline(
        sdxl_model_path=SDXL_MODEL_PATH,
        lora_path=LORA_PATH,
    )
    
    if use_stage1:
        pipeline.load("swinir")
    else:
        pipeline.load_stage2()
    
    results = batch_process(
        input_dir,
        output_dir,
        pipeline,
        strength=strength,
        mask_type=mask_type,
    )
    
    pipeline.cleanup()
    return results


def compare_pipelines(input_path: str, output_dir: str):
    """Compare different pipeline configurations"""
    from PIL import Image
    from photo_revival.hybrid_restoration_pipeline import HybridRestorationPipeline
    
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(input_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    configs = [
        {"name": "sdxl_only", "use_stage1": False, "strength": 0.5, "mask_type": "full"},
        {"name": "stage1_only", "use_stage1": True, "strength": 0.0, "mask_type": "full"},
        {"name": "hybrid_auto", "use_stage1": True, "strength": 0.35, "mask_type": "auto"},
        {"name": "hybrid_full", "use_stage1": True, "strength": 0.35, "mask_type": "full"},
        {"name": "hybrid_light", "use_stage1": True, "strength": 0.2, "mask_type": "auto"},
    ]
    
    pipeline = HybridRestorationPipeline(
        sdxl_model_path=SDXL_MODEL_PATH,
        lora_path=LORA_PATH,
    )
    pipeline.load("swinir")
    
    print(f"\nComparing {len(configs)} configurations...")
    
    for config in configs:
        print(f"\n  Running: {config['name']}")
        
        if config["strength"] == 0.0:
            final = pipeline.run_stage1(image)
        else:
            final, _, _ = pipeline.run(
                image,
                stage1_enabled=config["use_stage1"],
                strength=config["strength"],
                mask_type=config["mask_type"],
            )
        
        output_path = os.path.join(output_dir, f"{base_name}_{config['name']}.jpg")
        final.save(output_path, "JPEG", quality=95)
        print(f"    Saved: {output_path}")
    
    pipeline.cleanup()
    print(f"\nComparison complete! Results in: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Start for Hybrid Restoration")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument("--strength", type=float, default=0.35, help="Inpainting strength")
    parser.add_argument("--mask_type", type=str, default="auto", choices=["auto", "full", "edges"])
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1")
    parser.add_argument("--compare", action="store_true", help="Compare different configurations")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_pipelines(args.input, args.output)
    elif os.path.isfile(args.input):
        restore_single_image(
            args.input,
            args.output,
            strength=args.strength,
            mask_type=args.mask_type,
            use_stage1=not args.skip_stage1,
        )
    elif os.path.isdir(args.input):
        restore_directory(
            args.input,
            args.output,
            strength=args.strength,
            mask_type=args.mask_type,
            use_stage1=not args.skip_stage1,
        )
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

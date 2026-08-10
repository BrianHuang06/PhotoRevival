# Legacy training experiments

These scripts record earlier experimental training routes and are not the recommended entry points.

- `train_inpainting_lora_v2.py`: original full-mask inpainting experiment (v25 output).
- `train_lora_enhanced.py`: configurable v26 experiment.
- `train_lora_optimized.py`: SwinIR + SDXL LoRA experiment (v27 output).
- `train_lora_v28.py`: v27 variant with BOPBTL masks.

Use `python -m scripts.training.train_lora` for new LoRA training. SwinIR and ControlNet remain separate because they train different model families.

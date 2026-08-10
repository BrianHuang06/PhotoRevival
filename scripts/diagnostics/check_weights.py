import os, sys, numpy as np, torch
from safetensors.torch import load_file

from photo_revival.paths import CHECKPOINTS_DIR

lora_dir = str(CHECKPOINTS_DIR / "lora_inpainting_v26")
checkpoints = ["epoch5", "epoch10", "epoch15", "epoch20", "epoch25", "epoch35", "epoch40", "epoch45", "epoch50", "final"]

print("=" * 70)
print("1. CHECK: Do LoRA weight files exist and have different sizes?")
print("=" * 70)

for ckpt in checkpoints:
    path = os.path.join(lora_dir, ckpt, "adapter_model.safetensors")
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {ckpt}: {size:,} bytes")
    else:
        print(f"  {ckpt}: NOT FOUND")

print()
print("=" * 70)
print("2. CHECK: Are the LoRA weights actually different between checkpoints?")
print("=" * 70)

weights = {}
for ckpt in checkpoints:
    path = os.path.join(lora_dir, ckpt, "adapter_model.safetensors")
    if os.path.exists(path):
        st = load_file(path)
        first_key = list(st.keys())[0]
        first_val = st[first_key].float().numpy()
        weights[ckpt] = first_val
        print(f"  {ckpt}: {len(st)} keys, first_key={first_key}, shape={first_val.shape}, "
              f"mean={first_val.mean():.6f}, std={first_val.std():.6f}")

print()
print("3. CHECK: Pairwise difference between checkpoints")
print("-" * 50)

ckpt_list = [c for c in checkpoints if c in weights]
for i in range(len(ckpt_list) - 1):
    a, b = ckpt_list[i], ckpt_list[i + 1]
    diff = np.abs(weights[a] - weights[b]).mean()
    max_diff = np.abs(weights[a] - weights[b]).max()
    print(f"  {a} vs {b}: mean_diff={diff:.8f}, max_diff={max_diff:.8f}")

print()
print("=" * 70)
print("4. CHECK: Are LoRA weights near zero (not trained)?")
print("=" * 70)

for ckpt in ckpt_list:
    path = os.path.join(lora_dir, ckpt, "adapter_model.safetensors")
    st = load_file(path)
    all_vals = torch.cat([v.float().flatten() for v in st.values()])
    print(f"  {ckpt}: global_mean={all_vals.mean():.6f}, global_std={all_vals.std():.6f}, "
          f"global_max={all_vals.max():.6f}, global_min={all_vals.min():.6f}")

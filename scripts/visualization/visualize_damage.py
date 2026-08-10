# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageDraw

from photo_revival.paths import DATASETS_DIR, EVALUATIONS_DIR

SYNTHETIC_DIR = DATASETS_DIR / "03_Synthetic_Dataset" / "03_Synthetic_Dataset"
DAMAGED_DIR = str(SYNTHETIC_DIR / "Train_GT_Damaged")
CLEAN_DIR = str(SYNTHETIC_DIR / "Train_GT_Clean")
OUTPUT_DIR = str(EVALUATIONS_DIR / "damage_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = ['lrp_img10.jpg', 'lrp_img100.jpg', 'lrp_img119.jpg']

for f in files:
    dmg_path = os.path.join(DAMAGED_DIR, f)
    clean_path = os.path.join(CLEAN_DIR, f)
    
    if not os.path.exists(dmg_path) or not os.path.exists(clean_path):
        continue
    
    dmg = Image.open(dmg_path).convert('RGB')
    clean = Image.open(clean_path).convert('RGB')
    
    w, h = clean.size
    dmg = dmg.resize((w, h), Image.LANCZOS)
    
    dmg_arr = np.array(dmg)
    clean_arr = np.array(clean)
    diff = np.abs(dmg_arr.astype(float) - clean_arr.astype(float))
    diff_mask = (diff.mean(axis=2) > 20).astype(np.uint8) * 255
    
    mask_img = Image.fromarray(diff_mask)
    
    comparison = Image.new('RGB', (w * 3 + 40, h + 60), (255, 255, 255))
    
    dmg_resized = dmg.resize((min(w, 400), min(h, 400)), Image.LANCZOS)
    clean_resized = clean.resize((min(w, 400), min(h, 400)), Image.LANCZOS)
    mask_resized = mask_img.resize((min(w, 400), min(h, 400)), Image.LANCZOS)
    
    new_w, new_h = dmg_resized.size
    comparison = Image.new('RGB', (new_w * 3 + 40, new_h + 60), (255, 255, 255))
    
    comparison.paste(dmg_resized, (10, 50))
    comparison.paste(clean_resized, (new_w + 20, 50))
    comparison.paste(mask_resized, (new_w * 2 + 30, 50))
    
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), 'Damaged', fill=(0, 0, 0))
    draw.text((new_w + 20, 10), 'Clean', fill=(0, 0, 0))
    draw.text((new_w * 2 + 30, 10), 'Damage Mask', fill=(0, 0, 0))
    
    output_path = os.path.join(OUTPUT_DIR, f'compare_{f}')
    comparison.save(output_path)
    print(f'Saved: {output_path}')

print('\nDone! Check the damage_comparison folder.')

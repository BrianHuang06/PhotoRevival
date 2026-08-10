# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photo_revival.paths import DATASETS_DIR, EVALUATIONS_DIR

CLEAN_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
DEGRADED_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
OUTPUT_DIR = str(EVALUATIONS_DIR / "degraded_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

samples = ['lrp_img10', 'lrp_img100', 'lrp_img119']

degradation_types = [
    'Noise_JPEG_Blur',
    'Faded_Scratches_Blur', 
    'Complex_All'
]

for sample in samples:
    clean_path = os.path.join(CLEAN_DIR, f'{sample}.jpg')
    if not os.path.exists(clean_path):
        continue
    
    clean = Image.open(clean_path).convert('RGB')
    
    images = [clean]
    labels = ['Clean (GT)']
    
    for deg_type in degradation_types:
        deg_path = os.path.join(DEGRADED_DIR, f'{sample}_({deg_type}).jpg')
        if os.path.exists(deg_path):
            deg_img = Image.open(deg_path).convert('RGB')
            deg_img = deg_img.resize(clean.size, Image.LANCZOS)
            images.append(deg_img)
            labels.append(deg_type.replace('_', '\n'))
    
    if len(images) < 2:
        continue
    
    img_w, img_h = images[0].size
    scale = min(300 / img_w, 300 / img_h, 1.0)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    images = [img.resize((new_w, new_h), Image.LANCZOS) for img in images]
    
    total_w = len(images) * new_w + (len(images) + 1) * 10
    total_h = new_h + 80
    
    comparison = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        x = 10 + i * (new_w + 10)
        comparison.paste(img, (x, 60))
        
        draw = ImageDraw.Draw(comparison)
        bbox = draw.textbbox((0, 0), label)
        text_w = bbox[2] - bbox[0]
        text_x = x + (new_w - text_w) // 2
        draw.text((text_x, 10), label, fill=(0, 0, 0))
    
    output_path = os.path.join(OUTPUT_DIR, f'{sample}_comparison.jpg')
    comparison.save(output_path, quality=95)
    print(f'Saved: {output_path}')

print('\nDone! Check the degraded_comparison folder.')
print('\nDegradation types:')
print('  - Noise_JPEG_Blur: Noise + JPEG artifacts + Blur')
print('  - Faded_Scratches_Blur: Faded + Scratches + Blur')
print('  - Complex_All: All degradation types combined')

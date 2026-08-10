# -*- coding: utf-8 -*-
"""
Create comparison images for BOPBTL masks
"""

import os
import cv2
import numpy as np

from photo_revival.paths import EVALUATIONS_DIR

def make_comparison(original, mask):
    h, w = original.shape[:2]
    
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    overlay = original.copy()
    overlay[mask > 128] = [0, 0, 255]
    blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:w*2] = mask_color
    comparison[:, w*2:] = blended
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask (BOPBTL)", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Overlay", (w * 2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    return comparison

def main():
    input_dir = str(EVALUATIONS_DIR / "bopbtl_masks" / "input")
    mask_dir = str(EVALUATIONS_DIR / "bopbtl_masks" / "mask")
    output_dir = str(EVALUATIONS_DIR / "bopbtl_comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:5]
    
    print(f"Creating comparison for {len(files)} images...")
    
    for fname in files:
        orig = cv2.imread(os.path.join(input_dir, fname))
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        
        if orig is None or mask is None:
            continue
        
        if orig.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        
        comparison = make_comparison(orig, mask)
        
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_comparison.jpg"), comparison)
        
        coverage = np.mean(mask) / 255 * 100
        print(f"{fname}: mask coverage = {coverage:.1f}%")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()

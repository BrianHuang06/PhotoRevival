# -*- coding: utf-8 -*-
"""
Improved Mask Detection - Focus on actual damage regions
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from photo_revival.paths import DATASETS_DIR, EVALUATIONS_DIR

class PreciseMaskDetector:
    """
    Precise mask detector for scratches and damage
    """
    
    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect damaged regions precisely
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Binary mask (H, W), 255 = damaged, 0 = clean
        """
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
        """Detect thin linear scratches"""
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


def create_comparison(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison"""
    h, w = original.shape[:2]
    
    if len(original.shape) == 2:
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = original.copy()
        if orig_rgb.shape[2] == 4:
            orig_rgb = orig_rgb[:, :, :3]
    
    mask_vis = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
    
    overlay = orig_rgb.copy()
    red_overlay = np.zeros_like(orig_rgb)
    red_overlay[mask > 128] = [255, 0, 0]
    blended = cv2.addWeighted(orig_rgb, 0.7, red_overlay, 0.3, 0)
    
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = orig_rgb
    comparison[:, w:w*2] = mask_rgb
    comparison[:, w*2:] = blended
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "Overlay", (w * 2 + 10, 30), font, 0.8, (255, 255, 255), 2)
    
    return comparison


def main():
    test_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    output_dir = str(EVALUATIONS_DIR / "precise_masks")
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = PreciseMaskDetector(sensitivity=0.5)
    
    test_files = sorted([f for f in os.listdir(test_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:5]
    
    print(f"Processing {len(test_files)} images...")
    
    for fname in test_files:
        print(f"\n{fname}")
        
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = detector.detect(img_rgb)
        
        comparison = create_comparison(img_rgb, mask)
        
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_comparison.jpg"),
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base}_mask.jpg"), mask)
        
        coverage = np.mean(mask) / 255 * 100
        print(f"  Mask coverage: {coverage:.1f}%")
    
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()

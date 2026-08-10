# -*- coding: utf-8 -*-
"""
Generate mask comparison using scratch textures
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random

from photo_revival.paths import DATASETS_DIR, EVALUATIONS_DIR, TEXTURES_DIR

class TextureBasedMaskDetector:
    """
    Mask detector using scratch texture analysis
    """
    
    def __init__(self, texture_dir: str = None, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self.texture_features = None
        
        if texture_dir:
            self._learn_from_textures(texture_dir)
    
    def _learn_from_textures(self, texture_dir: str):
        """Learn scratch characteristics from texture samples"""
        texture_path = Path(texture_dir) / "Resource Boy - Grunge Textures"
        files = list(texture_path.glob("*.jpg"))[:30]
        
        if not files:
            print(f"No textures found in {texture_path}")
            return
        
        print(f"Learning from {len(files)} texture samples...")
        
        self.texture_features = {
            "blackhat_thresholds": [],
            "edge_densities": [],
            "contrast_values": [],
        }
        
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, (512, 512))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            self.texture_features["blackhat_thresholds"].append(np.percentile(blackhat, 90))
            
            edges = cv2.Canny(img, 50, 150)
            self.texture_features["edge_densities"].append(np.mean(edges) / 255)
            
            self.texture_features["contrast_values"].append(np.std(img))
        
        for k, v in self.texture_features.items():
            if v:
                self.texture_features[k] = np.mean(v)
        
        print(f"Learned features: {self.texture_features}")
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect damaged regions using learned texture features
        
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
        
        mask = np.maximum(mask, self._detect_scratch_lines(gray))
        mask = np.maximum(mask, self._detect_dust_and_spots(gray))
        mask = np.maximum(mask, self._detect_creases(gray))
        mask = np.maximum(mask, self._detect_white_damages(gray))
        mask = np.maximum(mask, self._detect_dark_damages(gray))
        
        mask = self._refine_mask(mask)
        
        return mask
    
    def _detect_scratch_lines(self, gray: np.ndarray) -> np.ndarray:
        """Detect thin scratch lines"""
        combined = np.zeros_like(gray)
        
        for kernel_size in [9, 15, 21]:
            for angle in [0, 90]:
                if angle == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
                
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
                
                thresh = 10 + int(20 * self.sensitivity)
                _, binary = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
                
                combined = np.maximum(combined, binary)
        
        for kernel_size in [7, 11]:
            diag_kernel = np.eye(kernel_size, dtype=np.uint8)
            anti_diag_kernel = np.fliplr(diag_kernel)
            
            for k in [diag_kernel, anti_diag_kernel]:
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
                _, binary = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
                combined = np.maximum(combined, binary)
        
        return combined
    
    def _detect_dust_and_spots(self, gray: np.ndarray) -> np.ndarray:
        """Detect dust spots and small artifacts"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        
        _, binary = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 2000:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        
        return result
    
    def _detect_creases(self, gray: np.ndarray) -> np.ndarray:
        """Detect creases and folds using edge detection"""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )
        
        result = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 20:
                    cv2.line(result, (x1, y1), (x2, y2), 255, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def _detect_white_damages(self, gray: np.ndarray) -> np.ndarray:
        """Detect white scratches and spots"""
        white_mask = (gray > 245).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask
    
    def _detect_dark_damages(self, gray: np.ndarray) -> np.ndarray:
        """Detect dark scratches and spots"""
        dark_mask = (gray < 10).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        
        return dark_mask
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask with morphological operations"""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask


def create_comparison_image(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison image"""
    h, w = original.shape[:2]
    
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original.copy()
    
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    overlay = original_rgb.copy()
    overlay[mask > 128] = [255, 0, 0]
    blended = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)
    
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = original_rgb
    comparison[:, w:w*2] = mask_rgb
    comparison[:, w*2:] = blended
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Overlay", (w * 2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    return comparison


def main():
    texture_dir = str(TEXTURES_DIR / "Resource-Boy-Grunge-Textures")
    test_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    output_dir = str(EVALUATIONS_DIR / "mask_comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Texture-Based Mask Detection")
    print("="*60)
    
    detector = TextureBasedMaskDetector(texture_dir, sensitivity=0.5)
    
    test_files = sorted([f for f in os.listdir(test_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:5]
    
    print(f"\nProcessing {len(test_files)} test images...")
    
    for fname in test_files:
        print(f"\nProcessing: {fname}")
        
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  Cannot read: {img_path}")
            continue
        
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = detector.detect(img_rgb)
        
        comparison = create_comparison_image(img_rgb, mask)
        
        base_name = os.path.splitext(fname)[0]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.jpg"), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.jpg"), mask)
        
        mask_coverage = np.mean(mask) / 255 * 100
        print(f"  Mask coverage: {mask_coverage:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Analyze scratch textures and improve detection
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random

from photo_revival.paths import ARTIFACTS_DIR, TEXTURES_DIR

def analyze_textures(texture_dir, num_samples=20):
    """Analyze scratch texture characteristics"""
    texture_path = Path(texture_dir) / "Resource Boy - Grunge Textures"
    files = list(texture_path.glob("*.jpg"))
    
    if not files:
        print(f"No texture files found in {texture_path}")
        return None
    
    samples = random.sample(files, min(num_samples, len(files)))
    
    stats = {
        "avg_brightness": [],
        "contrast": [],
        "edge_density": [],
        "line_density_h": [],
        "line_density_v": [],
        "noise_level": [],
    }
    
    print(f"Analyzing {len(samples)} texture samples...")
    
    for f in samples:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        img = cv2.resize(img, (512, 512))
        
        stats["avg_brightness"].append(np.mean(img))
        stats["contrast"].append(np.std(img))
        
        edges = cv2.Canny(img, 50, 150)
        stats["edge_density"].append(np.mean(edges) / 255)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        blackhat_h = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_h)
        blackhat_v = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_v)
        stats["line_density_h"].append(np.mean(blackhat_h))
        stats["line_density_v"].append(np.mean(blackhat_v))
        
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        stats["noise_level"].append(np.var(laplacian))
    
    results = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in stats.items()}
    
    print("\nTexture Analysis Results:")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k}: mean={v['mean']:.2f}, std={v['std']:.2f}")
    
    return results


def create_improved_detector(
    texture_dir,
    output_path=str(ARTIFACTS_DIR / "generated" / "improved_scratch_detector.py"),
):
    """Create improved scratch detector based on texture analysis"""
    
    code = '''# -*- coding: utf-8 -*-
"""
Improved Scratch Detector based on Grunge Texture Analysis
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ImprovedScratchDetector:
    """
    Improved scratch detector using multi-scale analysis and texture features
    """
    
    def __init__(
        self,
        sensitivity: float = 0.5,
        min_line_length: int = 10,
        max_line_width: int = 5,
    ):
        self.sensitivity = sensitivity
        self.min_line_length = min_line_length
        self.max_line_width = max_line_width
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect scratches with improved accuracy
        
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
        mask = np.maximum(mask, self._detect_dust_spots(gray))
        mask = np.maximum(mask, self._detect_creases(gray))
        mask = np.maximum(mask, self._detect_tears(gray))
        
        mask = self._refine_mask(mask)
        
        return mask
    
    def _detect_scratch_lines(self, gray: np.ndarray) -> np.ndarray:
        """Detect thin scratch lines using multi-scale approach"""
        masks = []
        
        for scale in [0.5, 1.0, 1.5]:
            if scale != 1.0:
                h, w = gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(gray, (new_w, new_h))
            else:
                scaled = gray
            
            for angle in [0, 45, 90, 135]:
                kernel_size = max(9, int(15 * scale))
                
                if angle == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                elif angle == 90:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
                elif angle == 45:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    kernel = np.eye(kernel_size, dtype=np.uint8)
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    kernel = np.fliplr(np.eye(kernel_size, dtype=np.uint8))
                
                blackhat = cv2.morphologyEx(scaled, cv2.MORPH_BLACKHAT, kernel)
                
                thresh = int(15 + 20 * self.sensitivity)
                _, binary = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
                
                if scale != 1.0:
                    binary = cv2.resize(binary, (w, h))
                
                masks.append(binary)
        
        combined = np.zeros_like(gray)
        for m in masks:
            combined = np.maximum(combined, m)
        
        return combined
    
    def _detect_dust_spots(self, gray: np.ndarray) -> np.ndarray:
        """Detect dust spots and small artifacts"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        
        _, binary = cv2.threshold(diff, 20 + int(30 * self.sensitivity), 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 1000:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        
        return result
    
    def _detect_creases(self, gray: np.ndarray) -> np.ndarray:
        """Detect creases and folds"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        _, binary = cv2.threshold(magnitude, 50 + int(50 * self.sensitivity), 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 
                                threshold=50 + int(50 * self.sensitivity),
                                minLineLength=self.min_line_length,
                                maxLineGap=5)
        
        result = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), 255, 2)
        
        return result
    
    def _detect_tears(self, gray: np.ndarray) -> np.ndarray:
        """Detect tears and missing pieces"""
        edges = cv2.Canny(gray, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(gray)
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 50:
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) > 5:
                    cv2.drawContours(result, [cnt], -1, 255, -1)
        
        return result
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask with morphological operations"""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask


def test_detector(image_path: str, output_dir: str = "mask_test_output"):
    """Test the improved detector on an image"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detector = ImprovedScratchDetector(sensitivity=0.5)
    mask = detector.detect(img_rgb)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), img)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.jpg"), mask)
    
    overlay = img.copy()
    overlay[mask > 128] = [0, 255, 0]
    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.jpg"), result)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Test image path")
    parser.add_argument("--output", type=str, default="mask_test_output")
    args = parser.parse_args()
    
    if args.image:
        test_detector(args.image, args.output)
'''
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"Improved detector saved to: {output_path}")
    return output_path


def main():
    texture_dir = str(TEXTURES_DIR / "Resource-Boy-Grunge-Textures")
    
    print("="*60)
    print("Scratch Texture Analysis")
    print("="*60)
    
    results = analyze_textures(texture_dir)
    
    if results:
        print("\n" + "="*60)
        print("Creating improved detector...")
        print("="*60)
        create_improved_detector(texture_dir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Improved scratch detector - learn local scratch patterns from textures
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle

from photo_revival.paths import CLASSIFIERS_DIR, DATASETS_DIR, EVALUATIONS_DIR, TEXTURES_DIR


class FeatureExtractor:
    """Extract texture features from patches"""

    def __init__(self, patch_size=32):
        self.patch_size = patch_size
        self._gabor_kernels = self._build_gabor_kernels()

    def _build_gabor_kernels(self):
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            for sigma in [1.0, 2.0, 3.0]:
                for lambd in [3.0, 5.0, 8.0]:
                    kernel = cv2.getGaborKernel(
                        ksize=(self.patch_size, self.patch_size),
                        sigma=sigma, theta=theta, lambd=lambd, gamma=0.5, psi=0
                    )
                    kernels.append(kernel)
        return kernels

    def extract(self, patch_gray):
        features = []
        features.extend(self._gabor_features(patch_gray))
        features.extend(self._lbp_features(patch_gray))
        features.extend(self._statistical_features(patch_gray))
        features.extend(self._gradient_features(patch_gray))
        return np.array(features, dtype=np.float32)

    def _gabor_features(self, patch):
        responses = []
        for kernel in self._gabor_kernels:
            filtered = cv2.filter2D(patch, cv2.CV_32F, kernel)
            responses.append(np.mean(np.abs(filtered)))
            responses.append(np.std(filtered))
        return responses

    def _lbp_features(self, patch, n_points=24, radius=3):
        h, w = patch.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = int(round(radius * np.cos(angle)))
            y = int(round(radius * np.sin(angle)))
            shifted = np.zeros_like(patch, dtype=np.float32)
            src_y = slice(max(0, -y), min(h, h - y))
            src_x = slice(max(0, -x), min(w, w - x))
            dst_y = slice(max(0, y), min(h, h + y))
            dst_x = slice(max(0, x), min(w, w + x))
            shifted[dst_y, dst_x] = patch[src_y, src_x].astype(np.float32)
            lbp += ((shifted >= patch.astype(np.float32)) * (1 << i)).astype(np.uint8)
        n_bins = 2 ** min(n_points, 8)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist.tolist()

    def _statistical_features(self, patch):
        patch_f = patch.astype(np.float32)
        mean_val = np.mean(patch_f)
        std_val = np.std(patch_f)
        skewness = np.mean(((patch_f - mean_val) / (std_val + 1e-7)) ** 3)
        kurtosis = np.mean(((patch_f - mean_val) / (std_val + 1e-7)) ** 4)
        hist, _ = np.histogram(patch.ravel(), bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return [mean_val, std_val, skewness, kurtosis, entropy]

    def _gradient_features(self, patch):
        sobel_x = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist.astype(np.float32)
        dir_hist /= (dir_hist.sum() + 1e-7)
        return [mag_mean, mag_std, np.max(dir_hist)] + dir_hist.tolist()


def extract_scratch_patches_from_textures(texture_dir, patch_size=32, num_patches=3000):
    """
    Extract patches that contain visible scratch lines from texture images.
    Focus on high-contrast, line-like regions.
    """
    texture_path = Path(texture_dir) / "Resource Boy - Grunge Textures"
    files = list(texture_path.glob("*.jpg"))
    
    patches = []
    rng = np.random.RandomState(42)
    
    print(f"Extracting scratch patches from {len(files)} texture files...")
    
    for f in files:
        if len(patches) >= num_patches:
            break
        
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        img = cv2.resize(img, (512, 512))
        
        # Detect high-contrast line regions (potential scratches)
        for kernel_len in [15, 21]:
            for angle in [0, 90]:
                if angle == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
                
                blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
                
                # Find regions with strong line response
                thresh = np.percentile(blackhat, 90)
                mask = blackhat > thresh
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if len(patches) >= num_patches:
                        break
                    
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    
                    # Ensure patch is within image bounds
                    if bw < 5 or bh < 5:
                        continue
                    
                    # Random offset around the detected region
                    cx = x + bw // 2
                    cy = y + bh // 2
                    
                    half = patch_size // 2
                    px = max(half, min(512 - half, cx + rng.randint(-10, 10)))
                    py = max(half, min(512 - half, cy + rng.randint(-10, 10)))
                    
                    patch = img[py - half:py + half, px - half:px + half]
                    
                    if patch.shape == (patch_size, patch_size):
                        patches.append(patch)
    
    print(f"  Extracted {len(patches)} scratch patches")
    return patches[:num_patches]


def extract_normal_patches_from_clean(clean_dir, patch_size=32, num_patches=3000):
    """Extract random patches from clean images (no scratches)"""
    files = list(Path(clean_dir).glob("*.jpg")) + list(Path(clean_dir).glob("*.png"))
    
    patches = []
    rng = np.random.RandomState(42)
    
    print(f"Extracting normal patches from {len(files)} clean images...")
    
    for f in files:
        if len(patches) >= num_patches:
            break
        
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        img = cv2.resize(img, (512, 512))
        
        patches_per_img = max(1, num_patches // len(files))
        
        for _ in range(patches_per_img):
            if len(patches) >= num_patches:
                break
            
            h, w = img.shape
            y = rng.randint(0, h - patch_size)
            x = rng.randint(0, w - patch_size)
            
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    
    print(f"  Extracted {len(patches)} normal patches")
    return patches[:num_patches]


def train_classifier(texture_dir, clean_dir, output_path="scratch_classifier_v2.pkl"):
    """Train improved classifier"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    patch_size = 32
    
    # Extract patches
    scratch_patches = extract_scratch_patches_from_textures(texture_dir, patch_size, 3000)
    normal_patches = extract_normal_patches_from_clean(clean_dir, patch_size, 3000)
    
    # Extract features
    extractor = FeatureExtractor(patch_size)
    
    X = []
    y = []
    
    print("\nExtracting features...")
    for i, patch in enumerate(scratch_patches):
        feat = extractor.extract(patch)
        X.append(feat)
        y.append(1)
        if (i + 1) % 500 == 0:
            print(f"  Scratch: {i + 1}/{len(scratch_patches)}")
    
    for i, patch in enumerate(normal_patches):
        feat = extractor.extract(patch)
        X.append(feat)
        y.append(0)
        if (i + 1) % 500 == 0:
            print(f"  Normal: {i + 1}/{len(normal_patches)}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    scaler_mean = np.mean(X, axis=0)
    scaler_std = np.std(X, axis=0) + 1e-7
    X_norm = (X - scaler_mean) / scaler_std
    
    print(f"\nFeature dim: {X_norm.shape[1]}")
    print(f"Scratch: {np.sum(y == 1)}, Normal: {np.sum(y == 0)}")
    
    # Train
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    
    scores = cross_val_score(clf, X_norm, y, cv=5, scoring='accuracy')
    print(f"Cross-val accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    clf.fit(X_norm, y)
    
    # Save
    data = {
        "classifier": clf,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "patch_size": patch_size,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nModel saved to {output_path}")
    
    return clf, scaler_mean, scaler_std, extractor


class ImprovedMaskDetector:
    """Use improved classifier for mask detection"""
    
    def __init__(self, model_path="scratch_classifier_v2.pkl"):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]
        self.patch_size = data["patch_size"]
        self.extractor = FeatureExtractor(self.patch_size)
    
    def detect(self, image, stride=4, threshold=0.3):
        """
        Detect scratch regions
        
        Args:
            image: RGB numpy array
            stride: Sliding window stride (smaller = more precise)
            threshold: Probability threshold (lower = more sensitive)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        ps = self.patch_size
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Sliding window
        patches_data = []
        positions = []
        
        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                patch = gray[y:y + ps, x:x + ps]
                feat = self.extractor.extract(patch)
                patches_data.append(feat)
                positions.append((y, x))
        
        if not patches_data:
            return np.zeros((h, w), dtype=np.uint8)
        
        X = np.array(patches_data)
        X_norm = (X - self.scaler_mean) / self.scaler_std
        
        probs = self.classifier.predict_proba(X_norm)[:, 1]
        
        for (y, x), prob in zip(positions, probs):
            prob_map[y:y + ps, x:x + ps] += prob
            count_map[y:y + ps, x:x + ps] += 1
        
        count_map = np.maximum(count_map, 1)
        avg_prob = prob_map / count_map
        
        mask = (avg_prob > threshold).astype(np.uint8) * 255
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask


def make_comparison(original, mask):
    h, w = original.shape[:2]
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    
    overlay = original.copy()
    overlay[mask > 128] = [255, 50, 50]
    blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:w * 2] = mask_rgb
    comparison[:, w * 2:] = blended
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "Mask", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "Overlay", (w * 2 + 10, 30), font, 0.8, (255, 255, 255), 2)
    
    return comparison


def test_detector(output_dir=str(EVALUATIONS_DIR / "improved_masks")):
    test_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    os.makedirs(output_dir, exist_ok=True)
    
    detector = ImprovedMaskDetector(str(CLASSIFIERS_DIR / "scratch_classifier_v2.pkl"))
    
    files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')])[:5]
    
    print(f"\nTesting on {len(files)} images...")
    
    for fname in files:
        img = cv2.imread(os.path.join(test_dir, fname))
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = detector.detect(img_rgb, stride=4, threshold=0.3)
        
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_mask.jpg"), mask)
        
        comparison = make_comparison(img_rgb, mask)
        cv2.imwrite(os.path.join(output_dir, f"{base}_comparison.jpg"),
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        coverage = np.mean(mask) / 255 * 100
        print(f"{fname}: {coverage:.1f}%")
    
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()
    
    if args.mode in ("train", "both"):
        train_classifier(
            str(TEXTURES_DIR / "Resource-Boy-Grunge-Textures"),
            str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean"),
            str(CLASSIFIERS_DIR / "scratch_classifier_v2.pkl")
        )
    
    if args.mode in ("test", "both"):
        test_detector()

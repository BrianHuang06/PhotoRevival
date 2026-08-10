# -*- coding: utf-8 -*-
"""
Learn scratch features from texture dataset and build a classifier-based mask detector.

Pipeline:
  1. Extract patches from Grunge textures (positive = scratch)
  2. Extract patches from clean photos (negative = normal)
  3. Compute features: Gabor, LBP, HOG, local statistics
  4. Train a Random Forest classifier
  5. Use sliding window + classifier for mask detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from photo_revival.paths import CLASSIFIERS_DIR, DATASETS_DIR, EVALUATIONS_DIR, TEXTURES_DIR


class FeatureExtractor:
    """Extract multi-scale texture features from image patches"""

    def __init__(self, patch_size=32):
        self.patch_size = patch_size
        self._gabor_kernels = self._build_gabor_kernels()

    def _build_gabor_kernels(self):
        """Build a bank of Gabor filters at different orientations and frequencies"""
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            for sigma in [1.0, 2.0, 3.0]:
                for lambd in [3.0, 5.0, 8.0]:
                    kernel = cv2.getGaborKernel(
                        ksize=(self.patch_size, self.patch_size),
                        sigma=sigma,
                        theta=theta,
                        lambd=lambd,
                        gamma=0.5,
                        psi=0,
                    )
                    kernels.append(kernel)
        return kernels

    def extract(self, patch_gray):
        """
        Extract features from a single grayscale patch.

        Returns:
            1D feature vector
        """
        features = []

        features.extend(self._gabor_features(patch_gray))
        features.extend(self._lbp_features(patch_gray))
        features.extend(self._statistical_features(patch_gray))
        features.extend(self._gradient_features(patch_gray))
        features.extend(self._frequency_features(patch_gray))

        return np.array(features, dtype=np.float32)

    def _gabor_features(self, patch):
        """Gabor filter responses - captures oriented texture patterns (scratches)"""
        responses = []
        for kernel in self._gabor_kernels:
            filtered = cv2.filter2D(patch, cv2.CV_32F, kernel)
            responses.append(np.mean(np.abs(filtered)))
            responses.append(np.std(filtered))
        return responses

    def _lbp_features(self, patch, n_points=24, radius=3):
        """Local Binary Pattern histogram - captures local texture patterns"""
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
        """Local statistical features"""
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
        """Gradient-based features - scratches have strong directional gradients"""
        sobel_x = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)

        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)

        dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist.astype(np.float32)
        dir_hist /= (dir_hist.sum() + 1e-7)

        dir_dominance = np.max(dir_hist)

        return [mag_mean, mag_std, dir_dominance] + dir_hist.tolist()

    def _frequency_features(self, patch):
        """Frequency domain features using DCT-like approach"""
        patch_f = patch.astype(np.float32)
        h, w = patch_f.shape

        block_means = []
        block_stds = []
        bs = 8
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block = patch_f[y:y + bs, x:x + bs]
                block_means.append(np.mean(block))
                block_stds.append(np.std(block))

        return [np.std(block_means), np.mean(block_stds), np.max(block_stds)]


class ScratchClassifierTrainer:
    """Train a classifier to distinguish scratch vs normal patches"""

    def __init__(
        self,
        texture_dir,
        clean_dir,
        patch_size=32,
        samples_per_class=2000,
    ):
        self.texture_dir = texture_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.samples_per_class = samples_per_class
        self.feature_extractor = FeatureExtractor(patch_size)
        self.classifier = None
        self.scaler_mean = None
        self.scaler_std = None

    def _extract_patches_from_dir(self, img_dir, num_patches, seed=42):
        """Randomly extract patches from images in a directory"""
        rng = np.random.RandomState(seed)
        files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
        if not files:
            print(f"  No images found in {img_dir}")
            return []

        patches = []
        patches_per_file = max(1, num_patches // len(files))

        for f in files:
            if len(patches) >= num_patches:
                break
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (256, 256))

            for _ in range(patches_per_file):
                if len(patches) >= num_patches:
                    break
                h, w = img.shape
                if h < self.patch_size or w < self.patch_size:
                    continue
                y = rng.randint(0, h - self.patch_size)
                x = rng.randint(0, w - self.patch_size)
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)

        return patches

    def train(self):
        """Train the classifier"""
        print("=" * 60)
        print("Training Scratch Classifier")
        print("=" * 60)

        print("\n[1/4] Extracting positive patches (scratch textures)...")
        positive_patches = self._extract_patches_from_dir(
            os.path.join(self.texture_dir, "Resource Boy - Grunge Textures"),
            self.samples_per_class,
        )
        print(f"  Got {len(positive_patches)} positive patches")

        print("\n[2/4] Extracting negative patches (clean photos)...")
        negative_patches = self._extract_patches_from_dir(
            self.clean_dir,
            self.samples_per_class,
        )
        print(f"  Got {len(negative_patches)} negative patches")

        print("\n[3/4] Extracting features...")
        X = []
        y = []

        for i, patch in enumerate(positive_patches):
            feat = self.feature_extractor.extract(patch)
            X.append(feat)
            y.append(1)
            if (i + 1) % 500 == 0:
                print(f"  Positive: {i + 1}/{len(positive_patches)}")

        for i, patch in enumerate(negative_patches):
            feat = self.feature_extractor.extract(patch)
            X.append(feat)
            y.append(0)
            if (i + 1) % 500 == 0:
                print(f"  Negative: {i + 1}/{len(negative_patches)}")

        X = np.array(X)
        y = np.array(y)

        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-7
        X_norm = (X - self.scaler_mean) / self.scaler_std

        print(f"\n  Feature dim: {X_norm.shape[1]}")
        print(f"  Positive: {np.sum(y == 1)}, Negative: {np.sum(y == 0)}")

        print("\n[4/4] Training Random Forest classifier...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        )

        scores = cross_val_score(self.classifier, X_norm, y, cv=5, scoring='accuracy')
        print(f"  Cross-val accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        self.classifier.fit(X_norm, y)

        importances = self.classifier.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        print(f"\n  Top-10 feature importances:")
        for idx in top_indices:
            print(f"    Feature {idx}: {importances[idx]:.4f}")

        return self

    def save(self, path="scratch_classifier.pkl"):
        """Save trained model"""
        data = {
            "classifier": self.classifier,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "patch_size": self.patch_size,
            "feature_dim": len(self.scaler_mean),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nModel saved to {path}")

    def load(self, path="scratch_classifier.pkl"):
        """Load trained model"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]
        self.patch_size = data["patch_size"]
        self.feature_extractor = FeatureExtractor(self.patch_size)
        print(f"Model loaded from {path}")
        return self


class LearnedMaskDetector:
    """Use trained classifier for mask detection via sliding window"""

    def __init__(self, model_path="scratch_classifier.pkl"):
        self.trainer = ScratchClassifierTrainer.__new__(ScratchClassifierTrainer)
        self.trainer.load(model_path)
        self.patch_size = self.trainer.patch_size

    def detect(self, image, stride=8, threshold=0.5):
        """
        Detect scratch regions using sliding window classification

        Args:
            image: RGB numpy array (H, W, 3)
            stride: Sliding window stride
            threshold: Probability threshold for classification

        Returns:
            mask: (H, W) uint8, 255 = scratch
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        ps = self.patch_size

        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        patches_data = []
        positions = []

        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                patch = gray[y:y + ps, x:x + ps]
                feat = self.trainer.feature_extractor.extract(patch)
                patches_data.append(feat)
                positions.append((y, x))

        if not patches_data:
            return np.zeros((h, w), dtype=np.uint8)

        X = np.array(patches_data)
        X_norm = (X - self.trainer.scaler_mean) / self.trainer.scaler_std

        probs = self.trainer.classifier.predict_proba(X_norm)[:, 1]

        for (y, x), prob in zip(positions, probs):
            prob_map[y:y + ps, x:x + ps] += prob
            count_map[y:y + ps, x:x + ps] += 1

        count_map = np.maximum(count_map, 1)
        avg_prob = prob_map / count_map

        mask = (avg_prob > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask


def train_and_save():
    """Train and save the classifier"""
    trainer = ScratchClassifierTrainer(
        texture_dir=str(TEXTURES_DIR / "Resource-Boy-Grunge-Textures"),
        clean_dir=str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean"),
        patch_size=32,
        samples_per_class=2000,
    )
    trainer.train()
    trainer.save(str(CLASSIFIERS_DIR / "scratch_classifier.pkl"))


def test_detection():
    """Test the trained classifier on degraded images"""
    test_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
    output_dir = str(EVALUATIONS_DIR / "learned_masks")
    os.makedirs(output_dir, exist_ok=True)

    detector = LearnedMaskDetector(str(CLASSIFIERS_DIR / "scratch_classifier.pkl"))

    files = sorted([f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.png'))])[:5]

    print(f"\nTesting on {len(files)} images...")

    for fname in files:
        print(f"\n{fname}")
        img = cv2.imread(os.path.join(test_dir, fname))
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = detector.detect(img_rgb, stride=8, threshold=0.5)

        base = os.path.splitext(fname)[0]

        cv2.imwrite(os.path.join(output_dir, f"{base}_mask.jpg"), mask)

        comparison = _make_comparison(img_rgb, mask)
        cv2.imwrite(os.path.join(output_dir, f"{base}_comparison.jpg"),
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        coverage = np.mean(mask) / 255 * 100
        print(f"  Mask coverage: {coverage:.1f}%")

    print(f"\nResults: {output_dir}")


def _make_comparison(original, mask):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train_and_save()
    if args.mode in ("test", "both"):
        test_detection()

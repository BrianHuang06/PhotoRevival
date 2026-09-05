"""Recheck source damage labels against the existing texture catalog.

Physical damage labels are assigned by comparing local source-image patches
with enabled texture-material prototypes. Non-texture attributes such as blur,
noise, color cast, and exposure remain image-property checks because the
material catalog does not contain those phenomena.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts.synthetic.source_pipeline import (
    DAMAGE_TYPES,
    _damage_evidence,
    _feature_vector,
)
from scripts.synthetic.texture_assets import DEFAULT_TEXTURE_DIR


TEXTURE_DAMAGE_TYPES = (
    "scratch",
    "crack",
    "fold",
    "dust",
    "stain",
    "mold",
    "tear",
    "missing_region",
)

NON_TEXTURE_DAMAGE_TYPES = (
    "fading",
    "color_cast",
    "blur",
    "noise",
    "jpeg_artifact",
    "uneven_exposure",
)

TEXTURE_SUBLABEL_TO_DAMAGE = {
    "scratch": ("scratch",),
    "crack": ("crack",),
    "fold": ("fold",),
    "dust": ("dust",),
    "stain": ("stain",),
    "mold": ("mold",),
    "surface_wear": ("stain",),
    "edge_wear": ("tear",),
    "tear_missing": ("tear", "missing_region"),
}

NON_TEXTURE_THRESHOLDS = {
    "fading": 0.73,
    "color_cast": 0.72,
    "blur": 0.82,
    "noise": 0.76,
    "jpeg_artifact": 0.78,
    "uneven_exposure": 0.72,
}

TEXTURE_THRESHOLD_MARGIN = 0.05

LABEL_FIELDS = [
    "source_id",
    "filename",
    "quality_label",
    "color_mode",
    "damage_label_method",
    "damage_label_list",
    *[f"damage_{name}" for name in DAMAGE_TYPES],
    *[f"confidence_{name}" for name in DAMAGE_TYPES],
    *[f"evidence_{name}" for name in DAMAGE_TYPES],
    "texture_match_label_list",
    "texture_match_scores",
    "texture_match_files",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_gray(path: Path, size: int = 256) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _descriptor(gray: np.ndarray) -> np.ndarray:
    image = gray.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)
    high = np.abs(image - soft)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=False)
    orientation, _ = np.histogram(
        angle,
        bins=8,
        range=(0.0, 2.0 * math.pi),
        weights=magnitude,
    )
    orientation = orientation.astype(np.float32)
    orientation /= max(float(orientation.sum()), 1e-6)

    block_values: list[float] = []
    row_blocks = np.array_split(high, 4, axis=0)
    for row_block in row_blocks:
        for block in np.array_split(row_block, 4, axis=1):
            block_values.append(float(np.mean(block)))

    edges = cv2.Canny(gray, 40, 120)
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    interior = image[2:-2, 2:-2]
    stats = np.array(
        [
            float(np.mean(high)),
            float(np.std(high)),
            float(np.percentile(high, 75)),
            float(np.percentile(high, 95)),
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.mean(edges > 0)),
            float(np.mean(border > 0.9)),
            float(np.mean(interior < 0.08)),
            float(np.mean(interior > 0.92)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, orientation, np.asarray(block_values, dtype=np.float32)])


def _patch_descriptors(gray: np.ndarray) -> np.ndarray:
    descriptors = []
    for y in range(0, 193, 64):
        for x in range(0, 193, 64):
            descriptors.append(_descriptor(gray[y:y + 64, x:x + 64]))
    return np.asarray(descriptors, dtype=np.float32)


def _standardize(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (values - mean) / scale


def _similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    distance /= math.sqrt(left.shape[-1])
    return np.exp(-2.5 * distance)


class TextureMatcher:
    """Match source-image patches against cataloged texture prototypes."""

    def __init__(
        self,
        texture_dir: Path,
        catalog_path: Path,
        clean_reference_paths: list[Path],
    ) -> None:
        self.texture_dir = Path(texture_dir)
        self.catalog_path = Path(catalog_path)
        catalog = _read_csv(self.catalog_path)
        self.catalog_by_damage: dict[str, list[dict[str, str]]] = defaultdict(list)
        texture_descriptors: list[np.ndarray] = []
        texture_labels: list[str] = []
        texture_files: list[str] = []

        for row in catalog:
            if row.get("enabled") != "1":
                continue
            sub_label = row.get("sub_label", "")
            damage_labels = TEXTURE_SUBLABEL_TO_DAMAGE.get(sub_label, ())
            if not damage_labels:
                continue
            path = self.texture_dir / row["file"]
            patches = _patch_descriptors(_read_gray(path))
            for patch in patches:
                texture_descriptors.append(patch)
                texture_labels.append(sub_label)
                texture_files.append(row["file"])
            for damage_name in damage_labels:
                self.catalog_by_damage[damage_name].append(row)

        if not texture_descriptors:
            raise ValueError("No enabled texture descriptors available")

        all_texture = np.asarray(texture_descriptors, dtype=np.float32)
        self.mean = all_texture.mean(axis=0)
        self.scale = all_texture.std(axis=0)
        self.scale[self.scale < 1e-4] = 1.0
        self.texture_patches = _standardize(all_texture, self.mean, self.scale)
        self.texture_patch_labels = texture_labels
        self.texture_patch_files = texture_files
        self.prototypes: dict[str, np.ndarray] = {}
        self.file_descriptors: dict[str, np.ndarray] = {}

        for damage_name in TEXTURE_DAMAGE_TYPES:
            rows = self.catalog_by_damage.get(damage_name, [])
            if not rows:
                continue
            indices = [
                index
                for index, sub_label in enumerate(texture_labels)
                if sub_label in {
                    row.get("sub_label", "")
                    for row in rows
                }
            ]
            self.prototypes[damage_name] = self.texture_patches[indices]
            file_values = []
            file_names = []
            for row in rows:
                path = self.texture_dir / row["file"]
                file_values.append(
                    _standardize(
                        _descriptor(_read_gray(path))[None, :],
                        self.mean,
                        self.scale,
                    )[0]
                )
                file_names.append(row["file"])
            self.file_descriptors[damage_name] = (
                np.asarray(file_values, dtype=np.float32),
                file_names,
            )

        self.thresholds = self._calibrate(clean_reference_paths)

    def _score_patches(self, source: np.ndarray, damage_name: str) -> float:
        similarities = _similarity(source, self.prototypes[damage_name])
        return float(similarities.max())

    def _calibrate(self, clean_reference_paths: list[Path]) -> dict[str, float]:
        clean_patches = [
            _standardize(
                _patch_descriptors(_read_gray(path)),
                self.mean,
                self.scale,
            )
            for path in clean_reference_paths
        ]
        thresholds: dict[str, float] = {}
        for damage_name in TEXTURE_DAMAGE_TYPES:
            if damage_name not in self.prototypes:
                thresholds[damage_name] = 1.0
                continue
            clean_scores = [
                self._score_patches(source, damage_name)
                for source in clean_patches
            ]
            baseline = max(clean_scores) if clean_scores else 0.0
            thresholds[damage_name] = min(
                0.60,
                baseline + TEXTURE_THRESHOLD_MARGIN,
            )
        return thresholds

    def classify(self, path: Path) -> dict[str, object]:
        source = _standardize(
            _patch_descriptors(_read_gray(path)),
            self.mean,
            self.scale,
        )
        scores: dict[str, float] = {}
        files: dict[str, list[str]] = {}
        labels: list[str] = []
        for damage_name in TEXTURE_DAMAGE_TYPES:
            score = self._score_patches(source, damage_name)
            scores[damage_name] = round(score, 4)
            threshold = self.thresholds.get(damage_name, 1.0)
            if score >= threshold:
                labels.append(damage_name)
                file_values, file_names = self.file_descriptors[damage_name]
                similarity = _similarity(source, file_values).max(axis=0)
                top = np.argsort(-similarity)[:3]
                files[damage_name] = [file_names[index] for index in top]
        return {
            "labels": labels,
            "scores": scores,
            "files": files,
        }


def _confidence(score: float, threshold: float, positive: bool) -> float:
    if positive:
        margin = score - threshold
    else:
        margin = threshold - score
    return round(float(np.clip(0.50 + 0.49 * math.tanh(margin * 12.0), 0.50, 0.99)), 4)


def _source_label(
    path: Path,
    row: dict[str, str],
    matcher: TextureMatcher,
) -> dict[str, object]:
    features, color_mode = _feature_vector(path)
    source_evidence = _damage_evidence(features, color_mode)
    texture = matcher.classify(path)
    texture_labels = set(texture["labels"])
    final_labels: dict[str, int] = {}
    confidence: dict[str, float] = {}
    evidence: dict[str, float] = {}

    for name in TEXTURE_DAMAGE_TYPES:
        score = float(texture["scores"].get(name, 0.0))
        threshold = matcher.thresholds.get(name, 1.0)
        positive = name in texture_labels
        final_labels[name] = int(positive)
        confidence[name] = _confidence(score, threshold, positive)
        evidence[name] = round(score, 4)

    for name in NON_TEXTURE_DAMAGE_TYPES:
        score = float(source_evidence[name])
        threshold = NON_TEXTURE_THRESHOLDS[name]
        positive = score >= threshold
        final_labels[name] = int(positive)
        confidence[name] = _confidence(score, threshold, positive)
        evidence[name] = round(score, 4)

    if row.get("quality_label") == "clean_target_ok":
        for name in DAMAGE_TYPES:
            final_labels[name] = 0
            confidence[name] = 0.99
            evidence[name] = 0.0
        texture_labels.clear()

    label_list = "|".join(name for name in DAMAGE_TYPES if final_labels[name])
    notes = ""
    if color_mode == "black_white":
        notes = "black_white_is_source_attribute_not_damage"
    return {
        "source_id": row.get("source_id", ""),
        "filename": row.get("filename", ""),
        "quality_label": row.get("quality_label", ""),
        "color_mode": color_mode,
        "damage_label_method": "texture_catalog_match_v1+source_attribute_v1",
        "damage_label_list": label_list,
        **{f"damage_{name}": final_labels[name] for name in DAMAGE_TYPES},
        **{
            f"confidence_{name}": f"{confidence[name]:.4f}"
            for name in DAMAGE_TYPES
        },
        **{
            f"evidence_{name}": f"{evidence[name]:.4f}"
            for name in DAMAGE_TYPES
        },
        "texture_match_label_list": "|".join(sorted(texture_labels)),
        "texture_match_scores": json.dumps(
            texture["scores"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "texture_match_files": json.dumps(
            texture["files"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "notes": notes,
    }


def recheck_labels(
    images_dir: Path,
    source_manifest: Path,
    texture_dir: Path,
    texture_catalog: Path,
    output: Path,
) -> dict[str, object]:
    source_rows = _read_csv(Path(source_manifest))
    clean_paths = [
        Path(images_dir) / row["filename"]
        for row in source_rows
        if row.get("quality_label") == "clean_target_ok"
    ]
    matcher = TextureMatcher(texture_dir, texture_catalog, clean_paths)
    rows: list[dict[str, object]] = []
    for row in source_rows:
        path = Path(images_dir) / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source image: {path}")
        rows.append(_source_label(path, row, matcher))
    _write_csv(output, rows)

    counts = Counter(
        name
        for row in rows
        for name in DAMAGE_TYPES
        if row[f"damage_{name}"] == 1
    )
    summary = {
        "method": "texture_catalog_match_v1+source_attribute_v1",
        "source_rows": len(rows),
        "clean_target_rows": sum(
            row.get("quality_label") == "clean_target_ok"
            for row in source_rows
        ),
        "damage_counts": dict(counts),
        "texture_damage_types": list(TEXTURE_DAMAGE_TYPES),
        "source_attribute_damage_types": list(NON_TEXTURE_DAMAGE_TYPES),
        "texture_thresholds": matcher.thresholds,
        "catalog_path": str(texture_catalog),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recheck source labels against texture materials")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, default=DEFAULT_TEXTURE_DIR)
    parser.add_argument(
        "--texture-catalog",
        type=Path,
        default=Path(__file__).with_name("texture_catalog.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = recheck_labels(
        args.images,
        args.source_manifest,
        args.texture_dir,
        args.texture_catalog,
        args.output,
    )
    if args.summary:
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

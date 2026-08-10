"""Audit image datasets and generate deterministic visual contact sheets."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photo_revival.paths import ARTIFACTS_DIR, DATASETS_DIR


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REPORT_DIR = ARTIFACTS_DIR / "reports" / "dataset_audit"


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(gray: np.ndarray, hash_size: int = 12) -> str:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return np.packbits(bits).tobytes().hex()


def inspect_directory(directory: Path) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_groups: dict[str, list[str]] = defaultdict(list)

    for path in image_files(directory):
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("OpenCV could not decode image")
            height, width = bgr.shape[:2]
            preview = bgr
            if max(width, height) > 768:
                scale = 768 / max(width, height)
                preview = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
            sha = file_sha256(path)
            phash = difference_hash(gray)
            exact_groups[sha].append(path.name)
            perceptual_groups[phash].append(path.name)
            records.append(
                {
                    "name": path.name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4),
                    "file_size": path.stat().st_size,
                    "brightness": round(float(gray.mean()), 3),
                    "contrast": round(float(gray.std()), 3),
                    "saturation": round(float(hsv[:, :, 1].mean()), 3),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                }
            )
        except Exception as exc:  # Keep auditing after an isolated bad file.
            errors.append({"name": path.name, "error": str(exc)})

    def describe(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {}
        return {
            "min": round(float(array.min()), 3),
            "median": round(float(np.median(array)), 3),
            "max": round(float(array.max()), 3),
            "mean": round(float(array.mean()), 3),
        }

    resolutions = Counter(f"{item['width']}x{item['height']}" for item in records)
    summary = {
        "directory": str(directory.relative_to(DATASETS_DIR)),
        "count": len(records),
        "decode_errors": errors,
        "resolution_count": len(resolutions),
        "most_common_resolutions": resolutions.most_common(10),
        "width": describe([item["width"] for item in records]),
        "height": describe([item["height"] for item in records]),
        "aspect_ratio": describe([item["aspect_ratio"] for item in records]),
        "file_size": describe([item["file_size"] for item in records]),
        "brightness": describe([item["brightness"] for item in records]),
        "contrast": describe([item["contrast"] for item in records]),
        "saturation": describe([item["saturation"] for item in records]),
        "sharpness": describe([item["sharpness"] for item in records]),
        "very_small_images": [
            item["name"] for item in records if item["width"] < 256 or item["height"] < 256
        ],
        "below_512_images": [
            item["name"] for item in records if item["width"] < 512 or item["height"] < 512
        ],
        "extreme_aspect_images": [
            item["name"] for item in records if item["aspect_ratio"] < 0.45 or item["aspect_ratio"] > 2.2
        ],
        "exact_duplicate_groups": [names for names in exact_groups.values() if len(names) > 1],
        "same_dhash_groups": [names for names in perceptual_groups.values() if len(names) > 1],
    }
    return summary, records


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def save_sheet(rows: list[tuple[str, list[Path]]], output: Path, columns: list[str]) -> None:
    cell_width, cell_height, label_height = 280, 230, 28
    width = cell_width * len(columns)
    height = label_height + len(rows) * (cell_height + label_height)
    sheet = Image.new("RGB", (width, height), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, title in enumerate(columns):
        draw.text((column * cell_width + 8, 8), title, fill="black", font=font)
    for row_index, (label, paths) in enumerate(rows):
        y = label_height + row_index * (cell_height + label_height)
        for column, path in enumerate(paths):
            sheet.paste(fit_image(path, (cell_width, cell_height)), (column * cell_width, y))
        draw.text((8, y + cell_height + 7), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def synthetic_pairing(root: Path) -> dict:
    clean = {path.stem: path for path in image_files(root / "Train_GT_Clean")}
    damaged = {path.stem: path for path in image_files(root / "Train_GT_Damaged")}
    degraded: dict[str, list[Path]] = defaultdict(list)
    for path in image_files(root / "Train_Input_Degraded"):
        base = re.sub(r"_\([^)]*\)$", "", path.stem)
        degraded[base].append(path)
    return {
        "clean_count": len(clean),
        "damaged_count": len(damaged),
        "degraded_count": sum(map(len, degraded.values())),
        "clean_without_damaged": sorted(set(clean) - set(damaged)),
        "damaged_without_clean": sorted(set(damaged) - set(clean)),
        "clean_without_degraded": sorted(set(clean) - set(degraded)),
        "degraded_variants_per_clean": dict(sorted(Counter(map(len, degraded.values())).items())),
    }


def configured_training_quality() -> dict:
    root = DATASETS_DIR / "archive" / "03_Synthetic_Dataset"
    clean_dir = root / "Train_GT_Clean"
    degraded_dir = root / "Train_Input_Degraded"
    mask_dir = ARTIFACTS_DIR / "cache" / "bopbtl_masks"
    mask_paths = {path.name: path for path in mask_dir.rglob("*.jpg")}
    metrics = []
    mask_coverage = []
    dimensions_match = 0

    for clean_path in image_files(clean_dir):
        degraded_path = degraded_dir / clean_path.name
        clean = cv2.imdecode(np.fromfile(clean_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        degraded = cv2.imdecode(np.fromfile(degraded_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if clean.shape == degraded.shape:
            dimensions_match += 1
        difference = clean.astype(np.float32) - degraded.astype(np.float32)
        mse = float(np.mean(difference**2))
        metrics.append(
            {
                "name": clean_path.name,
                "mae": float(np.mean(np.abs(difference))),
                "psnr": 99.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse))),
                "mean_color_shift": float(
                    np.mean(np.abs(clean.mean(axis=(0, 1)) - degraded.mean(axis=(0, 1))))
                ),
            }
        )
        mask_path = mask_paths.get(clean_path.name)
        if mask_path:
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask_coverage.append(float((mask > 127).mean()))

    def percentile(values: list[float]) -> dict:
        array = np.asarray(values)
        return {
            "min": round(float(array.min()), 4),
            "p10": round(float(np.percentile(array, 10)), 4),
            "median": round(float(np.median(array)), 4),
            "p90": round(float(np.percentile(array, 90)), 4),
            "max": round(float(array.max()), 4),
            "mean": round(float(array.mean()), 4),
        }

    return {
        "pair_count": len(metrics),
        "matching_dimensions": dimensions_match,
        "mae": percentile([item["mae"] for item in metrics]),
        "psnr_db": percentile([item["psnr"] for item in metrics]),
        "mean_color_shift": percentile([item["mean_color_shift"] for item in metrics]),
        "mask_count": len(mask_coverage),
        "mask_coverage": percentile(mask_coverage),
        "empty_masks": sum(value == 0 for value in mask_coverage),
        "masks_below_1_percent": sum(value < 0.01 for value in mask_coverage),
        "masks_above_20_percent": sum(value > 0.2 for value in mask_coverage),
    }


def create_visual_samples() -> None:
    rng = random.Random(20260719)
    synthetic = DATASETS_DIR / "03_Synthetic_Dataset"
    clean_files = image_files(synthetic / "Train_GT_Clean")
    selected = rng.sample(clean_files, min(12, len(clean_files)))
    synthetic_rows = []
    for clean in selected:
        variants = sorted((synthetic / "Train_Input_Degraded").glob(f"{clean.stem}_*.jpg"))
        synthetic_rows.append((clean.stem, [clean, synthetic / "Train_GT_Damaged" / clean.name, variants[0]]))
    save_sheet(
        synthetic_rows,
        REPORT_DIR / "synthetic_pairs.jpg",
        ["Clean target", "Damaged target", "Degraded input"],
    )

    archive = DATASETS_DIR / "archive" / "03_Synthetic_Dataset"
    archive_clean = image_files(archive / "Train_GT_Clean")
    selected = rng.sample(archive_clean, min(12, len(archive_clean)))
    archive_rows = [
        (
            clean.stem,
            [clean, archive / "Train_Input_Degraded" / clean.name],
        )
        for clean in selected
    ]
    save_sheet(archive_rows, REPORT_DIR / "configured_training_pairs.jpg", ["Clean target", "Training input"])

    mask_paths = {path.name: path for path in (ARTIFACTS_DIR / "cache" / "bopbtl_masks").rglob("*.jpg")}
    mask_rows = [
        (
            clean.stem,
            [archive / "Train_Input_Degraded" / clean.name, mask_paths[clean.name]],
        )
        for clean in selected
        if clean.name in mask_paths
    ]
    save_sheet(mask_rows, REPORT_DIR / "configured_training_masks.jpg", ["Training input", "BOPBTL mask"])

    photos = DATASETS_DIR / "photo_dataset"
    damaged_files = image_files(photos / "damaged_photos")
    selected = rng.sample(damaged_files, min(12, len(damaged_files)))
    restoration_rows = []
    for damaged in selected:
        base = damaged.stem
        restoration_rows.append(
            (
                base,
                [
                    damaged,
                    photos / "restored_photos" / f"{base}_mask.jpg",
                    photos / "restored_photos" / f"{base}_restored.jpg",
                ],
            )
        )
    save_sheet(restoration_rows, REPORT_DIR / "restoration_samples.jpg", ["Damaged", "Mask", "Restored"])


def main() -> None:
    directories = [
        DATASETS_DIR / "03_Synthetic_Dataset" / "Train_GT_Clean",
        DATASETS_DIR / "03_Synthetic_Dataset" / "Train_GT_Damaged",
        DATASETS_DIR / "03_Synthetic_Dataset" / "Train_Input_Degraded",
        DATASETS_DIR / "photo_dataset" / "clean_photos",
        DATASETS_DIR / "photo_dataset" / "damaged_photos",
        DATASETS_DIR / "photo_dataset" / "restored_photos",
        DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean",
        DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded",
    ]
    report = {
        "directories": {},
        "pairing": synthetic_pairing(DATASETS_DIR / "03_Synthetic_Dataset"),
        "configured_training_quality": configured_training_quality(),
    }
    for directory in directories:
        summary, _ = inspect_directory(directory)
        report["directories"][summary["directory"]] = summary

    archive_root = DATASETS_DIR / "archive" / "03_Synthetic_Dataset"
    main_root = DATASETS_DIR / "03_Synthetic_Dataset"
    report["archive_comparison"] = {}
    for folder in ("Train_GT_Clean", "Train_Input_Degraded"):
        main_hashes = {file_sha256(path) for path in image_files(main_root / folder)}
        archive_hashes = {file_sha256(path) for path in image_files(archive_root / folder)}
        report["archive_comparison"][folder] = {
            "main_count": len(main_hashes),
            "archive_count": len(archive_hashes),
            "archive_fully_duplicated_in_main": archive_hashes <= main_hashes,
            "archive_unique_files": len(archive_hashes - main_hashes),
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    create_visual_samples()
    print(report_path)


if __name__ == "__main__":
    main()

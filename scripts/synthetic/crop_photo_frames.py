"""Detect framed historical photos and conservatively crop the photo area.

The detector is intentionally conservative. It crops only when a strong,
axis-aligned inner-frame candidate is found. The crop is expanded by a safety
margin so border damage and edge texture are retained. Images without a
reliable candidate are copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class FrameCandidate:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float
    geometry_score: float
    contrast_score: float
    centered_score: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image)


def _resize_for_detection(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _rectangularity(points: np.ndarray) -> float:
    area = abs(float(cv2.contourArea(points)))
    x, y, width, height = cv2.boundingRect(points)
    box_area = float(width * height)
    return float(np.clip(area / max(box_area, 1.0), 0.0, 1.0))


def _candidate_score(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[float, float, float, float]:
    height, width = gray.shape
    box_width = x1 - x0
    box_height = y1 - y0
    area_ratio = (box_width * box_height) / float(width * height)
    border = max(3, round(min(box_width, box_height) * 0.025))
    inside = gray[y0:y1, x0:x1]
    if inside.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    strips = [
        gray[max(0, y0 - border):y0, x0:x1],
        gray[y1:min(height, y1 + border), x0:x1],
        gray[y0:y1, max(0, x0 - border):x0],
        gray[y0:y1, x1:min(width, x1 + border)],
    ]
    strips = [strip for strip in strips if strip.size]
    outer_mean = float(np.mean(np.concatenate([strip.ravel() for strip in strips])))
    inner_mean = float(np.mean(inside))
    contrast_score = float(np.clip(abs(outer_mean - inner_mean) / 55.0, 0.0, 1.0))

    margin_x = min(x0, width - x1) / max(width, 1)
    margin_y = min(y0, height - y1) / max(height, 1)
    centered_score = float(np.clip((margin_x + margin_y) / 0.18, 0.0, 1.0))

    size_score = float(np.clip((area_ratio - 0.16) / 0.58, 0.0, 1.0))
    border_presence = float(np.clip(min(margin_x, margin_y) / 0.05, 0.0, 1.0))
    geometry_score = 0.65 * size_score + 0.35 * border_presence
    confidence = 0.50 * geometry_score + 0.35 * contrast_score + 0.15 * centered_score
    return confidence, geometry_score, contrast_score, centered_score


def detect_frame(image: np.ndarray, min_confidence: float = 0.62) -> FrameCandidate | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    candidates: list[FrameCandidate] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        if box_width < width * 0.25 or box_height < height * 0.25:
            continue
        if box_width > width * 0.96 or box_height > height * 0.96:
            continue
        if min(x, y, width - (x + box_width), height - (y + box_height)) < min(width, height) * 0.012:
            continue
        rectangularity = _rectangularity(polygon)
        if rectangularity < 0.82:
            continue

        confidence, geometry, contrast, centered = _candidate_score(
            gray, x, y, x + box_width, y + box_height
        )
        confidence = float(np.clip(confidence * (0.75 + 0.25 * rectangularity), 0.0, 1.0))
        if confidence < min_confidence:
            continue

        # Expand in source coordinates to keep photo-edge damage and texture.
        safety = max(4, round(min(box_width, box_height) * 0.035))
        x0 = max(0, x - safety)
        y0 = max(0, y - safety)
        x1 = min(width, x + box_width + safety)
        y1 = min(height, y + box_height + safety)
        candidates.append(
            FrameCandidate(
                x0=round(x0 / scale),
                y0=round(y0 / scale),
                x1=round(x1 / scale),
                y1=round(y1 / scale),
                confidence=round(confidence, 4),
                geometry_score=round(geometry, 4),
                contrast_score=round(contrast, 4),
                centered_score=round(centered, 4),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.confidence, item.width * item.height))


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] | None = None,
) -> None:
    fields = fields or [
        "source_id",
        "dataset_id",
        "dataset_class",
        "original_filename",
        "output_filename",
        "original_path",
        "output_path",
        "detected",
        "confidence",
        "geometry_score",
        "contrast_score",
        "centered_score",
        "original_width",
        "original_height",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_width",
        "crop_height",
        "crop_margin_ratio",
        "crop_reason",
        "output_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_sources(
    images_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    min_confidence: float = 0.62,
) -> dict[str, object]:
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results: list[dict[str, object]] = []
    detected_count = 0
    for row in rows:
        source_id = row.get("source_id", "")
        filename = row.get("filename", "")
        source = images_dir / filename
        if not source.is_file():
            continue
        image = _read_image(source)
        height, width = image.shape[:2]
        candidate = detect_frame(image, min_confidence=min_confidence)
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if candidate is None:
            shutil.copy2(source, destination)
            results.append({
                "source_id": source_id,
                "original_filename": filename,
                "output_filename": filename,
                "detected": 0,
                "confidence": "",
                "geometry_score": "",
                "contrast_score": "",
                "centered_score": "",
                "original_width": width,
                "original_height": height,
                "crop_x0": 0,
                "crop_y0": 0,
                "crop_x1": width,
                "crop_y1": height,
                "crop_width": width,
                "crop_height": height,
                "crop_margin_ratio": 0,
                "crop_reason": "no_reliable_inner_frame",
            })
            continue

        detected_count += 1
        crop = image[candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        Image.fromarray(crop).save(destination)
        results.append({
            "source_id": source_id,
            "original_filename": filename,
            "output_filename": filename,
            "detected": 1,
            "confidence": candidate.confidence,
            "geometry_score": candidate.geometry_score,
            "contrast_score": candidate.contrast_score,
            "centered_score": candidate.centered_score,
            "original_width": width,
            "original_height": height,
            "crop_x0": candidate.x0,
            "crop_y0": candidate.y0,
            "crop_x1": candidate.x1,
            "crop_y1": candidate.y1,
            "crop_width": candidate.width,
            "crop_height": candidate.height,
            "crop_margin_ratio": round(
                min(candidate.x0, candidate.y0, width - candidate.x1, height - candidate.y1)
                / min(width, height),
                4,
            ),
            "crop_reason": "inner_frame_detected_with_safety_margin",
        })

    index_path = output_dir / "frame_crop_index.csv"
    _write_csv(index_path, results)
    summary = {
        "source_rows": len(rows),
        "processed_rows": len(results),
        "detected_and_cropped": detected_count,
        "unchanged": len(results) - detected_count,
        "min_confidence": min_confidence,
        "safety_policy": "expand detected inner frame by 3.5 percent of the smaller image dimension",
        "output_dir": str(output_dir),
        "index": str(index_path),
    }
    (output_dir / "frame_crop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def crop_dataset(
    dataset_root: Path,
    dataset_index_path: Path,
    output_root: Path,
    min_confidence: float = 0.62,
) -> dict[str, object]:
    """Create a cropped sibling of the formal two-class dataset."""
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_images = output_root / "dataset_images_cropped"
    output_images.mkdir(parents=True, exist_ok=True)
    with Path(dataset_index_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    crop_rows: list[dict[str, object]] = []
    cropped_index_rows: list[dict[str, object]] = []
    detected_count = 0
    for row in rows:
        relative_source = Path(row["export_path"])
        source = dataset_root / relative_source
        if not source.is_file():
            raise FileNotFoundError(f"Dataset image not found: {source}")
        image = _read_image(source)
        height, width = image.shape[:2]
        candidate = detect_frame(image, min_confidence=min_confidence)
        destination_relative = Path("dataset_images_cropped") / relative_source.relative_to("dataset_images")
        destination = output_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if candidate is None:
            shutil.copy2(source, destination)
            crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, width, height
            confidence = ""
            geometry_score = ""
            contrast_score = ""
            centered_score = ""
            detected = 0
            reason = "no_reliable_inner_frame"
        else:
            detected_count += 1
            crop = image[candidate.y0:candidate.y1, candidate.x0:candidate.x1]
            Image.fromarray(crop).save(destination)
            crop_x0, crop_y0 = candidate.x0, candidate.y0
            crop_x1, crop_y1 = candidate.x1, candidate.y1
            confidence = candidate.confidence
            geometry_score = candidate.geometry_score
            contrast_score = candidate.contrast_score
            centered_score = candidate.centered_score
            detected = 1
            reason = "inner_frame_detected_with_safety_margin"

        output_height = crop_y1 - crop_y0
        output_width = crop_x1 - crop_x0
        crop_record = {
            "source_id": row.get("source_id", ""),
            "dataset_id": row.get("dataset_id", ""),
            "dataset_class": row.get("dataset_class", ""),
            "original_filename": row.get("export_filename", ""),
            "output_filename": destination.name,
            "original_path": row.get("export_path", ""),
            "output_path": destination_relative.as_posix(),
            "detected": detected,
            "confidence": confidence,
            "geometry_score": geometry_score,
            "contrast_score": contrast_score,
            "centered_score": centered_score,
            "original_width": width,
            "original_height": height,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "crop_width": output_width,
            "crop_height": output_height,
            "crop_margin_ratio": round(
                min(crop_x0, crop_y0, width - crop_x1, height - crop_y1)
                / min(width, height),
                4,
            ),
            "crop_reason": reason,
            "output_sha256": _sha256(destination),
        }
        crop_rows.append(crop_record)

        updated = dict(row)
        updated["export_path"] = destination_relative.as_posix()
        updated["export_filename"] = destination.name
        updated["width"] = output_width
        updated["height"] = output_height
        cropped_index_rows.append({**updated, **{
            "frame_detected": detected,
            "frame_confidence": confidence,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "crop_width": output_width,
            "crop_height": output_height,
            "crop_margin_ratio": crop_record["crop_margin_ratio"],
            "cropped_sha256": crop_record["output_sha256"],
        }})

    frame_index = output_root / "frame_crop_index.csv"
    _write_csv(frame_index, crop_rows)
    index_fields = list(cropped_index_rows[0]) if cropped_index_rows else []
    _write_csv(output_root / "dataset_index_cropped.csv", cropped_index_rows, index_fields)
    for class_name in ("damaged_obvious", "clean_near"):
        _write_csv(
            output_root / f"{class_name}_index_cropped.csv",
            [row for row in cropped_index_rows if row.get("dataset_class") == class_name],
            index_fields,
        )
    summary = {
        "input_dataset_records": len(rows),
        "detected_and_cropped": detected_count,
        "unchanged": len(rows) - detected_count,
        "min_confidence": min_confidence,
        "safety_policy": "expand detected inner frame by 3.5 percent of the smaller image dimension",
        "input_index": str(dataset_index_path),
        "output_images": str(output_images),
        "index": str(frame_index),
        "dataset_index": str(output_root / "dataset_index_cropped.csv"),
    }
    (output_root / "frame_crop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservatively crop photos inside detected frames")
    parser.add_argument("--images", type=Path, help="Source image directory used with --manifest")
    parser.add_argument("--manifest", type=Path, help="Source manifest used with --images")
    parser.add_argument("--dataset-index", type=Path, help="Formal dataset index used with --dataset-root")
    parser.add_argument("--dataset-root", type=Path, help="Package root containing dataset_images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.62)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset_index and args.dataset_root:
        summary = crop_dataset(
            args.dataset_root,
            args.dataset_index,
            args.output_dir,
            args.min_confidence,
        )
    elif args.images and args.manifest:
        summary = crop_sources(
            args.images,
            args.manifest,
            args.output_dir,
            args.min_confidence,
        )
    else:
        raise SystemExit("Use --images with --manifest, or --dataset-root with --dataset-index")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

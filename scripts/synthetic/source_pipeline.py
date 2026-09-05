"""Prepare the source manifest for the C-role dataset workflow.

Damage labels are produced by recheck_damage_labels.py. Physical damage labels
are compared with the existing texture catalog; image-property labels use
source-image evidence for phenomena without a matching material category.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from scripts.synthetic.texture_assets import PROJECT_ROOT

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from scripts.data.team_dataset_workflow import (  # noqa: E402
    IMAGE_SUFFIXES,
    read_csv,
    sha256,
    utc_now,
    write_csv,
)


DAMAGE_TYPES = (
    "scratch",
    "crack",
    "fold",
    "dust",
    "stain",
    "mold",
    "tear",
    "missing_region",
    "fading",
    "color_cast",
    "blur",
    "noise",
    "jpeg_artifact",
    "uneven_exposure",
)

SOURCE_MANIFEST_FIELDS = [
    "seq",
    "origin_filename",
    "source_id",
    "filename",
    "dataset_kind",
    "provider",
    "source_url",
    "license",
    "rights_url",
    "sha256",
    "origin_sha256",
    "width",
    "height",
    "content_type",
    "color_mode",
    "quality_label",
    "split",
    "source_path",
]

def _stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 100


def _split_for(source_id: str) -> str:
    bucket = _stable_bucket(source_id)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _resize_array(image: Image.Image, size: int = 384) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _tile_std(gray: np.ndarray, tiles: int = 8) -> float:
    height, width = gray.shape
    values: list[float] = []
    for y in range(tiles):
        y0 = y * height // tiles
        y1 = max(y0 + 1, (y + 1) * height // tiles)
        for x in range(tiles):
            x0 = x * width // tiles
            x1 = max(x0 + 1, (x + 1) * width // tiles)
            values.append(float(gray[y0:y1, x0:x1].mean()))
    return float(np.std(values))


def _feature_vector(path: Path) -> tuple[dict[str, float], str]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        array = _resize_array(image)
        color_mode = "grayscale" if float(array.max(axis=2).min()) == float(array.min(axis=2).max()) else ""
        if not color_mode:
            chroma = array.max(axis=2) - array.min(axis=2)
            color_mode = "black_white" if float(np.mean(chroma < 0.035)) >= 0.985 else "color"

    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    if min(gray.shape) > 2:
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
    else:
        dx = np.zeros((1, 1), dtype=np.float32)
        dy = np.zeros((1, 1), dtype=np.float32)
    gradient = np.hypot(
        np.pad(dx, ((0, 0), (0, 1))),
        np.pad(dy, ((0, 1), (0, 0))),
    )
    with Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)), "L") as gray_image:
        soft = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255.0
    high = np.abs(gray - soft)
    chroma = array.max(axis=2) - array.min(axis=2)
    dark = gray < max(0.10, _percentile(gray, 8))
    bright = gray > min(0.92, _percentile(gray, 92))
    edge = gradient > max(0.08, _percentile(gradient, 85))
    directional = abs(float(np.mean(np.abs(dx))) - float(np.mean(np.abs(dy))))
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]],
    )
    interior = gray[2:-2, 2:-2] if min(gray.shape) > 4 else gray
    block_width = max(1, gray.shape[1] // 8)
    block_height = max(1, gray.shape[0] // 8)
    block_edges: list[float] = []
    for x in range(block_width, gray.shape[1], block_width):
        block_edges.append(float(np.mean(np.abs(gray[:, x] - gray[:, x - 1]))))
    for y in range(block_height, gray.shape[0], block_height):
        block_edges.append(float(np.mean(np.abs(gray[y, :] - gray[y - 1, :]))))
    edge_ratio = float(np.mean(edge))
    high_ratio = float(np.mean(high > max(0.018, _percentile(high, 80))))
    local_blob = float(np.std(soft))
    exposure = _tile_std(gray)
    border_gap = abs(float(border.mean()) - float(interior.mean()))
    dynamic_range = _percentile(gray, 95) - _percentile(gray, 5)
    features = {
        "contrast": _clip01(dynamic_range / 0.75),
        "edge_ratio": _clip01(edge_ratio / 0.35),
        "high_frequency": _clip01(high_ratio / 0.30),
        "line_structure": _clip01((edge_ratio * 0.7 + directional * 2.0) / 0.45),
        "spot_structure": _clip01((float(np.mean(dark | bright)) + high_ratio) / 0.65),
        "low_frequency_blobs": _clip01(local_blob / 0.18),
        "border_damage": _clip01((border_gap + edge_ratio * 0.35) / 0.45),
        "border_extreme": _clip01(float(np.mean((border < 0.035) | (border > 0.97))) / 0.55),
        "color_cast_strength": _clip01(
            float(np.max(np.mean(array, axis=(0, 1))) - np.min(np.mean(array, axis=(0, 1))))
            / 0.18
        ),
        "blur_strength": _clip01(1.0 - edge_ratio / 0.22),
        "noise_strength": _clip01(
            float(np.std(high)) / 0.075
        ),
        "jpeg_blockiness": _clip01(
            (float(np.mean(block_edges)) if block_edges else 0.0) / 0.12
        ),
        "uneven_exposure": _clip01(exposure / 0.24),
        "dark_area": _clip01(float(np.mean(dark)) / 0.30),
        "bright_area": _clip01(float(np.mean(bright)) / 0.30),
        "interior_extreme": _clip01(float(np.mean((interior < 0.035) | (interior > 0.97))) / 0.35),
        "chroma": _clip01(float(np.mean(chroma)) / 0.28),
    }
    return features, color_mode


def _damage_evidence(features: dict[str, float], color_mode: str) -> dict[str, float]:
    line = features["line_structure"]
    high = features["high_frequency"]
    spots = features["spot_structure"]
    blobs = features["low_frequency_blobs"]
    border = features["border_damage"]
    cast = features["color_cast_strength"] * features["chroma"]
    evidence = {
        "scratch": 0.60 * line + 0.40 * high,
        "crack": 0.50 * line + 0.25 * high + 0.25 * features["dark_area"],
        "fold": 0.60 * line + 0.40 * features["border_damage"],
        "dust": 0.65 * spots + 0.35 * high,
        "stain": 0.65 * blobs + 0.35 * spots,
        "mold": 0.50 * blobs + 0.30 * spots + 0.20 * features["dark_area"],
        "tear": 0.55 * border + 0.25 * features["border_extreme"] + 0.20 * line,
        "missing_region": 0.55 * features["interior_extreme"] + 0.45 * border,
        "fading": 1.0 - features["contrast"],
        "color_cast": 0.0 if color_mode == "black_white" else cast,
        "blur": features["blur_strength"],
        "noise": 0.60 * features["noise_strength"] + 0.40 * high,
        "jpeg_artifact": 0.65 * features["jpeg_blockiness"] + 0.35 * high,
        "uneven_exposure": features["uneven_exposure"],
    }
    return {name: _clip01(value) for name, value in evidence.items()}


def _clean_source_row(
    row: dict[str, str],
    image_path: Path,
    source_dir: Path,
) -> dict[str, object]:
    source_id = row.get("source_id") or Path(row.get("filename", image_path.name)).stem
    source_path = os.path.relpath(image_path, source_dir.parent).replace("\\", "/")
    return {
        **{field: row.get(field, "") for field in SOURCE_MANIFEST_FIELDS},
        "source_id": source_id,
        "filename": row.get("filename") or image_path.name,
        "split": _split_for(source_id),
        "source_path": source_path,
    }


def _duplicate_fields(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = row.get("sha256", "").strip()
        if digest:
            groups[digest].append(row)
    output: dict[str, dict[str, object]] = {}
    for digest, members in groups.items():
        members = sorted(members, key=lambda item: (item.get("seq", ""), item.get("source_id", "")))
        if len(members) == 1:
            row = members[0]
            output[row.get("source_id", "")] = {
                "duplicate_group": "",
                "duplicate_role": "unique",
                "duplicate_count": 1,
                "canonical_source_id": row.get("source_id", ""),
                "canonical_filename": row.get("filename", ""),
            }
            continue
        group_id = f"sha256:{digest[:12]}"
        canonical = members[0]
        for index, row in enumerate(members):
            output[row.get("source_id", "")] = {
                "duplicate_group": group_id,
                "duplicate_role": "canonical" if index == 0 else "duplicate",
                "duplicate_count": len(members),
                "canonical_source_id": canonical.get("source_id", ""),
                "canonical_filename": canonical.get("filename", ""),
            }
    return output


def prepare_sources(
    images_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Write the cleaned source manifest under output_dir.

    C damage labels are generated separately by recheck_damage_labels.py,
    which compares physical damage against the existing texture catalog.
    """
    images_dir = Path(images_dir)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(manifest_path)
    duplicate_info = _duplicate_fields(rows)
    clean_rows: list[dict[str, object]] = []
    missing: list[str] = []
    excluded_duplicates: list[str] = []
    for row in rows:
        filename = row.get("filename", "")
        image_path = images_dir / filename
        if not image_path.is_file():
            missing.append(filename)
            continue
        source_id = row.get("source_id") or Path(filename).stem
        info = duplicate_info.get(source_id, {
            "duplicate_group": "",
            "duplicate_role": "unique",
            "duplicate_count": 1,
            "canonical_source_id": source_id,
            "canonical_filename": filename,
        })
        if info.get("duplicate_role") == "duplicate":
            excluded_duplicates.append(source_id)
            continue
        clean_row = _clean_source_row(row, image_path, images_dir)
        clean_rows.append(clean_row)
    write_csv(output_dir / "source_manifest.csv", clean_rows, SOURCE_MANIFEST_FIELDS)

    duplicate_groups = Counter(
        row["duplicate_group"] for row in clean_rows if row.get("duplicate_group")
    )
    summary = {
        "created_at": utc_now(),
        "images_dir": str(images_dir),
        "manifest_path": str(manifest_path),
        "counts": {
            "manifest_rows": len(rows),
            "processed_rows": len(clean_rows),
            "missing_images": len(missing),
            "excluded_duplicate_rows": len(excluded_duplicates),
            "existing_damage": sum(row.get("quality_label") == "existing_damage" for row in clean_rows),
            "clean_target_ok": sum(row.get("quality_label") == "clean_target_ok" for row in clean_rows),
            "duplicate_groups": len(duplicate_groups),
        },
        "missing_filenames": missing,
        "excluded_duplicate_source_ids": excluded_duplicates,
        "duplicate_groups": dict(duplicate_groups),
        "classifier": {
            "method": "delegated_to_texture_catalog_match_v1+source_attribute_v1",
            "black_white_policy": "color_mode_only",
            "labels": list(DAMAGE_TYPES),
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the C-role source manifest")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = prepare_sources(args.images, args.manifest, args.output_dir)
    print(
        f"prepared {summary['counts']['processed_rows']} source rows; "
        f"{summary['counts']['clean_target_ok']} clean target candidates -> {args.output_dir}; "
        "run recheck_damage_labels.py for C labels"
    )


if __name__ == "__main__":
    main()

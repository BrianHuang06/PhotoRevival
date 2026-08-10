"""Build and review the material-only texture catalog.

The source texture pack uses numeric filenames and does not provide semantic
labels. The semantic labels in this script are assigned by visual review.
Measured image features are retained only as auditable material-selection
fields; they do not decide the damage label.

Run from the repository root:

    python -m scripts.synthetic.analyze_textures
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.synthetic.texture_assets import DEFAULT_TEXTURE_DIR, list_texture_paths


DEFAULT_CATALOG_PATH = Path(__file__).with_name("texture_catalog.csv")
DEFAULT_REVIEW_DIR = Path(__file__).with_name("texture_reviews")

# Main label -> sub-labels. The path in the catalog is written as
# ``main_label--sub_label`` so the hierarchy is explicit to downstream code.
LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "detail": ("scratch", "crack", "dust"),
    "area": ("stain", "mold", "surface_wear"),
    "structure": ("edge_wear", "fold", "tear_missing"),
    "reject": ("reject",),
}
CATEGORIES = tuple(
    sub_label
    for sub_labels in LABEL_GROUPS.values()
    for sub_label in sub_labels
)
SUB_LABEL_TO_MAIN = {
    sub_label: main_label
    for main_label, sub_labels in LABEL_GROUPS.items()
    for sub_label in sub_labels
}

# Exactly one primary sub-label is assigned to every numbered source by visual
# review. Overlapping visual appearances are recorded in AUXILIARY_TAGS below.
MANUAL_PRIMARY: dict[str, set[int]] = {
    "surface_wear": {
        1, 3, 20, 22, 26, 28, 34, 38, 43, 44, 46, 49, 52, 58, 61, 63, 65,
        66, 73, 76, 77, 78, 82, 88, 95, 96, 104, 114, 130, 134, 140, 143,
        147, 148, 155, 157, 159, 161, 178, 186, 187, 189, 190, 195, 196,
        199, 200, 205, 206, 210, 218, 226, 228, 241,
    },
    "mold": {
        2, 6, 8, 16, 23, 39, 41, 42, 45, 47, 62, 64, 67, 75, 80, 85, 87,
        97, 103, 110, 121, 138, 142, 145, 151, 156, 180, 182, 193, 197,
        198, 207, 216, 217, 223, 225, 231, 234, 238,
    },
    "stain": {
        5, 7, 10, 12, 13, 21, 25, 29, 33, 40, 68, 74, 83, 84, 90, 91, 94,
        98, 101, 111, 112, 113, 118, 119, 125, 132, 137, 149, 154, 164,
        165, 167, 171, 192, 224, 229, 230, 232, 237, 239, 249,
    },
    "dust": {
        15, 18, 19, 27, 30, 50, 54, 60, 69, 71, 79, 86, 92, 100, 102, 108,
        129, 136, 139, 144, 152, 160, 162, 163, 168, 169, 172, 179, 183,
        188, 202, 204, 211, 212, 213, 214, 215, 219, 220, 227, 235, 244,
        245, 250,
    },
    "scratch": {
        9, 17, 31, 35, 48, 53, 72, 93, 106, 117, 122, 124, 127, 181, 209,
        243, 246,
    },
    "crack": {109, 116, 135, 185, 240},
    "fold": {
        32, 36, 59, 89, 105, 123, 126, 131, 153, 173, 175, 176, 242, 248,
    },
    "edge_wear": {
        4, 11, 14, 24, 37, 55, 56, 57, 70, 81, 99, 107, 120, 128, 133,
        146, 158, 166, 174, 177, 184, 191, 194, 208, 222, 233, 236,
    },
    "tear_missing": {51, 115, 141, 170, 201, 221},
    "reject": {150, 203, 247},
}

# Auxiliary labels are deliberately narrower than the primary map. They keep
# useful overlap information without turning "mixed" into a semantic bucket.
AUXILIARY_TAGS: dict[str, set[int]] = {
    "scratch": {
        11, 17, 24, 31, 35, 48, 53, 72, 93, 96, 106, 117, 122, 124, 127,
        155, 181, 209, 243, 246, 248,
    },
    "crack": {109, 116, 135, 185, 240},
    "fold": {
        32, 36, 59, 89, 105, 123, 126, 131, 153, 173, 175, 176, 242, 248,
    },
    "dust": {
        17, 19, 24, 50, 60, 69, 71, 79, 86, 92, 100, 102, 108, 111, 118,
        127, 129, 136, 139, 144, 152, 160, 162, 163, 168, 169, 172, 179,
        183, 188, 202, 204, 211, 212, 213, 214, 215, 219, 220, 227, 235,
        244, 245, 250,
    },
    "stain": {
        2, 4, 7, 10, 12, 13, 21, 25, 29, 32, 33, 37, 40, 41, 51, 56, 57,
        64, 67, 68, 74, 83, 84, 90, 91, 94, 97, 98, 101, 103, 110, 111,
        112, 113, 118, 119, 121, 125, 132, 137, 149, 154, 164, 165, 167,
        170, 171, 182, 186, 192, 193, 194, 201, 207, 208, 216, 221, 224,
        229, 230, 231, 232, 237, 238, 239, 249,
    },
    "mold": {
        2, 6, 8, 16, 23, 39, 41, 42, 45, 47, 62, 64, 67, 75, 80, 84, 85,
        87, 97, 103, 110, 121, 138, 142, 145, 151, 156, 164, 180, 182, 193,
        197, 198, 207, 216, 217, 223, 225, 231, 234, 238,
    },
    "surface_wear": {
        4, 11, 14, 20, 22, 28, 32, 34, 36, 37, 38, 40, 44, 46, 52, 55, 56,
        58, 61, 63, 65, 66, 73, 74, 76, 77, 78, 81, 82, 88, 94, 95, 98, 99,
        96, 104, 107, 114, 117, 120, 122, 123, 125, 126, 128, 130, 131, 134,
        140, 143, 146, 147, 148, 153, 155, 157, 158, 159, 161, 165, 170, 172, 173,
        175, 176, 177, 178, 184, 186, 187, 189, 190, 191, 194, 195, 196, 199,
        200, 205, 206, 208, 210, 218, 222, 226, 228, 233, 241, 242, 243, 248,
    },
    "edge_wear": {
        4, 11, 14, 24, 37, 51, 55, 56, 57, 70, 81, 99, 102, 107, 115, 120,
        128, 133, 141, 146, 158, 166, 170, 174, 177, 184, 191, 194, 200,
        201, 208, 221, 222, 233, 236,
    },
    "tear_missing": {51, 115, 141, 170, 201, 221},
}

FIELDNAMES = (
    "file",
    "main_label",
    "sub_label",
    "label_path",
    "primary_category",
    "tags",
    "label_source",
    "scale",
    "distribution",
    "density",
    "blend_mode",
    "confidence",
    "enabled",
    "review_status",
    "coverage",
    "bright_coverage",
    "edge_bias",
    "anisotropy",
    "edge_density",
    "coarse_ratio",
    "component_density",
    "largest_component",
)


@dataclass(frozen=True)
class Features:
    coverage: float
    bright_coverage: float
    edge_bias: float
    anisotropy: float
    edge_density: float
    coarse_ratio: float
    component_density: float
    largest_component: float


def _read_gray(path: Path, max_side: int = 640) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def extract_features(path: Path) -> Features:
    gray = _read_gray(path)
    height, width = gray.shape
    positive = gray >= 14
    bright = gray >= 64
    coverage = float(positive.mean())
    bright_coverage = float(bright.mean())

    border_y = max(1, height // 5)
    border_x = max(1, width // 5)
    border = np.zeros_like(positive)
    border[:border_y] = True
    border[-border_y:] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    center = ~border
    border_cov = float(positive[border].mean()) if border.any() else 0.0
    center_cov = float(positive[center].mean()) if center.any() else 0.0
    edge_bias = border_cov / max(center_cov, 1e-4)

    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=8.0)
    total_var = float(np.var(gray.astype(np.float32)))
    coarse_ratio = float(np.var(smooth.astype(np.float32)) / max(total_var, 1e-6))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy)
    weight_sum = float(magnitude.sum())
    if weight_sum:
        cos_sum = float((magnitude * np.cos(2.0 * angle)).sum())
        sin_sum = float((magnitude * np.sin(2.0 * angle)).sum())
        anisotropy = float(np.hypot(cos_sum, sin_sum) / weight_sum)
    else:
        anisotropy = 0.0

    edges = cv2.Canny(gray, 20, 60)
    edge_density = float((edges > 0).mean())

    component_mask = np.where(gray >= 24, 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    image_area = float(width * height)
    component_density = max(0, count - 1) / max(image_area / 1_000_000.0, 1e-6)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    positive_area = float(component_areas.sum())
    largest_component = (
        float(component_areas.max() / positive_area) if positive_area else 0.0
    )

    return Features(
        coverage=coverage,
        bright_coverage=bright_coverage,
        edge_bias=edge_bias,
        anisotropy=anisotropy,
        edge_density=edge_density,
        coarse_ratio=coarse_ratio,
        component_density=component_density,
        largest_component=largest_component,
    )


def _manual_primary(texture_id: int) -> str:
    matches = [
        sub_label
        for sub_label, texture_ids in MANUAL_PRIMARY.items()
        if texture_id in texture_ids
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Texture {texture_id:03d} must have exactly one visual primary "
            f"label; found {matches}"
        )
    return matches[0]


def _manual_tags(texture_id: int, primary: str) -> set[str]:
    tags = {
        category
        for category, texture_ids in AUXILIARY_TAGS.items()
        if texture_id in texture_ids
    }
    tags.add(primary)
    return tags


def validate_manual_labels(texture_paths: list[Path]) -> None:
    source_ids = {int(path.stem) for path in texture_paths}
    mapped_ids: set[int] = set()
    duplicates: dict[int, list[str]] = {}
    for sub_label, texture_ids in MANUAL_PRIMARY.items():
        if sub_label not in CATEGORIES:
            raise ValueError(f"Unknown manual sub-label: {sub_label}")
        for texture_id in texture_ids:
            if texture_id in mapped_ids:
                duplicates[texture_id] = [
                    label
                    for label, ids in MANUAL_PRIMARY.items()
                    if texture_id in ids
                ]
            mapped_ids.add(texture_id)
    if duplicates:
        raise ValueError(f"Duplicate visual primary labels: {duplicates}")
    missing = sorted(source_ids - mapped_ids)
    extra = sorted(mapped_ids - source_ids)
    if missing or extra:
        raise ValueError(
            f"Visual primary label coverage mismatch; missing={missing}, "
            f"extra={extra}"
        )


def classify(path: Path, features: Features) -> dict[str, str]:
    texture_id = int(path.stem)
    primary = _manual_primary(texture_id)
    main_label = SUB_LABEL_TO_MAIN[primary]
    tags = _manual_tags(texture_id, primary)

    if primary in {"scratch", "crack", "fold"} or features.anisotropy >= 0.12:
        scale = "linear"
    elif features.component_density >= 2200 or features.coverage < 0.025:
        scale = "fine"
    elif features.largest_component >= 0.4 or features.coarse_ratio >= 0.35:
        scale = "coarse"
    else:
        scale = "multi_scale"

    if primary in {"edge_wear", "tear_missing"} or features.edge_bias >= 2.0:
        distribution = "edge"
    elif primary in {"scratch", "crack", "fold"} or features.anisotropy >= 0.12:
        distribution = "directional"
    elif features.coverage < 0.05:
        distribution = "sparse"
    else:
        distribution = "diffuse"

    if features.coverage < 0.03:
        density = "sparse"
    elif features.coverage < 0.15:
        density = "medium"
    else:
        density = "dense"

    blend_mode = (
        "screen"
        if primary in {"scratch", "crack", "dust", "fold"}
        else "multiply"
    )

    return {
        "file": path.name,
        "main_label": main_label,
        "sub_label": primary,
        "label_path": f"{main_label}--{primary}",
        "primary_category": primary,
        "tags": ";".join(sorted(tags)),
        "label_source": "manual_visual",
        "scale": scale,
        "distribution": distribution,
        "density": density,
        "blend_mode": blend_mode,
        "confidence": "0.97",
        "enabled": "0" if primary == "reject" else "1",
        "review_status": "manual_visual",
        "coverage": f"{features.coverage:.6f}",
        "bright_coverage": f"{features.bright_coverage:.6f}",
        "edge_bias": f"{features.edge_bias:.4f}",
        "anisotropy": f"{features.anisotropy:.4f}",
        "edge_density": f"{features.edge_density:.6f}",
        "coarse_ratio": f"{features.coarse_ratio:.4f}",
        "component_density": f"{features.component_density:.2f}",
        "largest_component": f"{features.largest_component:.4f}",
    }


def write_catalog(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _render_category_sheet(
    category: str,
    rows: list[dict[str, str]],
    texture_dir: Path,
    output: Path,
) -> None:
    columns = 5
    tile_width, tile_height = 230, 190
    image_width, image_height = 210, 145
    rows_count = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, max(1, rows_count) * tile_height + 34),
        (238, 238, 238),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 9), f"{category} ({len(rows)})", fill=(0, 0, 0))

    for index, row in enumerate(rows):
        column = index % columns
        grid_row = index // columns
        x = column * tile_width + 10
        y = grid_row * tile_height + 34
        with Image.open(texture_dir / row["file"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            image = ImageOps.autocontrast(image).convert("RGB")
            thumb = ImageOps.contain(
                image, (image_width, image_height), Image.Resampling.LANCZOS
            )
        canvas = Image.new("RGB", (image_width, image_height), (22, 22, 22))
        canvas.paste(
            thumb,
            ((image_width - thumb.width) // 2, (image_height - thumb.height) // 2),
        )
        sheet.paste(canvas, (x, y))
        label = (
            f"{row['file']}  {row['label_path']}  "
            f"cov={float(row['coverage']) * 100:.1f}%"
        )
        draw.text((x, y + image_height + 5), label, fill=(0, 0, 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=92)


def render_review_sheets(
    rows: list[dict[str, str]], texture_dir: Path, review_dir: Path
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("category_*.jpg"):
        old.unlink()
    for category in CATEGORIES:
        selected = [row for row in rows if row["sub_label"] == category]
        if selected:
            _render_category_sheet(
                category,
                selected,
                texture_dir,
                review_dir / f"category_{category}.jpg",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify unlabeled texture materials and render review sheets"
    )
    parser.add_argument("--texture-dir", type=Path, default=DEFAULT_TEXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    texture_dir = Path(args.texture_dir)
    texture_paths = list_texture_paths(texture_dir)
    validate_manual_labels(texture_paths)
    rows: list[dict[str, str]] = []
    for index, path in enumerate(texture_paths, start=1):
        features = extract_features(path)
        rows.append(classify(path, features))
        if index % 25 == 0:
            print(f"analyzed {index} textures")

    write_catalog(rows, Path(args.output))
    render_review_sheets(rows, texture_dir, Path(args.review_dir))
    counts = Counter(row["sub_label"] for row in rows)
    print(f"catalog written: {args.output}")
    print(f"review sheets: {args.review_dir}")
    print("category counts:")
    for category in CATEGORIES:
        if counts[category]:
            print(f"  {category}: {counts[category]}")


if __name__ == "__main__":
    main()

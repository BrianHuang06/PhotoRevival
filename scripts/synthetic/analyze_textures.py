"""Build and review the material-only texture catalog.

The source texture pack uses numeric filenames and does not provide semantic
labels. This script combines visually reviewed tags with measured image
features, writes a deterministic CSV catalog, and renders per-category contact
sheets for later human correction.

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

try:  # pragma: no cover - environment dependent
    from photo_revival.age_photo import DEFAULT_TEXTURE_DIR, list_texture_paths
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from photo_revival.age_photo import DEFAULT_TEXTURE_DIR, list_texture_paths


DEFAULT_CATALOG_PATH = Path(__file__).with_name("texture_catalog.csv")
DEFAULT_REVIEW_DIR = Path("artifacts/texture_analysis/catalog")

CATEGORIES = (
    "scratch",
    "crack",
    "dust",
    "stain",
    "mold",
    "surface_wear",
    "edge_wear",
    "fold",
    "tear_missing",
    "mixed",
    "reject",
)

# These sets come from visual inspection of all 250 numbered texture sheets.
# A texture may occur in several sets; PRIMARY_PRIORITY resolves its main role
# while the complete overlap is retained in the semicolon-separated tags.
MANUAL_TAGS: dict[str, set[int]] = {
    "fold": {
        28, 32, 35, 44, 46, 58, 59, 65, 66, 73, 78, 89, 105, 117, 123,
        126, 131, 143, 153, 173, 175, 176, 181, 186, 200, 226, 233, 242, 248,
    },
    "crack": {48, 109, 116, 135, 185, 215, 240, 246},
    "tear_missing": {51, 115, 141, 170, 201, 221},
    "scratch": {
        9, 31, 35, 48, 93, 96, 117, 124, 135, 159, 181, 185, 209, 215,
        246, 248, 250,
    },
    "dust": {
        15, 18, 24, 69, 71, 72, 79, 83, 86, 102, 112, 119, 129, 139, 144,
        162, 163, 168, 169, 188, 202, 204, 211, 213, 219, 227, 235, 245,
    },
    "stain": {
        4, 7, 21, 25, 32, 37, 51, 56, 68, 91, 101, 112, 115, 132, 137,
        141, 154, 165, 167, 170, 182, 186, 192, 193, 194, 201, 208, 221,
        229, 234, 237, 239, 250,
    },
    "mold": {
        2, 4, 8, 10, 16, 23, 39, 41, 45, 47, 51, 55, 57, 62, 64, 67,
        75, 84, 85, 87, 97, 98, 101, 103, 107, 121, 128, 130, 133, 134,
        137, 138, 142, 145, 151, 156, 157, 158, 164, 167, 180, 182, 184,
        191, 193, 194, 197, 198, 207, 208, 210, 216, 217, 223, 225, 228,
        229, 231, 234, 236, 238, 239, 241, 243,
    },
    "edge_wear": {
        11, 14, 24, 37, 55, 56, 81, 99, 102, 107, 115, 120, 128, 133,
        135, 146, 158, 166, 170, 174, 176, 194, 200, 208, 221, 222, 233,
        236, 248,
    },
    "reject": {150, 203, 247},
}

PRIMARY_PRIORITY = (
    "reject",
    "tear_missing",
    "crack",
    "fold",
    "edge_wear",
    "scratch",
    "dust",
    "stain",
    "mold",
)

FIELDNAMES = (
    "file",
    "primary_category",
    "tags",
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


def _manual_tags(texture_id: int) -> set[str]:
    return {
        category for category, texture_ids in MANUAL_TAGS.items()
        if texture_id in texture_ids
    }


def _feature_tags(features: Features) -> set[str]:
    tags: set[str] = set()
    if features.coverage <= 0.0015 or features.bright_coverage <= 0.00005:
        tags.add("reject")
    if features.edge_bias >= 2.2:
        tags.add("edge_wear")
    if features.anisotropy >= 0.16:
        tags.add("scratch")
    if features.component_density >= 3200 and features.largest_component < 0.10:
        tags.add("dust")
    if features.coverage >= 0.20 and features.coarse_ratio >= 0.20:
        tags.add("mold")
    if features.largest_component >= 0.45 and features.coverage >= 0.06:
        tags.add("stain")
    return tags


def _fallback_primary(features: Features) -> str:
    if features.coverage <= 0.0015 or features.bright_coverage <= 0.00005:
        return "reject"
    if features.edge_bias >= 2.2:
        return "edge_wear"
    if features.anisotropy >= 0.16 and features.coverage < 0.12:
        return "scratch"
    if features.largest_component >= 0.45 and features.coverage >= 0.06:
        return "stain"
    if features.coverage >= 0.20 or (
        features.coverage >= 0.11 and features.edge_density >= 0.11
    ):
        return "mold"
    if features.coverage >= 0.020:
        return "surface_wear"
    return "mixed"


def classify(path: Path, features: Features) -> dict[str, str]:
    texture_id = int(path.stem)
    manual = _manual_tags(texture_id)
    measured = _feature_tags(features)
    tags = manual | measured

    primary = next(
        (category for category in PRIMARY_PRIORITY if category in manual),
        _fallback_primary(features),
    )
    tags.add(primary)

    if primary in {"scratch", "crack", "fold"} or features.anisotropy >= 0.12:
        scale = "linear"
    elif features.component_density >= 2200 or features.coverage < 0.025:
        scale = "fine"
    elif features.largest_component >= 0.4 or features.coarse_ratio >= 0.35:
        scale = "coarse"
    else:
        scale = "mixed"

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
        "screen" if primary in {"scratch", "dust", "fold"} else "multiply"
    )
    review_status = "manual" if manual else "auto"
    confidence = 0.97 if manual else 0.72

    return {
        "file": path.name,
        "primary_category": primary,
        "tags": ";".join(sorted(tags)),
        "scale": scale,
        "distribution": distribution,
        "density": density,
        "blend_mode": blend_mode,
        "confidence": f"{confidence:.2f}",
        "enabled": "0" if primary == "reject" else "1",
        "review_status": review_status,
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
            f"{row['file']}  {row['review_status']}  "
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
        selected = [row for row in rows if row["primary_category"] == category]
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
    rows: list[dict[str, str]] = []
    for index, path in enumerate(list_texture_paths(texture_dir), start=1):
        features = extract_features(path)
        rows.append(classify(path, features))
        if index % 25 == 0:
            print(f"analyzed {index} textures")

    write_catalog(rows, Path(args.output))
    render_review_sheets(rows, texture_dir, Path(args.review_dir))
    counts = Counter(row["primary_category"] for row in rows)
    print(f"catalog written: {args.output}")
    print(f"review sheets: {args.review_dir}")
    print("category counts:")
    for category in CATEGORIES:
        if counts[category]:
            print(f"  {category}: {counts[category]}")


if __name__ == "__main__":
    main()

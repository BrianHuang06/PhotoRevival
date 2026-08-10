"""Strict material-only synthetic degradation engine.

Every visible degraded pixel comes from a cataloged texture material. The
engine may crop, resize, rotate, flip, threshold, change opacity, and composite
those materials. It does not draw damage shapes or synthesize color shifts,
noise, blur, grain, exposure changes, missing regions, or compression damage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.synthetic.texture_assets import DEFAULT_TEXTURE_DIR, list_texture_paths

try:
    from PIL import __version__ as PILLOW_VERSION
except ImportError:  # pragma: no cover - very old Pillow fallback
    PILLOW_VERSION = "unknown"


DEFAULT_CATALOG_PATH = Path(__file__).with_name("texture_catalog.csv")
ALLOWED_MATERIAL_OPERATIONS = {"texture_composite"}
ALLOWED_BLEND_MODES = {"screen", "multiply"}
ALLOWED_TRANSFORMS = (
    "crop",
    "resize",
    "rotate_90",
    "flip",
    "threshold",
    "opacity",
    "composite",
)

DETAIL_CATEGORIES = ("scratch", "crack", "dust")
AREA_CATEGORIES = ("stain", "mold", "surface_wear", "mixed")
STRUCTURE_CATEGORIES = ("edge_wear", "fold", "tear_missing")
RUNTIME_CATEGORIES = DETAIL_CATEGORIES + AREA_CATEGORIES + STRUCTURE_CATEGORIES

# Each severity is a category recipe. Coverage is requested per material layer;
# the final union is constrained by coverage_target.
SEVERITY_PROFILE: dict[str, dict] = {
    "light": {
        "slots": (
            {
                "group": "detail_or_surface",
                "categories": ("scratch", "dust", "surface_wear"),
                "coverage": (2.0, 5.0),
                "opacity": (0.24, 0.42),
            },
        ),
        "coverage_target": (1.0, 8.0),
    },
    "medium": {
        "slots": (
            {
                "group": "detail",
                "categories": DETAIL_CATEGORIES,
                "coverage": (3.0, 7.0),
                "opacity": (0.34, 0.54),
            },
            {
                "group": "area",
                "categories": AREA_CATEGORIES,
                "coverage": (5.0, 10.0),
                "opacity": (0.32, 0.52),
            },
        ),
        "coverage_target": (5.0, 20.0),
    },
    "heavy": {
        "slots": (
            {
                "group": "detail",
                "categories": DETAIL_CATEGORIES,
                "coverage": (5.0, 9.0),
                "opacity": (0.48, 0.68),
            },
            {
                "group": "area",
                "categories": AREA_CATEGORIES,
                "coverage": (8.0, 13.0),
                "opacity": (0.46, 0.66),
            },
            {
                "group": "structure",
                "categories": STRUCTURE_CATEGORIES,
                "coverage": (7.0, 12.0),
                "opacity": (0.52, 0.74),
            },
        ),
        "coverage_target": (15.0, 32.0),
    },
}

SEVERITY_ORDER = {"light": 1, "medium": 2, "heavy": 3}

CATALOG_REQUIRED_FIELDS = {
    "file",
    "primary_category",
    "tags",
    "scale",
    "distribution",
    "density",
    "blend_mode",
    "confidence",
    "enabled",
}


def derive_variant_seed(global_seed: int, source_index: int, severity: str) -> int:
    """Derive a stable per-variant seed from a global seed."""
    key = (
        global_seed * 0x9E3779B97F4A7C15
        + source_index * 0xBF58476D1CE4E5B9
        + SEVERITY_ORDER[severity] * 0x94D049BB133111EB
    ) & 0xFFFFFFFFFFFFFFFF
    key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    key = (key ^ (key >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return key ^ (key >> 31)


def _randfloat(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


@dataclass(frozen=True)
class TextureRecord:
    path: Path
    file: str
    primary_category: str
    tags: tuple[str, ...]
    scale: str
    distribution: str
    density: str
    blend_mode: str
    confidence: float
    enabled: bool
    source_coverage: float | None


@dataclass
class DegradationConfig:
    min_coverage: float = 1.0
    max_coverage: float = 35.0
    texture_dir: Path | None = None
    catalog_path: Path | None = None
    max_adapt_rounds: int = 6
    warnings: list[str] = field(default_factory=list)


@dataclass
class DegradedVariant:
    degraded: Image.Image
    mask: Image.Image
    metadata: dict
    coverage: float


def coverage_of(mask: Image.Image | np.ndarray) -> float:
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    return 100.0 * float((values > 0).mean())


def _parse_enabled(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    return float(value)


def catalog_sha256(catalog_path: Path | None = None) -> str:
    path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_texture_catalog(
    texture_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> list[TextureRecord]:
    """Load and strictly validate the texture catalog against source assets."""
    texture_root = Path(texture_dir) if texture_dir else DEFAULT_TEXTURE_DIR
    catalog = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    if not catalog.is_file():
        raise FileNotFoundError(
            f"Texture catalog not found: {catalog}. "
            "Run python -m scripts.synthetic.analyze_textures first."
        )

    source_paths = list_texture_paths(texture_root)
    source_by_name = {path.name: path for path in source_paths}
    if len(source_by_name) != len(source_paths):
        raise ValueError(f"Duplicate texture filenames under {texture_root}")

    with catalog.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = CATALOG_REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"Texture catalog missing fields: {sorted(missing_fields)}"
            )
        raw_rows = list(reader)

    seen: set[str] = set()
    records: list[TextureRecord] = []
    for row_number, row in enumerate(raw_rows, start=2):
        filename = row["file"].strip()
        if not filename:
            raise ValueError(f"Texture catalog row {row_number} has no filename")
        if filename in seen:
            raise ValueError(f"Duplicate texture catalog entry: {filename}")
        seen.add(filename)
        if filename not in source_by_name:
            raise FileNotFoundError(
                f"Cataloged texture does not exist under {texture_root}: {filename}"
            )

        primary = row["primary_category"].strip()
        if primary not in RUNTIME_CATEGORIES + ("reject",):
            raise ValueError(
                f"{filename}: unsupported primary_category={primary!r}"
            )
        tags = tuple(
            sorted({tag.strip() for tag in row["tags"].split(";") if tag.strip()})
        )
        if primary not in tags:
            raise ValueError(f"{filename}: primary category missing from tags")
        invalid_tags = set(tags) - set(RUNTIME_CATEGORIES) - {"reject"}
        if invalid_tags:
            raise ValueError(f"{filename}: invalid tags {sorted(invalid_tags)}")

        blend_mode = row["blend_mode"].strip()
        if blend_mode not in ALLOWED_BLEND_MODES:
            raise ValueError(f"{filename}: invalid blend_mode={blend_mode!r}")

        records.append(
            TextureRecord(
                path=source_by_name[filename],
                file=filename,
                primary_category=primary,
                tags=tags,
                scale=row["scale"].strip(),
                distribution=row["distribution"].strip(),
                density=row["density"].strip(),
                blend_mode=blend_mode,
                confidence=float(row["confidence"]),
                enabled=_parse_enabled(row["enabled"]),
                source_coverage=_optional_float(row.get("coverage")),
            )
        )

    missing_catalog_entries = sorted(set(source_by_name) - seen)
    if missing_catalog_entries:
        raise ValueError(
            "Texture catalog does not classify every source asset; missing "
            f"{missing_catalog_entries[:10]}"
        )

    enabled = [record for record in records if record.enabled]
    if not enabled:
        raise ValueError("Texture catalog has no enabled materials")
    if any(record.primary_category == "reject" for record in enabled):
        raise ValueError("reject materials must not be enabled")
    for category in RUNTIME_CATEGORIES:
        if not any(record.primary_category == category for record in enabled):
            raise ValueError(f"Texture catalog has no enabled {category} material")
    return records


def texture_catalog_summary(
    texture_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> dict:
    records = load_texture_catalog(texture_dir, catalog_path)
    counts = {
        category: sum(
            1
            for record in records
            if record.enabled and record.primary_category == category
        )
        for category in RUNTIME_CATEGORIES
    }
    return {
        "total": len(records),
        "enabled": sum(record.enabled for record in records),
        "disabled": sum(not record.enabled for record in records),
        "per_category": counts,
        "sha256": catalog_sha256(catalog_path),
    }


def _records_by_category(
    records: list[TextureRecord],
) -> dict[str, list[TextureRecord]]:
    pools = {category: [] for category in RUNTIME_CATEGORIES}
    for record in records:
        if record.enabled and record.primary_category in pools:
            pools[record.primary_category].append(record)
    return pools


def _choose_record(
    rng: np.random.Generator,
    pools: dict[str, list[TextureRecord]],
    categories: tuple[str, ...],
    used_files: set[str],
) -> tuple[str, TextureRecord]:
    available_categories = [
        category
        for category in categories
        if any(record.file not in used_files for record in pools[category])
    ]
    if not available_categories:
        raise RuntimeError(
            f"No unused enabled material remains for categories {categories}"
        )
    category = available_categories[int(rng.integers(0, len(available_categories)))]
    candidates = [
        record for record in pools[category] if record.file not in used_files
    ]
    record = candidates[int(rng.integers(0, len(candidates)))]
    used_files.add(record.file)
    return category, record


def _load_transformed_texture(
    record: TextureRecord,
    rng: np.random.Generator,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    with Image.open(record.path) as opened:
        opened.draft("L", (max(width * 2, 512), max(height * 2, 512)))
        texture = ImageOps.exif_transpose(opened).convert("L")

    rotation = int(rng.integers(0, 4)) * 90
    if rotation:
        texture = texture.rotate(rotation, expand=True)
    flip_horizontal = bool(rng.integers(0, 2))
    flip_vertical = bool(rng.integers(0, 2))
    if flip_horizontal:
        texture = ImageOps.mirror(texture)
    if flip_vertical:
        texture = ImageOps.flip(texture)

    if record.primary_category in STRUCTURE_CATEGORIES:
        centering_choices = (0.0, 0.5, 1.0)
        center_x = centering_choices[int(rng.integers(0, len(centering_choices)))]
        center_y = centering_choices[int(rng.integers(0, len(centering_choices)))]
    else:
        center_x = _randfloat(rng, 0.15, 0.85)
        center_y = _randfloat(rng, 0.15, 0.85)

    fitted = ImageOps.fit(
        texture,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(center_x, center_y),
    )
    return np.asarray(fitted, dtype=np.float32), {
        "rotation": rotation,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "fit_centering": [round(center_x, 4), round(center_y, 4)],
        "resample": "LANCZOS",
    }


def _material_alpha(
    gray: np.ndarray,
    target_coverage: float,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select the strongest source-material pixels and build their alpha."""
    low = float(np.percentile(gray, 5.0))
    high = max(
        float(np.percentile(gray, 99.5)),
        float(gray.max()) * 0.35,
        low + 1.0,
    )
    signal = np.clip((gray - low) / (high - low), 0.0, 1.0)
    available = signal > 0.02
    available_indices = np.flatnonzero(available)
    if not len(available_indices):
        raise RuntimeError("Texture transform contains no usable material signal")

    desired = max(1, int(round(signal.size * target_coverage / 100.0)))
    desired = min(desired, len(available_indices))
    available_values = signal.ravel()[available_indices]
    if desired == len(available_indices):
        selected_indices = available_indices
    else:
        strongest = np.argpartition(available_values, -desired)[-desired:]
        selected_indices = available_indices[strongest]

    selected = np.zeros(signal.size, dtype=bool)
    selected[selected_indices] = True
    selected = selected.reshape(signal.shape)
    selected_values = signal[selected]
    threshold = float(selected_values.min())
    strength = np.zeros_like(signal, dtype=np.float32)
    strength[selected] = 0.35 + 0.65 * np.clip(
        (selected_values - threshold) / max(1.0 - threshold, 1e-6),
        0.0,
        1.0,
    )
    alpha = np.clip(strength * opacity, 0.0, 1.0)
    actual_coverage = 100.0 * float(selected.mean())
    return alpha, selected, {
        "black_level": round(low, 3),
        "white_level": round(high, 3),
        "threshold": round(threshold, 6),
        "requested_coverage": round(target_coverage, 4),
        "actual_coverage": round(actual_coverage, 4),
        "opacity": round(opacity, 4),
    }


def _build_material_layer(
    record: TextureRecord,
    category_group: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    target_coverage: float,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    gray, transform = _load_transformed_texture(record, rng, width, height)
    alpha, selected, threshold = _material_alpha(
        gray, target_coverage, opacity
    )
    params = {
        "texture": record.file,
        "category": record.primary_category,
        "category_group": category_group,
        "tags": list(record.tags),
        "scale": record.scale,
        "distribution": record.distribution,
        "density": record.density,
        "blend_mode": record.blend_mode,
        "catalog_confidence": round(record.confidence, 3),
        "catalog_source_coverage": record.source_coverage,
        "transform": transform,
        "thresholding": threshold,
    }
    return alpha, selected, params


def _apply_material_layer(
    base: np.ndarray,
    alpha: np.ndarray,
    blend_mode: str,
) -> np.ndarray:
    amount = alpha[..., None]
    if blend_mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - amount)
    if blend_mode == "multiply":
        return base * (1.0 - amount)
    raise ValueError(f"Unsupported material blend mode: {blend_mode}")


def _render_material_plan(
    source: Image.Image,
    plans: list[dict],
    rng: np.random.Generator,
    coverage_scale: float,
) -> tuple[Image.Image, Image.Image, float, list[dict]]:
    width, height = source.size
    source_values = np.asarray(source.convert("RGB"), dtype=np.uint8)
    result = source_values.astype(np.float32) / 255.0
    union = np.zeros((height, width), dtype=bool)
    steps: list[dict] = []
    for plan in plans:
        requested = min(34.0, max(0.1, plan["coverage"] * coverage_scale))
        alpha, selected, params = _build_material_layer(
            plan["record"],
            plan["group"],
            rng,
            width,
            height,
            requested,
            plan["opacity"],
        )
        result = _apply_material_layer(
            result, alpha, plan["record"].blend_mode
        )
        union |= selected
        steps.append({"op": "texture_composite", "params": params})

    degraded_values = np.clip(
        np.rint(result * 255.0),
        0,
        255,
    ).astype(np.uint8)
    degraded_values[~union] = source_values[~union]
    degraded = Image.fromarray(degraded_values, "RGB")
    mask = Image.fromarray(np.where(union, 255, 0).astype(np.uint8), "L")
    coverage = 100.0 * float(union.mean())
    return degraded, mask, coverage, steps


def validate_material_only_metadata(metadata: dict) -> list[str]:
    errors: list[str] = []
    if metadata.get("generation_mode") != "material_only":
        errors.append("generation_mode must be material_only")
    if metadata.get("restoration_type") != "local_repair":
        errors.append("restoration_type must be local_repair")
    severity = metadata.get("severity")
    profile = SEVERITY_PROFILE.get(severity)
    if profile is None:
        errors.append(f"unknown severity {severity!r}")
    if metadata.get("allowed_transforms") != list(ALLOWED_TRANSFORMS):
        errors.append("allowed_transforms does not match strict material policy")
    steps = metadata.get("steps") or []
    if not steps:
        errors.append("metadata has no material steps")
        return errors
    recipe = metadata.get("category_recipe") or []
    if profile is not None:
        expected_count = len(profile["slots"])
        if len(steps) != expected_count:
            errors.append(
                f"severity {severity} requires {expected_count} material steps"
            )
        if len(recipe) != expected_count:
            errors.append(
                f"severity {severity} requires {expected_count} recipe entries"
            )
    for index, step in enumerate(steps):
        if step.get("op") not in ALLOWED_MATERIAL_OPERATIONS:
            errors.append(
                f"step {index} uses forbidden operation {step.get('op')!r}"
            )
            continue
        params = step.get("params") or {}
        for key in (
            "texture",
            "category",
            "category_group",
            "tags",
            "blend_mode",
            "transform",
            "thresholding",
        ):
            if key not in params:
                errors.append(f"step {index} missing {key}")
        if params.get("blend_mode") not in ALLOWED_BLEND_MODES:
            errors.append(f"step {index} has invalid blend_mode")
        transform = params.get("transform") or {}
        if transform.get("rotation") not in {0, 90, 180, 270}:
            errors.append(f"step {index} has invalid rotation")
        if not isinstance(transform.get("flip_horizontal"), bool):
            errors.append(f"step {index} has invalid horizontal flip")
        if not isinstance(transform.get("flip_vertical"), bool):
            errors.append(f"step {index} has invalid vertical flip")
        if profile is not None and index < len(profile["slots"]):
            slot = profile["slots"][index]
            if params.get("category_group") != slot["group"]:
                errors.append(f"step {index} has incorrect category group")
            if params.get("category") not in slot["categories"]:
                errors.append(f"step {index} category is outside severity recipe")
        if index < len(recipe):
            recipe_entry = recipe[index]
            if recipe_entry.get("group") != params.get("category_group"):
                errors.append(f"step {index} recipe group mismatch")
            if recipe_entry.get("category") != params.get("category"):
                errors.append(f"step {index} recipe category mismatch")
            if recipe_entry.get("texture") != params.get("texture"):
                errors.append(f"step {index} recipe texture mismatch")
    return errors


def degrade_target(
    source_image: Image.Image,
    severity: str,
    seed: int,
    cfg: DegradationConfig | None = None,
) -> DegradedVariant:
    """Generate one strict material-only degraded variant."""
    if severity not in SEVERITY_PROFILE:
        raise ValueError(f"Unknown severity: {severity}")
    cfg = cfg or DegradationConfig()
    if cfg.min_coverage < 0 or cfg.max_coverage <= 0:
        raise ValueError("Coverage bounds must be positive")
    if cfg.min_coverage > cfg.max_coverage:
        raise ValueError("min_coverage cannot exceed max_coverage")

    texture_dir = Path(cfg.texture_dir) if cfg.texture_dir else DEFAULT_TEXTURE_DIR
    catalog_path = Path(cfg.catalog_path) if cfg.catalog_path else DEFAULT_CATALOG_PATH
    records = load_texture_catalog(texture_dir, catalog_path)
    pools = _records_by_category(records)
    profile = SEVERITY_PROFILE[severity]
    rng = np.random.default_rng(seed)

    used_files: set[str] = set()
    plans: list[dict] = []
    for slot in profile["slots"]:
        _, record = _choose_record(
            rng, pools, tuple(slot["categories"]), used_files
        )
        plans.append(
            {
                "group": slot["group"],
                "record": record,
                "coverage": _randfloat(rng, *slot["coverage"]),
                "opacity": _randfloat(rng, *slot["opacity"]),
            }
        )

    target_lo = max(float(profile["coverage_target"][0]), cfg.min_coverage)
    target_hi = min(float(profile["coverage_target"][1]), cfg.max_coverage)
    if target_lo > target_hi:
        raise ValueError(
            f"Coverage bounds [{cfg.min_coverage}, {cfg.max_coverage}] "
            f"do not overlap severity {severity} target "
            f"{profile['coverage_target']}"
        )

    coverage_scale = 1.0
    final: tuple[Image.Image, Image.Image, float, list[dict]] | None = None
    for _ in range(max(1, cfg.max_adapt_rounds)):
        rendered = _render_material_plan(
            source_image, plans, rng, coverage_scale
        )
        final = rendered
        coverage = rendered[2]
        if target_lo - 0.25 <= coverage <= target_hi + 0.25:
            break
        if coverage < target_lo:
            coverage_scale *= min(2.0, target_lo / max(coverage, 0.1) * 1.08)
        else:
            coverage_scale *= max(0.25, target_hi / coverage * 0.92)

    assert final is not None
    degraded, mask, coverage, steps = final
    if not (target_lo - 0.5 <= coverage <= target_hi + 0.5):
        raise RuntimeError(
            f"Could not reach {severity} material coverage target "
            f"[{target_lo:.2f}, {target_hi:.2f}]; got {coverage:.2f}"
        )

    metadata = {
        "seed": seed,
        "severity": severity,
        "generation_mode": "material_only",
        "restoration_type": "local_repair",
        "mask_coverage": round(coverage, 4),
        "image_size": list(source_image.size),
        "material_catalog": {
            "path": str(catalog_path),
            "sha256": catalog_sha256(catalog_path),
            "texture_dir": str(texture_dir),
        },
        "category_recipe": [
            {
                "group": plan["group"],
                "category": plan["record"].primary_category,
                "texture": plan["record"].file,
            }
            for plan in plans
        ],
        "allowed_transforms": list(ALLOWED_TRANSFORMS),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "steps": steps,
    }
    material_errors = validate_material_only_metadata(metadata)
    if material_errors:
        raise RuntimeError("; ".join(material_errors))
    return DegradedVariant(
        degraded=degraded,
        mask=mask,
        metadata=metadata,
        coverage=coverage,
    )


def save_variant_files(
    output_root: Path,
    source_id: str,
    severity: str,
    variant: DegradedVariant,
) -> None:
    output_root = Path(output_root)
    degraded_path = output_root / "degraded" / f"{source_id}_{severity}.png"
    mask_path = output_root / "masks" / f"{source_id}_{severity}.png"
    degraded_path.parent.mkdir(parents=True, exist_ok=True)
    variant.degraded.save(degraded_path, "PNG")
    variant.mask.save(mask_path, "PNG")


def write_source_metadata(
    output_root: Path,
    source_id: str,
    target_sha256: str,
    variants: dict[str, DegradedVariant],
) -> None:
    """Write one metadata file per source containing all severity variants."""
    output_root = Path(output_root)
    entries: dict[str, dict] = {}
    for severity, variant in variants.items():
        meta = dict(variant.metadata)
        meta["files"] = {
            "degraded": f"degraded/{source_id}_{severity}.png",
            "mask": f"masks/{source_id}_{severity}.png",
        }
        entries[severity] = meta
    doc = {
        "source_id": source_id,
        "target": f"target/{source_id}.png",
        "target_sha256": target_sha256,
        "variants": entries,
    }
    metadata_path = output_root / "metadata" / f"{source_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for legacy in metadata_path.parent.glob(f"{source_id}_*.json"):
        if legacy.name != metadata_path.name:
            legacy.unlink(missing_ok=True)


def build_source_collage(
    target: Path,
    degraded: list[Path],
    masks: list[Path],
    output: Path,
    labels: list[str],
) -> None:
    """Build a two-row target/degraded/mask review collage."""
    cell_width, cell_height, label_height, pad = 280, 220, 22, 8
    columns = 1 + len(degraded)
    rows = 2
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * pad,
            rows * (cell_height + label_height) + (rows + 1) * pad,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    def _paste(image: Image.Image, column: int, row: int, label: str) -> None:
        thumb = ImageOps.contain(
            image.convert("RGB"),
            (cell_width, cell_height),
            Image.Resampling.LANCZOS,
        )
        x = (
            pad
            + column * (cell_width + pad)
            + (cell_width - thumb.width) // 2
        )
        y = (
            pad
            + row * (cell_height + label_height)
            + (cell_height - thumb.height) // 2
        )
        sheet.paste(thumb, (x, y))
        draw.text(
            (
                pad + column * (cell_width + pad) + 4,
                pad
                + row * (cell_height + label_height)
                + cell_height
                + 2,
            ),
            label,
            fill="black",
        )

    with Image.open(target) as opened:
        _paste(opened, 0, 0, "target")
    for column, (degraded_path, mask_path, label) in enumerate(
        zip(degraded, masks, labels),
        start=1,
    ):
        with Image.open(degraded_path) as opened:
            degraded_img = opened.convert("RGB")
        _paste(degraded_img, column, 0, label)
        with Image.open(mask_path) as opened:
            mask = opened.convert("L").resize(
                degraded_img.size, Image.Resampling.NEAREST
            )
        overlay_red = Image.new("RGB", degraded_img.size, (255, 0, 0))
        mask_overlay = Image.composite(overlay_red, degraded_img, mask)
        _paste(mask_overlay, column, 1, f"{label} mask")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")

"""Generate reproducible light/medium/heavy synthetic degradation pairs.

This is the C-role deliverable of the team dataset workflow: it turns clean
target images into paired training data. For every clean image it writes:

    target/<source_id>.png
    degraded/<source_id>_<severity>.png
    masks/<source_id>_<severity>.png        # 0 = clean, 255 = damaged
    metadata/<source_id>.json               # all severities + material records

plus a manifest.csv, summary.json, MASK_HANDOFF.md and per-source comparison
collages. The damage mask is derived from cataloged texture material pixels,
never from a detector prediction. No procedural aging operation is allowed.
All randomness comes from a single per-variant numpy Generator so every
variant is reproducible from the seed recorded in metadata.

Usage (from the repo root):

    python -m scripts.synthetic.generate --make-fixture
    python -m scripts.synthetic.generate --clean-dir artifacts/synthetic_fixture/clean --output-dir artifacts/synthetic_fixture/out --limit 3
    python -m scripts.synthetic.generate --verify-only --output-dir artifacts/synthetic_fixture/out
    python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --sample 20
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from scripts.synthetic.engine import (
    DEFAULT_CATALOG_PATH,
    SEVERITY_PROFILE,
    DegradationConfig,
    build_source_collage,
    catalog_sha256,
    coverage_of,
    degrade_target,
    derive_variant_seed,
    evaluate_damage_score,
    load_texture_catalog,
    save_variant_files,
    texture_catalog_summary,
    validate_material_only_metadata,
    write_source_metadata,
)
from scripts.synthetic.texture_assets import PROJECT_ROOT

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.data.team_dataset_workflow import (
    IMAGE_SUFFIXES,
    read_csv,
    sha256,
    utc_now,
    write_csv,
    write_json,
)

DATA_DIR = PROJECT_ROOT / "data"

SYNTHETIC_MANIFEST_FIELDS = [
    "source_id", "severity", "seed", "restoration_type", "mask_coverage",
    "damage_score", "layer_count", "label_combination", "width", "height",
    "target_path", "degraded_path", "mask_path", "metadata_path",
    "sha256_degraded", "created_at",
]

DEFAULT_FIXTURE_DIR = Path("artifacts/synthetic_fixture/clean")
FIXTURE_SIZES = [(512, 384), (320, 240), (256, 256)]


def create_review_contact_sheet(
    paths: list[Path],
    output: Path,
    sample_size: int = 36,
) -> None:
    """Render wide comparison collages at a legible two-column review size."""
    if not paths:
        return
    selected = paths[:sample_size]
    columns = min(2, len(selected))
    cell_width, cell_height, label_height, padding = 620, 300, 28, 12
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * padding,
            rows * (cell_height + label_height) + (rows + 1) * padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail(
                (cell_width - 2 * padding, cell_height - 2 * padding),
                Image.Resampling.LANCZOS,
            )
        column = index % columns
        row = index // columns
        cell_x = padding + column * cell_width
        cell_y = padding + row * (cell_height + label_height)
        image_x = cell_x + (cell_width - image.width) // 2
        image_y = cell_y + (cell_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.text(
            (cell_x + 4, cell_y + cell_height + 4),
            path.stem[:72],
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=92)


def _markdown_relative_path(path: Path, document_dir: Path) -> str:
    return os.path.relpath(path, document_dir).replace("\\", "/")


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def write_mask_handoff(
    output_root: Path,
    destination: Path | None = None,
) -> Path:
    """Write a portable mask inventory grouped by main--sub label combinations."""
    output_root = Path(output_root)
    destination = (
        Path(destination)
        if destination is not None
        else output_root / "MASK_HANDOFF.md"
    )
    manifest_path = output_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot build handoff without {manifest_path}")

    rows = read_csv(manifest_path)
    label_counts: Counter[str] = Counter()
    texture_counts: Counter[str] = Counter()
    catalog_hashes: set[str] = set()
    row_details: list[dict] = []
    for row in rows:
        metadata_path = output_root / row["metadata_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        variant = metadata["variants"][row["severity"]]
        catalog_hashes.add(
            str((variant.get("material_catalog") or {}).get("sha256", ""))
        )
        steps = variant.get("steps") or []
        labels = list(
            dict.fromkeys(
                str(step["params"]["label_path"])
                for step in steps
            )
        )
        textures = [str(step["params"]["texture"]) for step in steps]
        label_counts.update(labels)
        texture_counts.update(textures)
        row_details.append(
            {
                **row,
                "labels": labels,
                "textures": textures,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    document_dir = destination.parent
    catalog_hash = ", ".join(sorted(value for value in catalog_hashes if value))
    lines = [
        "# Mask 素材标签与破损评分交接表",
        "",
        "## 交付概况",
        "",
        f"- 生成时间：{utc_now()}",
        f"- 样例目录：`{_markdown_cell(_markdown_relative_path(output_root, document_dir))}`",
        f"- Mask 数量：{len(row_details)}",
        f"- 素材目录哈希：`{catalog_hash}`",
        "- 生成模式：仅组合已分类素材，不使用程序生成的做旧纹理。",
        "- 标签来源：人工视觉分类（`manual_visual`），不是 OCR。",
        "- 标签格式：`主标签--子标签`；多标签组合用 ` + ` 连接。",
        "",
        "## 破损评分",
        "",
        "| 分量 | 最高分 | 含义 |",
        "|---|---:|---|",
        "| 覆盖率 | 50 | 最终二值 mask 的受损像素比例，32% 覆盖时达到该分量上限 |",
        "| 素材强度 | 32 | 各层覆盖率、透明度、破损类别、尺度、密度和分布加权累计 |",
        "| 层数存在度 | 8 | 随实际组合层数对数增长，不用固定素材数量限制等级 |",
        "| 标签多样性 | 10 | 组合中不同主标签和子标签的丰富度 |",
        "",
        "| 等级 | 分数范围 | 覆盖率约束 |",
        "|---|---:|---:|",
        "| light | 15-34 | 1%-8% |",
        "| medium | 35-64 | 5%-20% |",
        "| heavy | 65-90 | 15%-32% |",
        "",
        "## 标签使用统计",
        "",
        "| 主标签--子标签 | 使用该标签的 Mask 数 |",
        "|---|---:|",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| `{_markdown_cell(label)}` | {count} |")

    lines.extend(
        [
            "",
            "## 素材使用统计",
            "",
            "| 素材文件 | 使用次数 |",
            "|---|---:|",
        ]
    )
    for texture, count in sorted(texture_counts.items()):
        lines.append(f"| `{_markdown_cell(texture)}` | {count} |")

    grouped: dict[str, list[dict]] = {}
    for row in row_details:
        grouped.setdefault(row["label_combination"], []).append(row)
    lines.extend(["", "## 全部 Mask", ""])
    for combination in sorted(grouped):
        lines.extend(
            [
                f"### `{_markdown_cell(combination)}`",
                "",
                "| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |",
                "|---|---|---:|---:|---:|---|---|",
            ]
        )
        group_rows = sorted(
            grouped[combination],
            key=lambda item: (
                item["source_id"],
                {"light": 1, "medium": 2, "heavy": 3}[item["severity"]],
            ),
        )
        for row in group_rows:
            mask_path = _markdown_relative_path(
                output_root / row["mask_path"],
                document_dir,
            )
            degraded_path = _markdown_relative_path(
                output_root / row["degraded_path"],
                document_dir,
            )
            metadata_path = _markdown_relative_path(
                output_root / row["metadata_path"],
                document_dir,
            )
            textures = "<br>".join(
                f"`{_markdown_cell(texture)}`"
                for texture in row["textures"]
            )
            files = (
                f"[degraded](<{degraded_path}>)<br>"
                f"[metadata](<{metadata_path}>)"
            )
            lines.append(
                f"| ![{_markdown_cell(row['source_id'])} "
                f"{row['severity']}](<{mask_path}>) | "
                f"`{_markdown_cell(row['source_id'])}` / "
                f"`{row['severity']}` | "
                f"{float(row['damage_score']):.2f} | "
                f"{float(row['mask_coverage']):.2f}% | "
                f"{int(row['layer_count'])} | {textures} | {files} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 交接字段",
            "",
            "- `damage_score`：0-100 的综合破损分数。",
            "- `layer_count`：该图片实际组合的素材层数。",
            "- `label_combination`：按首次出现顺序去重后的标签组合。",
            "- `damage_assessment.layer_scores`：每一层素材的评分明细。",
            "- `category_recipe`：组合顺序、标签、类别和素材文件。",
            "- `mask_coverage`：最终联合二值 mask 的覆盖率。",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.batch_id:
        return DATA_DIR / "work" / "batches" / args.batch_id / "05_synthetic_pairs"
    raise SystemExit("--output-dir or --batch-id is required")


def _select_sources(
    clean_dir: Path, manifest_path: Path | None, limit: int | None
) -> list[tuple[str, Path]]:
    """Return [(source_id, image_path)] sorted by source_id."""
    clean_dir = Path(clean_dir)
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean image directory not found: {clean_dir}")
    if manifest_path is not None and Path(manifest_path).is_file():
        rows = read_csv(Path(manifest_path))
        accepted = [row for row in rows if row.get("quality_label") == "clean_target_ok"]
        files: dict[str, Path] = {}
        for row in accepted:
            filename = row.get("filename", "")
            image_path = clean_dir / filename
            if image_path.is_file():
                source_id = row.get("source_id") or Path(filename).stem
                files.setdefault(source_id, image_path)
        if not files:
            raise RuntimeError(
                f"No images with quality_label=clean_target_ok in {Path(manifest_path)}"
            )
        sources = sorted(files.items(), key=lambda item: item[0])
    else:
        sources = sorted(
            (path.stem, path)
            for path in clean_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if limit:
        sources = sources[:limit]
    return sources


def make_fixture(args: argparse.Namespace) -> None:
    """Create small deterministic 'clean' targets for offline testing."""
    fixture_dir = Path(args.fixture_dir) if args.fixture_dir else DEFAULT_FIXTURE_DIR
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    for index, (width, height) in enumerate(FIXTURE_SIZES, start=1):
        yy, xx = np.mgrid[:height, :width]
        red = np.clip(xx / max(width - 1, 1) * 255, 0, 255)
        green = np.clip(yy / max(height - 1, 1) * 255, 0, 255)
        blue = np.clip(150 + 80 * np.sin(xx / 35.0), 0, 255)
        arr = np.dstack([red, green, blue]).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            outline=(210, 200, 60),
            width=max(2, min(width, height) // 40),
        )
        draw.ellipse(
            [width // 2 - 40, height // 2 - 40, width // 2 + 40, height // 2 + 40],
            fill=(40, 160, 220),
        )
        # a few soft gray blobs to break up flatness
        blob_mask = Image.new("L", (width, height), 0)
        blob_draw = ImageDraw.Draw(blob_mask)
        blob_radius = max(20, min(width, height) // 8)
        for _ in range(6):
            cx = int(rng.integers(0, width))
            cy = int(rng.integers(0, height))
            r = int(rng.integers(20, blob_radius))
            blob_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        blob_mask = blob_mask.filter(ImageFilter.GaussianBlur(max(2, blob_radius // 6)))
        gray = Image.new("RGB", (width, height), (118, 118, 118))
        image = Image.composite(gray, image, blob_mask)
        image = image.filter(ImageFilter.GaussianBlur(1.2))
        output = fixture_dir / f"fixture_{index}.png"
        image.save(output, "PNG")
        print(f"fixture written: {output}")
    print(f"fixture directory: {fixture_dir}")


def generate(args: argparse.Namespace, sources: list[tuple[str, Path]], output_root: Path) -> None:
    output_root = Path(output_root)
    severities = args.severities
    cfg = DegradationConfig(
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        texture_dir=Path(args.texture_dir) if args.texture_dir else None,
        catalog_path=Path(args.texture_catalog) if args.texture_catalog else None,
    )
    catalog_summary = texture_catalog_summary(cfg.texture_dir, cfg.catalog_path)

    target_dir = output_root / "target"
    degraded_dir = output_root / "degraded"
    masks_dir = output_root / "masks"
    metadata_dir = output_root / "metadata"
    collages_dir = output_root / "collages"
    for directory in (target_dir, degraded_dir, masks_dir, metadata_dir, collages_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Start from any existing manifest so idempotent reruns never lose records.
    manifest_index: dict[tuple[str, str], dict] = {}
    manifest_csv = output_root / "manifest.csv"
    if manifest_csv.is_file():
        for row in read_csv(manifest_csv):
            manifest_index[(row["source_id"], row["severity"])] = row

    processed_sources: list[str] = []
    new_variants = 0

    for index, (source_id, image_path) in enumerate(sources):
        target_out = target_dir / f"{source_id}.png"
        if not target_out.exists() or args.force:
            if image_path.suffix.lower() == ".png":
                shutil.copy2(image_path, target_out)
            else:
                with Image.open(image_path) as opened:
                    ImageOps.exif_transpose(opened).convert("RGB").save(
                        target_out, "PNG", compress_level=1
                    )
        target_sha = sha256(target_out)

        # single metadata file per source; if it exists the source is complete
        metadata_path = metadata_dir / f"{source_id}.json"
        if metadata_path.exists() and not args.force:
            print(f"skip {source_id} (metadata exists)")
            continue

        variants_by_sev: dict[str, object] = {}
        for severity in severities:
            seed = derive_variant_seed(args.global_seed, index, severity)
            with Image.open(target_out) as opened:
                target_image = opened.convert("RGB")
            variant = degrade_target(target_image, severity, seed, cfg)
            save_variant_files(output_root, source_id, severity, variant)
            variants_by_sev[severity] = variant

            processed_sources.append(source_id)
            new_variants += 1
            manifest_index[(source_id, severity)] = {
                "source_id": source_id,
                "severity": severity,
                "seed": seed,
                "restoration_type": variant.metadata["restoration_type"],
                "mask_coverage": round(variant.coverage, 4),
                "damage_score": variant.metadata["damage_score"],
                "layer_count": variant.metadata["layer_count"],
                "label_combination": variant.metadata["label_combination"],
                "width": variant.degraded.width,
                "height": variant.degraded.height,
                "target_path": f"target/{source_id}.png",
                "degraded_path": f"degraded/{source_id}_{severity}.png",
                "mask_path": f"masks/{source_id}_{severity}.png",
                "metadata_path": f"metadata/{source_id}.json",
                "sha256_degraded": sha256(degraded_dir / f"{source_id}_{severity}.png"),
                "created_at": utc_now(),
            }
        if variants_by_sev:
            write_source_metadata(output_root, source_id, target_sha, variants_by_sev)

        # comparison collage for this source (all requested severities that exist)
        if not args.no_collages:
            existing = [
                severity
                for severity in severities
                if (degraded_dir / f"{source_id}_{severity}.png").exists()
            ]
            if existing:
                build_source_collage(
                    target_out,
                    [degraded_dir / f"{source_id}_{severity}.png" for severity in existing],
                    [masks_dir / f"{source_id}_{severity}.png" for severity in existing],
                    collages_dir / f"{source_id}_comparison.png",
                    existing,
                )
            else:
                print(f"warn: no variants for {source_id}, collage skipped")

    manifest_rows = [manifest_index[key] for key in sorted(manifest_index)]
    write_csv(output_root / "manifest.csv", manifest_rows, SYNTHETIC_MANIFEST_FIELDS)

    coverage_values = [
        float(row["mask_coverage"])
        for row in manifest_rows
    ]
    damage_score_values = [
        float(row["damage_score"])
        for row in manifest_rows
    ]
    layer_count_values = [
        float(row["layer_count"])
        for row in manifest_rows
    ]
    counts = Counter(row["severity"] for row in manifest_rows)
    restoration_counter = Counter(
        row["restoration_type"]
        for row in manifest_rows
    )
    ops_counter = Counter()
    for row in manifest_rows:
        metadata = json.loads(
            (output_root / row["metadata_path"]).read_text(encoding="utf-8")
        )
        variant = metadata["variants"][row["severity"]]
        ops_counter.update(
            step["op"]
            for step in variant.get("steps") or []
        )

    collage_paths = sorted(collages_dir.glob("*_comparison.png"))
    if collage_paths:
        create_review_contact_sheet(
            collage_paths, output_root / "contact_sheet.jpg"
        )

    if args.sample:
        # processed_sources lists one entry per variant; take the first N distinct sources
        sampled_sources: list[str] = []
        for source_id in processed_sources:
            if source_id not in sampled_sources:
                sampled_sources.append(source_id)
            if len(sampled_sources) >= args.sample:
                break
        sample_rows = [
            row for row in manifest_rows
            if row["source_id"] in set(sampled_sources)
        ]
        write_csv(output_root / "sample_manifest.csv", sample_rows, SYNTHETIC_MANIFEST_FIELDS)
        if collage_paths:
            create_review_contact_sheet(
                collage_paths[: args.sample], output_root / "contact_sheet_sample.jpg"
            )

    coverage_summary = _coverage_stats(coverage_values)
    summary = {
        "command": " ".join(sys.argv),
        "global_seed": args.global_seed,
        "generation_mode": "material_only",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "counts": {
            "sources": len({row["source_id"] for row in manifest_rows}),
            "variants": len(manifest_rows),
            "per_severity": dict(counts),
        },
        "coverage": coverage_summary,
        "damage_score": _numeric_stats(damage_score_values),
        "layer_count": _numeric_stats(layer_count_values),
        "restoration_types": dict(restoration_counter),
        "ops_used": dict(ops_counter),
        "material_catalog": catalog_summary,
        "warnings": cfg.warnings,
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "run_report.json", {
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "created_at": utc_now(),
        "output_dir": str(output_root),
        "processed_sources": processed_sources,
    })
    handoff_path = (
        Path(args.handoff_md)
        if args.handoff_md
        else output_root / "MASK_HANDOFF.md"
    )
    write_mask_handoff(output_root, handoff_path)

    print(
        f"generated {new_variants} new variants this run; "
        f"manifest has {len(manifest_rows)} total -> {output_root}; "
        f"handoff -> {handoff_path}"
    )


def _coverage_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None, "histogram": {}}
    buckets = {"0-1": 0, "1-5": 0, "5-10": 0, "10-20": 0, "20-35": 0, ">35": 0}
    for value in values:
        if value < 1:
            buckets["0-1"] += 1
        elif value < 5:
            buckets["1-5"] += 1
        elif value < 10:
            buckets["5-10"] += 1
        elif value < 20:
            buckets["10-20"] += 1
        elif value <= 35:
            buckets["20-35"] += 1
        else:
            buckets[">35"] += 1
    return {
        "min": round(min(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "max": round(max(values), 3),
        "histogram": buckets,
    }


def _numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "max": round(max(values), 3),
    }


def rebuild_contact_sheet(args: argparse.Namespace, output_root: Path) -> None:
    collages = sorted((output_root / "collages").glob("*_comparison.png"))
    if not collages:
        print("no collages found; nothing to rebuild")
        return
    create_review_contact_sheet(collages, output_root / "contact_sheet.jpg")
    print(f"contact sheet rebuilt: {output_root / 'contact_sheet.jpg'}")


def verify(args: argparse.Namespace, output_root: Path) -> None:
    """Validate an existing generation; exit 2 on any failure."""
    output_root = Path(output_root)
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    verify_cfg = DegradationConfig(
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        texture_dir=Path(args.texture_dir) if args.texture_dir else None,
        catalog_path=Path(args.texture_catalog) if args.texture_catalog else None,
    )
    expected_catalog_sha = catalog_sha256(verify_cfg.catalog_path)
    catalog_records = {
        record.file: record
        for record in load_texture_catalog(
            verify_cfg.texture_dir,
            verify_cfg.catalog_path,
        )
    }
    manifest_path = output_root / "manifest.csv"
    if not manifest_path.is_file():
        errors.append("manifest.csv missing")
    else:
        rows = read_csv(manifest_path)
        variants = 0
        reproduction_checked = False
        for row in rows:
            source_id = row["source_id"]
            severity = row["severity"]
            variants += 1
            rel = {
                "target": row["target_path"],
                "degraded": row["degraded_path"],
                "mask": row["mask_path"],
            }
            for label, rel_path in rel.items():
                path = output_root / rel_path
                if not path.is_file():
                    errors.append(f"{source_id}/{severity}: {label} file missing: {rel_path}")
            mask_path = output_root / rel["mask"]
            degraded_path = output_root / rel["degraded"]
            target_path = output_root / rel["target"]
            meta_path = output_root / row["metadata_path"]

            if not all(p.is_file() for p in (mask_path, degraded_path, target_path, meta_path)):
                continue

            degraded_size = None
            saved_mask = None
            saved_degraded = None
            with Image.open(target_path) as target_img, \
                 Image.open(degraded_path) as degraded_img, \
                 Image.open(mask_path) as mask_img:
                if degraded_img.size != target_img.size:
                    errors.append(f"{source_id}/{severity}: degraded size != target size")
                if mask_img.size != degraded_img.size:
                    errors.append(f"{source_id}/{severity}: mask size != degraded size")
                if mask_img.mode != "L":
                    errors.append(f"{source_id}/{severity}: mask mode is {mask_img.mode}, expected L")
                mask_values = set(np.asarray(mask_img).ravel().tolist())
                if not mask_values.issubset({0, 255}):
                    non_binary = sorted(v for v in mask_values if v not in (0, 255))[:5]
                    errors.append(f"{source_id}/{severity}: mask has non-binary values {non_binary}")
                restored_type = row["restoration_type"]
                if restored_type != "local_repair":
                    errors.append(
                        f"{source_id}/{severity}: restoration_type must be local_repair"
                    )
                if not mask_values or max(mask_values) == 0:
                    errors.append(f"{source_id}/{severity}: material mask is empty")
                mask_array = np.asarray(mask_img)
                target_array = np.asarray(target_img.convert("RGB"))
                degraded_array = np.asarray(degraded_img.convert("RGB"))
                damaged = mask_array > 0
                if damaged.any():
                    if np.any(target_array[~damaged] != degraded_array[~damaged]):
                        errors.append(
                            f"{source_id}/{severity}: pixels changed outside material mask"
                        )
                    if not np.any(target_array[damaged] != degraded_array[damaged]):
                        errors.append(
                            f"{source_id}/{severity}: material mask contains no changed pixels"
                        )
                recomputed = coverage_of(mask_img)
                degraded_size = degraded_img.size
                saved_mask = mask_array.copy()
                saved_degraded = degraded_array.copy()
            if sha256(degraded_path) != row.get("sha256_degraded"):
                errors.append(f"{source_id}/{severity}: degraded SHA-256 mismatch")
            recorded = float(row["mask_coverage"])
            if abs(recomputed - recorded) > 0.5:
                errors.append(
                    f"{source_id}/{severity}: recomputed coverage {recomputed:.2f} "
                    f"!= recorded {recorded:.2f}"
                )
            profile_lo, profile_hi = SEVERITY_PROFILE[severity]["coverage_target"]
            lo = max(args.min_coverage, profile_lo) - 0.5
            hi = min(args.max_coverage, profile_hi) + 0.5
            if not (lo <= recomputed <= hi):
                errors.append(
                    f"{source_id}/{severity}: coverage {recomputed:.2f} outside "
                    f"severity target [{lo:.2f}, {hi:.2f}]"
                )

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            variant_meta = metadata.get("variants", {}).get(severity)
            if variant_meta is None:
                errors.append(f"{source_id}/{severity}: variant missing in single metadata file")
            else:
                if variant_meta.get("severity") != severity:
                    errors.append(f"{source_id}/{severity}: metadata severity mismatch")
                if int(variant_meta.get("seed")) != int(row["seed"]):
                    errors.append(f"{source_id}/{severity}: metadata seed mismatch")
                if variant_meta.get("restoration_type") != restored_type:
                    errors.append(f"{source_id}/{severity}: metadata restoration_type mismatch")
                for material_error in validate_material_only_metadata(variant_meta):
                    errors.append(
                        f"{source_id}/{severity}: material metadata {material_error}"
                    )
                material_catalog = variant_meta.get("material_catalog") or {}
                if material_catalog.get("sha256") != expected_catalog_sha:
                    errors.append(f"{source_id}/{severity}: material catalog hash mismatch")
                steps = variant_meta.get("steps") or []
                if not steps:
                    errors.append(f"{source_id}/{severity}: metadata has no steps")
                assessment = evaluate_damage_score(recomputed, steps)
                score = float(assessment["score"])
                score_lo, score_hi = SEVERITY_PROFILE[severity]["score_target"]
                if not (score_lo <= score <= score_hi):
                    errors.append(
                        f"{source_id}/{severity}: damage score {score:.2f} "
                        f"outside [{score_lo:.2f}, {score_hi:.2f}]"
                    )
                if abs(score - float(row.get("damage_score", -1.0))) > 0.05:
                    errors.append(
                        f"{source_id}/{severity}: manifest damage score mismatch"
                    )
                if int(row.get("layer_count", -1)) != len(steps):
                    errors.append(
                        f"{source_id}/{severity}: manifest layer count mismatch"
                    )
                label_paths = list(
                    dict.fromkeys(
                        str((step.get("params") or {}).get("label_path"))
                        for step in steps
                    )
                )
                label_combination = " + ".join(label_paths)
                if row.get("label_combination") != label_combination:
                    errors.append(
                        f"{source_id}/{severity}: manifest label combination "
                        f"mismatch"
                    )
                for step in steps:
                    if "op" not in step or "params" not in step:
                        errors.append(f"{source_id}/{severity}: malformed step entry")
                        continue
                    params = step["params"]
                    texture_name = params.get("texture")
                    record = catalog_records.get(texture_name)
                    if record is None or not record.enabled:
                        errors.append(
                            f"{source_id}/{severity}: unapproved material "
                            f"{texture_name!r}"
                        )
                    elif (
                        params.get("category") != record.primary_category
                        or params.get("main_label") != record.main_label
                        or params.get("sub_label") != record.sub_label
                        or params.get("label_path") != record.label_path
                        or params.get("label_source") != record.label_source
                        or params.get("blend_mode") != record.blend_mode
                    ):
                        errors.append(
                            f"{source_id}/{severity}: material catalog fields "
                            f"do not match {texture_name}"
                        )
                size = variant_meta.get("image_size")
                if degraded_size is not None and size != list(degraded_size):
                    errors.append(f"{source_id}/{severity}: metadata image_size mismatch")
                actual_recipe = variant_meta.get("category_recipe") or []
                if len(actual_recipe) != len(steps):
                    errors.append(
                        f"{source_id}/{severity}: recipe/step count mismatch"
                    )
                profile = SEVERITY_PROFILE[severity]
                allowed_groups: dict[str, set[str]] = {}
                for slot in (
                    profile["required_slots"] + profile["optional_slots"]
                ):
                    allowed_groups.setdefault(slot["group"], set()).update(
                        slot["categories"]
                    )
                for slot in profile["required_slots"]:
                    if not any(
                        item.get("group") == slot["group"]
                        and item.get("category") in slot["categories"]
                        for item in actual_recipe
                    ):
                        errors.append(
                            f"{source_id}/{severity}: missing required group "
                            f"{slot['group']}"
                        )
                for item in actual_recipe:
                    group = item.get("group")
                    if (
                        group not in allowed_groups
                        or item.get("category") not in allowed_groups[group]
                    ):
                        errors.append(
                            f"{source_id}/{severity}: recipe item outside "
                            f"allowed label groups"
                        )

            # reproduction spot check (first variant only) - strongest guarantee
            if not reproduction_checked:
                try:
                    with Image.open(target_path) as target_img:
                        target_image = target_img.convert("RGB")
                    seed = int(row["seed"])
                    variant = degrade_target(
                        target_image, severity, seed,
                        verify_cfg,
                    )
                    regenerated = np.asarray(variant.mask)
                    if saved_mask is None or not np.array_equal(regenerated, saved_mask):
                        errors.append(f"{source_id}/{severity}: reproduction mask mismatch")
                    regenerated_image = np.asarray(variant.degraded)
                    if (
                        saved_degraded is None
                        or not np.array_equal(regenerated_image, saved_degraded)
                    ):
                        errors.append(
                            f"{source_id}/{severity}: reproduction image mismatch"
                        )
                    reproduction_checked = True
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"reproduction check failed: {exc}")

        expected = len(set((row["source_id"], row["severity"]) for row in rows))
        if variants != len(rows):
            errors.append(f"manifest row count mismatch: {variants} vs {len(rows)}")
        if not rows:
            errors.append("no variants recorded in manifest.csv")

    report = {
        "checked_at": utc_now(),
        "valid": not errors,
        "errors": errors[:200],
        "warnings": warnings,
    }
    write_json(output_root / "verify_report.json", report)
    if errors:
        print(f"VERIFY FAILED: {len(errors)} issue(s). See {output_root / 'verify_report.json'}")
        for error in errors[:30]:
            print(f"  - {error}")
        raise SystemExit(2)
    print(f"VERIFY PASSED: {len(rows)} variants checked in {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic degradation pairs + masks"
    )
    parser.add_argument("--batch-id", help="batch id; defaults output under data/work/batches")
    parser.add_argument("--clean-dir", help="directory of clean target images")
    parser.add_argument("--output-dir", help="output root (default: <batch>/05_synthetic_pairs)")
    parser.add_argument("--manifest", help="optional manifest.csv to filter clean_target_ok")
    parser.add_argument("--severities", nargs="+", default=["light", "medium", "heavy"],
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--max-coverage", type=float, default=35.0)
    parser.add_argument(
        "--texture-dir",
        type=Path,
        default=None,
        help="source texture directory (default: project Resource Boy pack)",
    )
    parser.add_argument(
        "--texture-catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="classified material catalog CSV",
    )
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="process first N sources")
    parser.add_argument("--sample", type=int, default=None,
                        help="like --limit but also writes sample_manifest.csv + sample contact sheet")
    parser.add_argument("--no-collages", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing variants")
    parser.add_argument(
        "--handoff-md",
        help="write the grouped mask handoff Markdown to this path",
    )
    parser.add_argument("--contact-sheet-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--fixture-dir", help="override fixture clean-image directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.make_fixture:
        make_fixture(args)
        return

    output_root = _resolve_output_root(args)

    if args.verify_only:
        verify(args, output_root)
        return
    if args.contact_sheet_only:
        rebuild_contact_sheet(args, output_root)
        return

    batch = args.batch_id
    if not batch and not args.clean_dir:
        raise SystemExit(
            "Provide --batch-id (defaults to data/work/batches/<batch>/05_synthetic_pairs) "
            "or --clean-dir + --output-dir"
        )
    clean_dir = Path(args.clean_dir) if args.clean_dir else (
        DATA_DIR / "work" / "batches" / batch / "02_cleaned" / "accepted"
    )
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None and batch:
        candidate = DATA_DIR / "work" / "batches" / batch / "02_cleaned" / "manifest.csv"
        if candidate.is_file():
            manifest_path = candidate

    limit = args.limit if args.limit is not None else args.sample
    sources = _select_sources(clean_dir, manifest_path, limit)
    if not sources:
        print(f"no clean sources found in {clean_dir}")
        raise SystemExit(2)
    print(f"processing {len(sources)} sources from {clean_dir}")
    generate(args, sources, output_root)


if __name__ == "__main__":
    main()

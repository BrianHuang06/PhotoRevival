"""Generate reproducible light/medium/heavy synthetic degradation pairs.

This is the C-role deliverable of the team dataset workflow: it turns clean
target images into paired training data. For every clean image it writes:

    target/<source_id>.png
    degraded/<source_id>_<severity>.png
    masks/<source_id>_<severity>.png        # 0 = clean, 255 = damaged
    metadata/<source_id>_<severity>.json    # seed + every applied step/param

plus a manifest.csv, summary.json and per-source comparison collages. The
local-damage mask is derived from the damage layer alpha channel, never from a
detector prediction. All randomness comes from a single per-variant numpy
Generator so every variant is reproducible from the seed recorded in metadata.

Usage (from the repo root):

    python -m scripts.synthetic.generate --make-fixture
    python -m scripts.synthetic.generate --clean-dir artifacts/synthetic_fixture/clean --output-dir artifacts/synthetic_fixture/out --limit 3
    python -m scripts.synthetic.generate --verify-only --output-dir artifacts/synthetic_fixture/out
    python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --sample 20
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Make ``photo_revival`` importable even when the package is not installed.
try:  # pragma: no cover - environment dependent
    import photo_revival  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from photo_revival.paths import DATA_DIR
from scripts.synthetic.engine import (
    DegradationConfig,
    build_source_collage,
    coverage_of,
    degrade_target,
    derive_variant_seed,
    save_variant,
)
from scripts.data.team_dataset_workflow import (
    IMAGE_SUFFIXES,
    create_contact_sheet,
    read_csv,
    sha256,
    utc_now,
    write_csv,
    write_json,
)

SYNTHETIC_MANIFEST_FIELDS = [
    "source_id", "severity", "seed", "restoration_type", "mask_coverage",
    "width", "height", "target_path", "degraded_path", "mask_path",
    "metadata_path", "sha256_degraded", "created_at",
]

DEFAULT_FIXTURE_DIR = Path("artifacts/synthetic_fixture/clean")
FIXTURE_SIZES = [(512, 384), (320, 240), (256, 256)]


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
        fill_mode=args.fill_mode,
        global_enabled=True,
        local_enabled=not args.global_only,
        imaging_enabled=True,
    )

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

    counts: Counter[str] = Counter()
    restoration_counter: Counter[str] = Counter()
    ops_counter: Counter[str] = Counter()
    coverage_values: list[float] = []
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

        for severity in severities:
            metadata_path = metadata_dir / f"{source_id}_{severity}.json"
            if metadata_path.exists() and not args.force:
                print(f"skip {source_id} {severity} (metadata exists)")
                continue
            seed = derive_variant_seed(args.global_seed, index, severity)
            with Image.open(target_out) as opened:
                target_image = opened.convert("RGB")
            variant = degrade_target(target_image, severity, seed, cfg)
            save_variant(output_root, source_id, severity, variant, target_sha)

            processed_sources.append(source_id)
            new_variants += 1
            counts[severity] += 1
            restoration_counter[variant.metadata["restoration_type"]] += 1
            coverage_values.append(variant.coverage)
            for step in variant.metadata["steps"]:
                ops_counter[step["op"]] += 1
            manifest_index[(source_id, severity)] = {
                "source_id": source_id,
                "severity": severity,
                "seed": seed,
                "restoration_type": variant.metadata["restoration_type"],
                "mask_coverage": round(variant.coverage, 4),
                "width": variant.degraded.width,
                "height": variant.degraded.height,
                "target_path": f"target/{source_id}.png",
                "degraded_path": f"degraded/{source_id}_{severity}.png",
                "mask_path": f"masks/{source_id}_{severity}.png",
                "metadata_path": f"metadata/{source_id}_{severity}.json",
                "sha256_degraded": sha256(degraded_dir / f"{source_id}_{severity}.png"),
                "created_at": utc_now(),
            }

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

    collage_paths = sorted(collages_dir.glob("*_comparison.png"))
    if collage_paths:
        create_contact_sheet(collage_paths, output_root / "contact_sheet.jpg")

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
            create_contact_sheet(
                collage_paths[: args.sample], output_root / "contact_sheet_sample.jpg"
            )

    coverage_summary = _coverage_stats(coverage_values)
    summary = {
        "command": " ".join(sys.argv),
        "global_seed": args.global_seed,
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "counts": {
            "sources": len(set(processed_sources)),
            "variants": len(manifest_rows),
            "per_severity": dict(counts),
        },
        "coverage": coverage_summary,
        "restoration_types": dict(restoration_counter),
        "ops_used": dict(ops_counter),
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

    print(
        f"generated {new_variants} new variants this run; "
        f"manifest has {len(manifest_rows)} total -> {output_root}"
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


def rebuild_contact_sheet(args: argparse.Namespace, output_root: Path) -> None:
    collages = sorted((output_root / "collages").glob("*_comparison.png"))
    if not collages:
        print("no collages found; nothing to rebuild")
        return
    create_contact_sheet(collages, output_root / "contact_sheet.jpg")
    print(f"contact sheet rebuilt: {output_root / 'contact_sheet.jpg'}")


def verify(args: argparse.Namespace, output_root: Path) -> None:
    """Validate an existing generation; exit 2 on any failure."""
    output_root = Path(output_root)
    errors: list[str] = []
    warnings: list[str] = []
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
                if restored_type == "local_repair" and max(mask_values) == 0:
                    errors.append(f"{source_id}/{severity}: empty mask for local_repair sample")
                if restored_type == "global_restoration" and max(mask_values) != 0:
                    errors.append(f"{source_id}/{severity}: non-empty mask for global_restoration sample")
                recomputed = coverage_of(mask_img)
                degraded_size = degraded_img.size
                saved_mask = np.asarray(mask_img)
            recorded = float(row["mask_coverage"])
            if abs(recomputed - recorded) > 0.5:
                errors.append(
                    f"{source_id}/{severity}: recomputed coverage {recomputed:.2f} "
                    f"!= recorded {recorded:.2f}"
                )
            if restored_type == "local_repair":
                lo = args.min_coverage - 0.5
                hi = args.max_coverage + 0.5
                if not (lo <= recomputed <= hi):
                    errors.append(
                        f"{source_id}/{severity}: coverage {recomputed:.2f} outside "
                        f"[{lo:.2f}, {hi:.2f}]"
                    )

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("severity") != severity:
                errors.append(f"{source_id}/{severity}: metadata severity mismatch")
            if int(metadata.get("seed")) != int(row["seed"]):
                errors.append(f"{source_id}/{severity}: metadata seed mismatch")
            if metadata.get("restoration_type") != restored_type:
                errors.append(f"{source_id}/{severity}: metadata restoration_type mismatch")
            steps = metadata.get("steps") or []
            if not steps:
                errors.append(f"{source_id}/{severity}: metadata has no steps")
            for step in steps:
                if "op" not in step or "params" not in step:
                    errors.append(f"{source_id}/{severity}: malformed step entry")
            size = metadata.get("image_size")
            if degraded_size is not None and size != list(degraded_size):
                errors.append(f"{source_id}/{severity}: metadata image_size mismatch")

            # reproduction spot check (first variant only) - strongest guarantee
            if not reproduction_checked and restored_type == "local_repair":
                try:
                    with Image.open(target_path) as target_img:
                        target_image = target_img.convert("RGB")
                    seed = int(row["seed"])
                    variant = degrade_target(
                        target_image, severity, seed,
                        DegradationConfig(
                            min_coverage=args.min_coverage,
                            max_coverage=args.max_coverage,
                            fill_mode=args.fill_mode,
                        ),
                    )
                    regenerated = np.asarray(variant.mask)
                    if saved_mask is None or not np.array_equal(regenerated, saved_mask):
                        errors.append(f"{source_id}/{severity}: reproduction mask mismatch")
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
    parser.add_argument("--fill-mode", default="paper", choices=["paper", "blurred_patch"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="process first N sources")
    parser.add_argument("--sample", type=int, default=None,
                        help="like --limit but also writes sample_manifest.csv + sample contact sheet")
    parser.add_argument("--global-only", action="store_true",
                        help="emit global-only variants (empty masks, global_restoration)")
    parser.add_argument("--no-collages", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing variants")
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

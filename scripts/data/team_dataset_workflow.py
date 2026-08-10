"""Run the team's traceable real-photo dataset workflow.

This module intentionally stops at human annotation. Automatic checks may prepare
work for annotators, but they must never invent damage labels or approval records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageOps

from photo_revival.paths import DATA_DIR


WORK_DIR = DATA_DIR / "work" / "batches"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_FIELDS = [
    "id", "source_id", "filename", "status", "dataset_kind", "provider",
    "source_url", "author", "license", "rights_url", "downloaded_at",
    "sha256", "width", "height", "content_type", "color_mode",
    "quality_label", "split", "annotator", "reviewer", "notes",
]
DAMAGE_FIELDS = [
    "source_id", "scratch", "crack", "fold", "dust", "stain", "mold",
    "tear", "missing_region", "fading", "color_cast", "blur", "noise",
    "jpeg_artifact", "uneven_exposure", "mask_path", "uncertain",
    "annotator", "reviewer", "status", "notes",
]
DAMAGE_LABELS = DAMAGE_FIELDS[1:15]
ACCEPTED_LICENSES = {
    "public domain",
    "cc0",
    "cc0 1.0",
    "cc0 1.0 universal",
    "cc by 3.0",
    "cc by 4.0",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def license_is_acceptable(value: str) -> bool:
    normalized = " ".join(value.strip().lower().replace("-", " ").split())
    return normalized in ACCEPTED_LICENSES


def batch_dir(batch_id: str) -> Path:
    if not batch_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in batch_id):
        raise ValueError("batch_id may contain only letters, numbers, '_' and '-'")
    return WORK_DIR / batch_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def create_contact_sheet(paths: list[Path], output: Path, sample_size: int = 36) -> None:
    if not paths:
        return
    selected = paths[:sample_size]
    cell_w, cell_h, label_h, columns = 220, 180, 24, 6
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_w - 8, cell_h - 8), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + (cell_h - image.height) // 2))
        draw.text((x + 4, y + cell_h + 3), path.stem[:30], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def import_loc(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    source_manifest = source_root / "manifest.jsonl"
    source_images = source_root / "images"
    target = batch_dir(args.batch_id)
    manifest_path = target / "01_rights_ok" / "manifest.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Batch already exists: {target}. Use --force only to rebuild it.")

    records = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_images = target / "01_rights_ok" / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for record in records:
        source_path = source_images / record["filename"]
        if not source_path.is_file():
            missing.append(record["filename"])
            continue
        item_id = f"loc_{record['id']}"
        destination = output_images / source_path.name
        shutil.copy2(source_path, destination)
        license_value = record.get("license", "")
        rows.append({
            "id": item_id,
            "source_id": item_id,
            "filename": source_path.name,
            "status": "RIGHTS_OK" if license_is_acceptable(license_value) else "NEW",
            "dataset_kind": "real_damaged",
            "provider": record.get("provider", "Library of Congress"),
            "source_url": record.get("item_url", ""),
            "author": "; ".join(record.get("contributors", [])),
            "license": license_value,
            "rights_url": record.get("rights_url", ""),
            "downloaded_at": record.get("downloaded_at", ""),
            "sha256": record.get("sha256") or sha256(source_path),
            "width": record.get("width", ""),
            "height": record.get("height", ""),
            "content_type": record.get("content_type", ""),
            "color_mode": record.get("mode", ""),
            "quality_label": "existing_damage",
            "split": "",
            "annotator": args.operator,
            "reviewer": "",
            "notes": record.get("title", ""),
        })

    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    write_json(target / "01_rights_ok" / "intake_report.json", {
        "batch_id": args.batch_id,
        "operator": args.operator,
        "created_at": utc_now(),
        "source_manifest": str(source_manifest),
        "imported": len(rows),
        "missing": missing,
        "license_check": "RIGHTS_OK is assigned only when license is in the project allowlist.",
    })
    status_counts = Counter(str(row["status"]) for row in rows)
    print(
        f"A intake complete: {status_counts['RIGHTS_OK']} RIGHTS_OK, "
        f"{status_counts['NEW']} NEW, {len(missing)} missing"
    )


def clean_batch(args: argparse.Namespace) -> None:
    target = batch_dir(args.batch_id)
    input_root = target / "01_rights_ok"
    manifest = read_csv(input_root / "manifest.csv")
    output_root = target / "02_cleaned"
    accepted_dir = output_root / "accepted"
    rejected_dir = output_root / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}
    result: list[dict[str, object]] = []
    reject_reasons: Counter[str] = Counter()
    for row in manifest:
        source = input_root / "images" / row["filename"]
        reason = ""
        try:
            with Image.open(source) as opened:
                opened.verify()
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                file_hash = sha256(source)
                if min(width, height) < args.min_short_side:
                    reason = "low_resolution"
                elif file_hash in seen_hashes:
                    reason = f"duplicate_of:{seen_hashes[file_hash]}"
                if reason:
                    shutil.copy2(source, rejected_dir / source.name)
                else:
                    image.save(accepted_dir / source.name, quality=95, subsampling=0)
                    seen_hashes[file_hash] = row["source_id"]
        except Exception as exc:  # Pillow reports format-specific decoding errors.
            width = height = 0
            reason = f"decode_error:{type(exc).__name__}"
            if source.is_file():
                shutil.copy2(source, rejected_dir / source.name)

        updated = dict(row)
        updated.update({
            "status": "REJECTED" if reason else "CLEANED",
            "width": width,
            "height": height,
            "color_mode": "RGB" if not reason else row.get("color_mode", ""),
            "quality_label": reason or "existing_damage",
            "annotator": args.operator,
        })
        result.append(updated)
        if reason:
            reject_reasons[reason.split(":", 1)[0]] += 1

    accepted = [row for row in result if row["status"] == "CLEANED"]
    write_csv(output_root / "manifest.csv", result, MANIFEST_FIELDS)
    create_contact_sheet([accepted_dir / str(row["filename"]) for row in accepted], output_root / "contact_sheet.jpg")
    write_json(output_root / "cleaning_report.json", {
        "batch_id": args.batch_id,
        "operator": args.operator,
        "created_at": utc_now(),
        "minimum_short_side": args.min_short_side,
        "input": len(manifest),
        "accepted": len(accepted),
        "rejected": len(result) - len(accepted),
        "reject_reasons": dict(reject_reasons),
        "manual_checks_still_required": ["watermark", "bad crop", "content category", "damage severity"],
    })
    print(f"B cleaning complete: {len(accepted)} CLEANED, {len(result) - len(accepted)} REJECTED")


def prepare_annotations(args: argparse.Namespace) -> None:
    target = batch_dir(args.batch_id)
    cleaned = read_csv(target / "02_cleaned" / "manifest.csv")
    accepted = [row for row in cleaned if row["status"] == "CLEANED"]
    rows: list[dict[str, str]] = []
    for source in accepted:
        row = {field: "" for field in DAMAGE_FIELDS}
        row.update({
            "source_id": source["source_id"],
            "uncertain": "0",
            "annotator": args.annotator,
            "reviewer": "",
            "status": "PENDING_MANUAL",
        })
        rows.append(row)
    output = target / "03_annotation" / "damage_annotations.csv"
    write_csv(output, rows, DAMAGE_FIELDS)
    write_json(target / "03_annotation" / "annotation_task.json", {
        "batch_id": args.batch_id,
        "created_at": utc_now(),
        "annotator": args.annotator,
        "count": len(rows),
        "image_dir": str(target / "02_cleaned" / "accepted"),
        "rules": "Each damage score must be an integer from 0 to 3. Use uncertain=1 when visual evidence is insufficient.",
    })
    print(f"C task prepared: {len(rows)} rows PENDING_MANUAL")


def validate_annotations(args: argparse.Namespace) -> None:
    target = batch_dir(args.batch_id)
    path = target / "03_annotation" / "damage_annotations.csv"
    rows = read_csv(path)
    errors: list[str] = []
    completed = 0
    for line_number, row in enumerate(rows, start=2):
        if row["status"] not in {"ANNOTATED", "REVIEWED"}:
            errors.append(f"line {line_number}: status is {row['status'] or 'blank'}")
            continue
        if not row["annotator"].strip():
            errors.append(f"line {line_number}: annotator is blank")
        for label in DAMAGE_LABELS:
            if row[label] not in {"0", "1", "2", "3"}:
                errors.append(f"line {line_number}: {label} must be 0, 1, 2 or 3")
        if not any(row[label] in {"1", "2", "3"} for label in DAMAGE_LABELS):
            errors.append(f"line {line_number}: no visible damage was marked")
        completed += 1
    report = {
        "batch_id": args.batch_id,
        "checked_at": utc_now(),
        "rows": len(rows),
        "completed_rows": completed,
        "valid": not errors,
        "errors": errors[:100],
    }
    write_json(target / "04_qa" / "annotation_validation.json", report)
    if errors:
        print(f"D validation blocked: {len(errors)} issue(s). See 04_qa/annotation_validation.json")
        raise SystemExit(2)
    print(f"D annotation validation passed: {len(rows)} rows")


def show_status(args: argparse.Namespace) -> None:
    target = batch_dir(args.batch_id)
    stages = {
        "A_RIGHTS": target / "01_rights_ok" / "manifest.csv",
        "B_CLEAN": target / "02_cleaned" / "manifest.csv",
        "C_ANNOTATE": target / "03_annotation" / "damage_annotations.csv",
        "D_QA": target / "04_qa" / "annotation_validation.json",
    }
    print(f"Batch: {args.batch_id}")
    for name, path in stages.items():
        if not path.exists():
            print(f"  {name:<12} NOT_STARTED")
            continue
        if path.suffix == ".csv":
            rows = read_csv(path)
            counts = Counter(row.get("status", "") for row in rows)
            print(f"  {name:<12} {dict(counts)}")
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
            print(f"  {name:<12} {'PASS' if report.get('valid') else 'BLOCKED'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traceable four-person dataset workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser("import-loc", help="A: import the license-filtered LOC collection")
    intake.add_argument("--batch-id", required=True)
    intake.add_argument("--operator", required=True)
    intake.add_argument("--source-root", default=str(DATA_DIR / "raw" / "library_of_congress" / "daguerreotypes"))
    intake.add_argument("--force", action="store_true")
    intake.set_defaults(func=import_loc)

    clean = subparsers.add_parser("clean", help="B: decode, normalize, deduplicate and screen resolution")
    clean.add_argument("--batch-id", required=True)
    clean.add_argument("--operator", required=True)
    clean.add_argument("--min-short-side", type=int, default=768)
    clean.set_defaults(func=clean_batch)

    annotate = subparsers.add_parser("prepare-annotations", help="C: create the manual damage-label task")
    annotate.add_argument("--batch-id", required=True)
    annotate.add_argument("--annotator", required=True)
    annotate.set_defaults(func=prepare_annotations)

    validate = subparsers.add_parser("validate-annotations", help="D: validate completed manual labels")
    validate.add_argument("--batch-id", required=True)
    validate.set_defaults(func=validate_annotations)

    status = subparsers.add_parser("status", help="Show the state of one batch")
    status.add_argument("--batch-id", required=True)
    status.set_defaults(func=show_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

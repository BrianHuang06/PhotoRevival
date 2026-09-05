"""Prune listed dataset images whose exported size was not changed.

The source manifest stores the original source dimensions. A listed image is
kept when its current package image dimensions differ from those dimensions;
listed images with unchanged dimensions are removed. The remaining package
images are renumbered and all result indexes are rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_PATH_RE = re.compile(r'"([A-Z]:\\[^"\r\n]+)"')
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8")
    return list(dict.fromkeys(Path(value) for value in IMAGE_PATH_RE.findall(text)))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def _safe_package_relative(path: Path, package_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(package_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Listed path is outside package: {path}") from exc
    return relative.as_posix()


def _numeric_seq(row: dict[str, str]) -> int:
    try:
        return int(row.get("dataset_seq", "0"))
    except ValueError:
        return 0


def _renumber_images(
    package_dir: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = sorted(rows, key=_numeric_seq)
    kept: list[dict[str, str]] = []
    class_sequences: Counter[str] = Counter()

    with tempfile.TemporaryDirectory(prefix=".renumber_", dir=package_dir) as staging_name:
        staging = Path(staging_name)
        for row in rows:
            old_path = package_dir / row["export_path"]
            if not old_path.is_file():
                raise FileNotFoundError(f"Dataset image is missing: {old_path}")
            staged_path = staging / row["dataset_class"] / row["export_filename"]
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(staged_path))
            row["_staged_path"] = str(staged_path)

        for dataset_seq, row in enumerate(rows, start=1):
            dataset_class = row["dataset_class"]
            class_sequences[dataset_class] += 1
            class_seq = class_sequences[dataset_class]
            suffix = Path(row["export_filename"]).suffix.lower() or ".png"
            export_filename = f"{dataset_class}_{class_seq:06d}{suffix}"
            export_relative = (
                Path("dataset_images") / dataset_class / export_filename
            )
            staged_path = Path(row.pop("_staged_path"))
            destination = package_dir / export_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_path), str(destination))
            width, height = _image_size(destination)

            row["dataset_seq"] = str(dataset_seq)
            row["class_seq"] = str(class_seq)
            row["dataset_id"] = f"{dataset_class}_{class_seq:06d}"
            row["export_filename"] = export_filename
            row["export_path"] = export_relative.as_posix()
            row["width"] = str(width)
            row["height"] = str(height)
            kept.append(row)

    return kept


def prune_dataset(package_dir: Path, paths_file: Path) -> dict[str, object]:
    package_dir = package_dir.resolve()
    listed_paths = _parse_paths(paths_file)
    if not listed_paths:
        raise ValueError(f"No quoted image paths found in {paths_file}")

    index_path = package_dir / "indexes" / "dataset_index.csv"
    source_manifest_path = package_dir / "provenance" / "source_manifest.csv"
    labels_path = package_dir / "labels" / "damage_labels.csv"
    dataset_labels_path = package_dir / "labels" / "dataset_labels.csv"
    index_rows = _read_csv(index_path)
    source_rows = _read_csv(source_manifest_path)
    damage_rows = _read_csv(labels_path)
    label_rows = _read_csv(dataset_labels_path)
    source_by_id = {row["source_id"]: row for row in source_rows}
    index_by_path = {row["export_path"]: row for row in index_rows}

    remove_ids: set[str] = set()
    kept_listed: list[str] = []
    for listed_path in listed_paths:
        relative = _safe_package_relative(listed_path, package_dir)
        row = index_by_path.get(relative)
        if row is None:
            raise KeyError(f"Listed path is not in dataset_index.csv: {listed_path}")
        image_path = package_dir / row["export_path"]
        current_size = _image_size(image_path)
        source = source_by_id.get(row["source_id"])
        if source is None:
            raise KeyError(f"Source row is missing: {row['source_id']}")
        source_size = (int(source["width"]), int(source["height"]))
        if current_size == source_size:
            remove_ids.add(row["source_id"])
        else:
            kept_listed.append(row["source_id"])

    remaining_rows = [
        row for row in index_rows if row["source_id"] not in remove_ids
    ]
    removed_rows = [
        row for row in index_rows if row["source_id"] in remove_ids
    ]
    for row in removed_rows:
        path = package_dir / row["export_path"]
        if path.is_file():
            path.unlink()

    renumbered_rows = _renumber_images(package_dir, remaining_rows)

    index_fields = list(_read_header(index_path))
    _write_csv(index_path, renumbered_rows, index_fields)
    _write_csv(
        package_dir / "indexes" / "damaged_index.csv",
        [row for row in renumbered_rows if row["dataset_class"] == "damaged_obvious"],
        list(_read_header(package_dir / "indexes" / "damaged_index.csv")),
    )
    _write_csv(
        package_dir / "indexes" / "clean_index.csv",
        [row for row in renumbered_rows if row["dataset_class"] == "clean_near"],
        list(_read_header(package_dir / "indexes" / "clean_index.csv")),
    )

    labels_by_id = {row["dataset_id"]: row for row in label_rows}
    updated_labels: list[dict[str, str]] = []
    for row in renumbered_rows:
        old_id = next(
            (
                candidate["dataset_id"]
                for candidate in label_rows
                if candidate["source_id"] == row["source_id"]
            ),
            "",
        )
        label = dict(labels_by_id[old_id])
        label["dataset_id"] = row["dataset_id"]
        label["dataset_class"] = row["dataset_class"]
        label["class_seq"] = row["class_seq"]
        updated_labels.append(label)
    _write_csv(dataset_labels_path, updated_labels, list(_read_header(dataset_labels_path)))

    filtered_sources = [
        row for row in source_rows if row["source_id"] not in remove_ids
    ]
    filtered_labels = [
        row for row in damage_rows if row["source_id"] not in remove_ids
    ]
    _write_csv(
        source_manifest_path,
        filtered_sources,
        list(_read_header(source_manifest_path)),
    )
    _write_csv(
        labels_path,
        filtered_labels,
        list(_read_header(labels_path)),
    )

    return {
        "listed_unique": len(listed_paths),
        "listed_kept_after_size_check": len(kept_listed),
        "removed": len(remove_ids),
        "removed_source_ids": sorted(remove_ids),
        "dataset_rows": len(renumbered_rows),
        "class_counts": dict(Counter(row["dataset_class"] for row in renumbered_rows)),
    }


def finalize_existing_package(package_dir: Path) -> dict[str, object]:
    """Finish metadata updates after an interrupted prune operation."""
    package_dir = package_dir.resolve()
    index_path = package_dir / "indexes" / "dataset_index.csv"
    source_manifest_path = package_dir / "provenance" / "source_manifest.csv"
    damage_labels_path = package_dir / "labels" / "damage_labels.csv"
    dataset_labels_path = package_dir / "labels" / "dataset_labels.csv"

    index_rows = _read_csv(index_path)
    source_rows = _read_csv(source_manifest_path)
    damage_rows = _read_csv(damage_labels_path)
    dataset_label_rows = _read_csv(dataset_labels_path)
    valid_ids = {row["source_id"] for row in index_rows}
    removed_ids = {
        row["source_id"]
        for row in source_rows
        if row.get("quality_label") == "existing_damage"
        and row["source_id"] not in valid_ids
    }

    source_rows = [row for row in source_rows if row["source_id"] not in removed_ids]
    damage_rows = [row for row in damage_rows if row["source_id"] not in removed_ids]
    _write_csv(
        source_manifest_path,
        source_rows,
        list(_read_header(source_manifest_path)),
    )
    _write_csv(
        damage_labels_path,
        damage_rows,
        list(_read_header(damage_labels_path)),
    )

    labels_by_source = {
        row["source_id"]: row for row in dataset_label_rows
    }
    updated_labels: list[dict[str, str]] = []
    for row in index_rows:
        label = dict(labels_by_source[row["source_id"]])
        label["dataset_id"] = row["dataset_id"]
        label["dataset_class"] = row["dataset_class"]
        label["class_seq"] = row["class_seq"]
        updated_labels.append(label)
    _write_csv(
        dataset_labels_path,
        updated_labels,
        list(_read_header(dataset_labels_path)),
    )

    summary_path = package_dir / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_rows"] = len(source_rows)
    summary["exported_rows"] = len(index_rows)
    summary["class_counts"] = dict(
        Counter(row["dataset_class"] for row in index_rows)
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "removed_source_rows": len(removed_ids),
        "source_rows": len(source_rows),
        "damage_label_rows": len(damage_rows),
        "dataset_rows": len(index_rows),
        "class_counts": summary["class_counts"],
    }


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle).__next__())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove listed images whose package dimensions were unchanged"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--paths-file", type=Path, required=True)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Finish metadata after an interrupted prune operation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = (
        finalize_existing_package(args.package_dir)
        if args.finalize_existing
        else prune_dataset(args.package_dir, args.paths_file)
    )
    print(result)


if __name__ == "__main__":
    main()

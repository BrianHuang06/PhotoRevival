"""Export the final two-class source dataset for D handoff.

The export excludes source rows marked ``quality_label=clean_target_ok``.
Remaining source images are split by the current C labels: at least one
positive damage label becomes ``damaged_obvious``; zero positive labels
becomes ``clean_near``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from scripts.synthetic.source_pipeline import DAMAGE_TYPES


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _positive_damage_count(row: dict[str, str]) -> int:
    return sum(
        1
        for name in DAMAGE_TYPES
        if str(row.get(f"damage_{name}", "")).strip() == "1"
    )


def _source_path(images_dir: Path, row: dict[str, str]) -> Path:
    filename = (row.get("filename") or "").strip()
    if not filename:
        raise ValueError(f"Source row has no filename: {row.get('source_id', '')}")
    path = images_dir / filename
    if path.is_file():
        return path
    source_path = (row.get("source_path") or "").replace("/", "\\")
    candidate = images_dir.parent / source_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Source image not found for {row.get('source_id', '')}: {path}")


def export_dataset(
    images_dir: Path,
    source_manifest: Path,
    damage_labels: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Copy the final two-class dataset and write all handoff indexes."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    source_rows = _read_csv(Path(source_manifest))
    label_rows = _read_csv(Path(damage_labels))
    labels_by_id = {row.get("source_id", ""): row for row in label_rows}

    index_fields = [
        "dataset_seq",
        "dataset_id",
        "dataset_class",
        "class_seq",
        "source_id",
        "original_filename",
        "export_filename",
        "export_path",
        "source_sha256",
        "width",
        "height",
        "content_type",
        "color_mode",
        "quality_label",
        "split",
        "dataset_kind",
        "provider",
        "source_url",
        "license",
        "rights_url",
        "damage_label_list",
        "damage_label_count",
        "damage_label_method",
        "texture_match_label_list",
        "texture_match_scores",
        "texture_match_files",
        *[f"damage_{name}" for name in DAMAGE_TYPES],
    ]

    exported: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    class_sequences: Counter[str] = Counter()
    missing_labels: list[str] = []
    missing_images: list[str] = []

    for source in source_rows:
        source_id = source.get("source_id", "")
        label = labels_by_id.get(source_id)
        if label is None:
            missing_labels.append(source_id)
            continue

        if source.get("quality_label") == "clean_target_ok":
            continue

        damage_count = _positive_damage_count(label)
        dataset_class = "damaged_obvious" if damage_count else "clean_near"
        class_sequences[dataset_class] += 1
        class_counts[dataset_class] += 1
        class_seq = class_sequences[dataset_class]
        dataset_seq = len(exported) + 1
        source_file = _source_path(images_dir, source)
        if not source_file.is_file():
            missing_images.append(source_id)
            continue

        suffix = source_file.suffix.lower() or ".png"
        export_filename = f"{dataset_class}_{class_seq:06d}{suffix}"
        export_relative = Path("dataset_images") / dataset_class / export_filename
        destination = output_dir / export_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)

        row: dict[str, object] = {
            "dataset_seq": dataset_seq,
            "dataset_id": f"{dataset_class}_{class_seq:06d}",
            "dataset_class": dataset_class,
            "class_seq": class_seq,
            "source_id": source_id,
            "original_filename": source.get("filename", ""),
            "export_filename": export_filename,
            "export_path": export_relative.as_posix(),
            "source_sha256": source.get("sha256", ""),
            "width": source.get("width", ""),
            "height": source.get("height", ""),
            "content_type": source.get("content_type", ""),
            "color_mode": source.get("color_mode", label.get("color_mode", "")),
            "quality_label": source.get("quality_label", ""),
            "split": source.get("split", ""),
            "dataset_kind": source.get("dataset_kind", ""),
            "provider": source.get("provider", ""),
            "source_url": source.get("source_url", ""),
            "license": source.get("license", ""),
            "rights_url": source.get("rights_url", ""),
            "damage_label_list": label.get("damage_label_list", ""),
            "damage_label_count": damage_count,
            "damage_label_method": label.get("damage_label_method", ""),
            "texture_match_label_list": label.get("texture_match_label_list", ""),
            "texture_match_scores": label.get("texture_match_scores", ""),
            "texture_match_files": label.get("texture_match_files", ""),
        }
        for name in DAMAGE_TYPES:
            row[f"damage_{name}"] = label.get(f"damage_{name}", "0")
            row[f"confidence_{name}"] = label.get(f"confidence_{name}", "")
            row[f"evidence_{name}"] = label.get(f"evidence_{name}", "")
        exported.append(row)

    if missing_labels:
        raise ValueError(f"Missing damage labels for {len(missing_labels)} source rows")
    if missing_images:
        raise FileNotFoundError(f"Missing source images for {len(missing_images)} source rows")

    indexes_dir = output_dir / "indexes"
    labels_dir = output_dir / "labels"
    metadata_dir = output_dir / "metadata"
    provenance_dir = output_dir / "provenance"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(indexes_dir / "dataset_index.csv", exported, index_fields)
    _write_csv(
        indexes_dir / "damaged_index.csv",
        [row for row in exported if row["dataset_class"] == "damaged_obvious"],
        index_fields,
    )
    _write_csv(
        indexes_dir / "clean_index.csv",
        [row for row in exported if row["dataset_class"] == "clean_near"],
        index_fields,
    )
    label_fields = [
        "dataset_id",
        "dataset_class",
        "class_seq",
        "source_id",
        "original_filename",
        "color_mode",
        "damage_label_list",
        "damage_label_count",
        "damage_label_method",
        "texture_match_label_list",
        "texture_match_scores",
        "texture_match_files",
        *[f"damage_{name}" for name in DAMAGE_TYPES],
        *[f"confidence_{name}" for name in DAMAGE_TYPES],
        *[f"evidence_{name}" for name in DAMAGE_TYPES],
    ]
    _write_csv(
        labels_dir / "dataset_labels.csv",
        exported,
        label_fields,
    )
    exported_ids = {str(row["source_id"]) for row in exported}
    current_source_rows = [
        row for row in source_rows if row.get("source_id", "") in exported_ids
    ]
    if current_source_rows:
        source_fields = list(current_source_rows[0])
        _write_csv(
            provenance_dir / "source_manifest.csv",
            current_source_rows,
            source_fields,
        )

    summary = {
        "dataset_rows": len(exported),
        "class_counts": dict(class_counts),
        "image_dir": "dataset_images",
        "index_files": [
            "indexes/dataset_index.csv",
            "indexes/damaged_index.csv",
            "indexes/clean_index.csv",
        ],
        "label_files": ["labels/dataset_labels.csv"],
        "provenance_files": ["provenance/source_manifest.csv"],
        "damage_types": list(DAMAGE_TYPES),
    }
    (metadata_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the final two-class source dataset")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--damage-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_dataset(
        args.images,
        args.source_manifest,
        args.damage_labels,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

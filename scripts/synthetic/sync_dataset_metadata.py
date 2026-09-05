"""Synchronize final dataset indexes with the image files on disk.

The final image directory is authoritative for output dimensions and hashes.
Source manifests keep source dimensions because they describe the original
downloaded files, not the final handoff images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


INDEX_NAMES = (
    "indexes/dataset_index.csv",
    "indexes/damaged_index.csv",
    "indexes/clean_index.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path.replace("/", "\\")).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Index path escapes package root: {relative_path}") from exc
    return candidate


def sync_dataset_metadata(package_root: Path) -> dict[str, object]:
    package_root = Path(package_root).resolve()
    all_rows: dict[str, dict[str, str]] = {}
    changed_rows = 0
    missing_files: list[str] = []
    for index_name in INDEX_NAMES:
        index_path = package_root / Path(index_name)
        rows = _read_csv(index_path)
        if not rows:
            raise ValueError(f"Index is empty: {index_path}")
        fields = list(rows[0])
        for extra in ("export_sha256", "file_size_bytes", "actual_format"):
            if extra not in fields:
                fields.append(extra)
        for row in rows:
            path = _resolve_inside(package_root, row["export_path"])
            if not path.is_file():
                missing_files.append(row["export_path"])
                continue
            with Image.open(path) as image:
                width, height = image.size
                actual_format = (image.format or path.suffix.lstrip(".")).lower()
            output_hash = _sha256(path)
            updates = {
                "width": str(width),
                "height": str(height),
                "export_sha256": output_hash,
                "file_size_bytes": str(path.stat().st_size),
                "actual_format": actual_format,
            }
            if any(row.get(key, "") != value for key, value in updates.items()):
                changed_rows += 1
            row.update(updates)
            all_rows[row["dataset_id"]] = row
        _write_csv(index_path, rows, fields)

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} indexed image files are missing: {missing_files[:3]}"
        )

    return {
        "package_root": str(package_root),
        "index_files": list(INDEX_NAMES),
        "indexed_images": len(all_rows),
        "updated_rows": changed_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize final dataset indexes with image files"
    )
    parser.add_argument("--package-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(sync_dataset_metadata(args.package_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Download license-allowlisted historical photographs from the Library of Congress."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

from photo_revival.paths import RAW_DATA_DIR


USER_AGENT = "PhotoRevivalDatasetBuilder/1.0 (research dataset acquisition)"
COLLECTIONS = {
    "daguerreotypes": {
        "api_url": "https://www.loc.gov/collections/daguerreotypes/",
        "provider": "Library of Congress",
        "collection": "Daguerreotype collection",
        "license": "Public Domain",
        "rights_url": (
            "https://www.loc.gov/collections/daguerreotypes/"
            "about-this-collection/rights-and-access/"
        ),
    }
}
ACCEPTED_LICENSES = {
    "public domain",
    "cc0",
    "cc0 1.0",
    "cc0 1.0 universal",
    "cc by 3.0",
    "cc by 4.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), default="daguerreotypes")
    parser.add_argument("--limit", type=int, default=50, help="Number of accepted images to obtain")
    parser.add_argument("--min-short-side", type=int, default=768)
    parser.add_argument("--query", help="Optional Library of Congress search query")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between requests in seconds")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def fetch_bytes(url: str, timeout: float, retries: int, delay: float) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers.get_content_type()
        except (HTTPError, URLError, TimeoutError, IncompleteRead) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay * (2**attempt))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def fetch_json(url: str, timeout: float, retries: int, delay: float) -> dict[str, Any]:
    payload, _ = fetch_bytes(url, timeout, retries, delay)
    return json.loads(payload.decode("utf-8"))


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def image_area(url: str) -> int:
    fragment = urlsplit(url).fragment
    width = re.search(r"(?:^|&)w=(\d+)", fragment)
    height = re.search(r"(?:^|&)h=(\d+)", fragment)
    return int(width.group(1)) * int(height.group(1)) if width and height else 0


def select_image_url(item: dict[str, Any]) -> str | None:
    urls = [url for url in item.get("image_url", []) if isinstance(url, str)]
    if not urls:
        return None
    return clean_url(max(urls, key=image_area))


def scalar_or_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def item_identifier(item: dict[str, Any]) -> str:
    nested = item.get("item") or {}
    raw = nested.get("id") or item.get("id") or item.get("url") or "unknown"
    raw = str(raw).rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "unknown"


def normalize_license(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").split())


def license_is_acceptable(value: str) -> bool:
    return normalize_license(value) in ACCEPTED_LICENSES


def inspect_image(payload: bytes) -> tuple[str, int, int, str]:
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
        image_format = (image.format or "JPEG").upper()
        mode = image.mode
    return image_format, width, height, mode


def image_extension(image_format: str) -> str:
    return {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}.get(
        image_format, ".img"
    )


def difference_hash(payload: bytes) -> str:
    with Image.open(BytesIO(payload)) as image:
        pixels = list(
            image.convert("L")
            .resize((13, 12), Image.Resampling.LANCZOS)
            .get_flattened_data()
        )
    bits = []
    for row in range(12):
        start = row * 13
        bits.extend(pixels[start + column + 1] > pixels[start + column] for column in range(12))
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return f"{value:036x}"


def read_manifest(path: Path) -> tuple[set[str], set[str]]:
    identifiers: set[str] = set()
    hashes: set[str] = set()
    if not path.exists():
        return identifiers, hashes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifiers.add(record["id"])
        hashes.add(record["sha256"])
    return identifiers, hashes


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_collection_report(output_dir: Path, manifest_path: Path) -> None:
    records = load_manifest(manifest_path)
    if not records:
        return
    short_sides = [min(record["width"], record["height"]) for record in records]
    report = {
        "count": len(records),
        "provider": records[0]["provider"],
        "collection": records[0]["collection"],
        "license": records[0]["license"],
        "rights_url": records[0]["rights_url"],
        "short_side": {
            "min": min(short_sides),
            "median": median(short_sides),
            "max": max(short_sides),
        },
        "exact_duplicate_count": len(records) - len({record["sha256"] for record in records}),
        "same_dhash_group_count": sum(
            count > 1
            for count in {
                value: sum(record["dhash"] == value for record in records)
                for value in {record["dhash"] for record in records}
            }.values()
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = records[:25]
    cell_width, cell_height, label_height, columns = 220, 220, 24, 5
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, (cell_height + label_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(selected):
        path = output_dir / "images" / record["filename"]
        with Image.open(path) as source:
            preview = source.convert("RGB")
        preview.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y))
        draw.text((x + 5, y + cell_height + 5), record["id"], fill="black", font=font)
    sheet.save(output_dir / "contact_sheet.jpg", quality=90)


def api_page_url(config: dict[str, str], page: int, args: argparse.Namespace) -> str:
    parameters: dict[str, str | int] = {
        "fo": "json",
        "c": min(max(args.page_size, 1), 100),
        "sp": page,
        "sb": "title",
    }
    if args.query:
        parameters["q"] = args.query
    return f"{config['api_url']}?{urlencode(parameters)}"


def rejection(identifier: str, reason: str, item_url: str | None = None) -> dict[str, Any]:
    return {
        "id": identifier,
        "reason": reason,
        "item_url": item_url,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def build_record(
    item: dict[str, Any],
    config: dict[str, str],
    identifier: str,
    filename: str,
    image_url: str,
    sha256: str,
    dhash: str,
    width: int,
    height: int,
    mode: str,
) -> dict[str, Any]:
    nested = item.get("item") or {}
    return {
        "id": identifier,
        "filename": filename,
        "provider": config["provider"],
        "collection": config["collection"],
        "title": item.get("title") or nested.get("title"),
        "date": item.get("date") or nested.get("date"),
        "contributors": scalar_or_list(item.get("contributor") or nested.get("contributors")),
        "subjects": scalar_or_list(item.get("subject") or nested.get("subjects")),
        "description": scalar_or_list(item.get("description") or nested.get("medium")),
        "item_url": item.get("url") or item.get("id"),
        "image_url": image_url,
        "license": config["license"],
        "rights_statement": nested.get("rights_information") or nested.get("rights_advisory"),
        "rights_url": config["rights_url"],
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": sha256,
        "dhash": dhash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    if args.limit < 1 or args.min_short_side < 1 or args.retries < 1:
        raise ValueError("limit, min-short-side, and retries must be positive")

    config = COLLECTIONS[args.collection]
    output_dir = args.output_dir or RAW_DATA_DIR / "library_of_congress" / args.collection
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    existing_ids, existing_hashes = read_manifest(manifest_path)
    initial_count = len(existing_ids)
    target_total = initial_count + args.limit
    page = 1
    accepted_this_run = 0
    inspected = 0

    print(f"Output: {output_dir}", flush=True)
    print(f"Existing: {initial_count}; requesting {args.limit} additional images", flush=True)
    while len(existing_ids) < target_total:
        page_url = api_page_url(config, page, args)
        response = fetch_json(page_url, args.timeout, args.retries, args.delay)
        results = response.get("results") or []
        if not results:
            break

        for item in results:
            if len(existing_ids) >= target_total:
                break
            identifier = item_identifier(item)
            if identifier in existing_ids:
                continue
            inspected += 1
            item_url = item.get("url") or item.get("id")
            if not license_is_acceptable(config["license"]):
                append_jsonl(rejected_path, rejection(identifier, "license_not_accepted", item_url))
                continue
            if item.get("access_restricted") is True or item.get("unrestricted") is False:
                append_jsonl(rejected_path, rejection(identifier, "access_restricted", item_url))
                continue
            image_url = select_image_url(item)
            if not image_url:
                append_jsonl(rejected_path, rejection(identifier, "no_image_url", item_url))
                continue

            try:
                payload, content_type = fetch_bytes(
                    image_url, args.timeout, args.retries, args.delay
                )
                image_format, width, height, mode = inspect_image(payload)
            except Exception as exc:
                append_jsonl(
                    rejected_path,
                    rejection(identifier, f"download_or_decode_error:{type(exc).__name__}", item_url),
                )
                continue
            if min(width, height) < args.min_short_side:
                append_jsonl(
                    rejected_path,
                    rejection(identifier, f"short_side_{min(width, height)}", item_url),
                )
                continue

            sha256 = hashlib.sha256(payload).hexdigest()
            if sha256 in existing_hashes:
                append_jsonl(rejected_path, rejection(identifier, "exact_duplicate", item_url))
                continue

            extension = image_extension(image_format)
            filename = f"loc_{identifier}{extension}"
            destination = image_dir / filename
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            record = build_record(
                item,
                config,
                identifier,
                filename,
                image_url,
                sha256,
                difference_hash(payload),
                width,
                height,
                mode,
            )
            record["content_type"] = content_type
            append_jsonl(manifest_path, record)
            existing_ids.add(identifier)
            existing_hashes.add(sha256)
            accepted_this_run += 1
            print(
                f"[{accepted_this_run}/{args.limit}] {filename} "
                f"({width}x{height})",
                flush=True,
            )
            time.sleep(args.delay)

        pagination = response.get("pagination") or {}
        if not pagination.get("next"):
            break
        page += 1
        time.sleep(args.delay)

    write_collection_report(output_dir, manifest_path)
    print(
        f"Finished: accepted={accepted_this_run}, inspected={inspected}, "
        f"manifest_total={len(existing_ids)}",
        flush=True,
    )
    if accepted_this_run < args.limit:
        print("The collection was exhausted before reaching the requested limit.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

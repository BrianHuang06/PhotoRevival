"""Generate a degraded image for every clean image in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from photo_revival.age_photo import DEFAULT_TEXTURE_DIR, IMAGE_EXTENSIONS, age_photo
from photo_revival.paths import DATASETS_DIR

DEFAULT_CLEAN_DIR = DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean"
DEFAULT_OUTPUT_DIR = (
    DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--textures", type=Path, default=DEFAULT_TEXTURE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    clean_files = sorted(
        path for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(clean_files)} clean photos")

    processed = 0
    for index, input_path in enumerate(clean_files, start=1):
        output_path = args.output / input_path.name
        if output_path.exists() and not args.overwrite:
            continue

        image_seed = None if args.seed is None else args.seed + index
        textures = age_photo(
            input_path,
            output_path,
            texture_dir=args.textures,
            seed=image_seed,
        )
        processed += 1
        names = ", ".join(path.name for path in textures)
        print(f"[{index}/{len(clean_files)}] {input_path.name} <- {names}")

    print(f"Done: generated {processed} photos in {args.output}")


if __name__ == "__main__":
    main()

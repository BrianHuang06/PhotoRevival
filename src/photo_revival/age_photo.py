"""Synthetic aging utilities for building paired restoration datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from photo_revival.paths import ARTIFACTS_DIR, TEXTURES_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_TEXTURE_DIR = (
    TEXTURES_DIR
    / "Resource-Boy-Grunge-Textures"
    / "Resource Boy - Grunge Textures"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "generated" / "aged_photos"


def list_texture_paths(texture_dir: Path = DEFAULT_TEXTURE_DIR) -> list[Path]:
    """Return all supported texture images in a directory."""
    texture_dir = Path(texture_dir)
    if not texture_dir.is_dir():
        raise FileNotFoundError(f"Texture directory not found: {texture_dir}")

    paths = sorted(
        path for path in texture_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No texture images found in: {texture_dir}")
    return paths


def choose_textures(
    texture_dir: Path = DEFAULT_TEXTURE_DIR,
    count: int = 2,
    rng: random.Random | None = None,
) -> list[Path]:
    """Choose one or more texture files for a synthetic aging pass."""
    if count < 1:
        raise ValueError("Texture count must be at least 1")

    paths = list_texture_paths(texture_dir)
    chooser = rng or random
    return chooser.sample(paths, min(count, len(paths)))


def load_texture_mask(
    texture_path: Path,
    target_size: tuple[int, int],
    threshold: int = 25,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Create a damage mask by cropping or tiling a texture image."""
    chooser = rng or random
    texture = Image.open(texture_path).convert("L")
    texture_width, texture_height = texture.size
    target_width, target_height = target_size

    if texture_width >= target_width and texture_height >= target_height:
        max_x = texture_width - target_width
        max_y = texture_height - target_height
        crop_x = chooser.randint(0, max_x) if max_x else 0
        crop_y = chooser.randint(0, max_y) if max_y else 0
        texture = texture.crop(
            (crop_x, crop_y, crop_x + target_width, crop_y + target_height)
        )
    else:
        source = np.asarray(texture)
        repeat_y = (target_height + texture_height - 1) // texture_height
        repeat_x = (target_width + texture_width - 1) // texture_width
        tiled = np.tile(source, (repeat_y, repeat_x))
        texture = Image.fromarray(tiled[:target_height, :target_width])

    values = np.asarray(texture, dtype=np.float32)
    return np.clip(values - threshold, 0, 255) / (255 - threshold)


def apply_texture_damage(
    photo: Image.Image,
    texture_paths: list[Path],
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply scratch, stain, dust, and edge damage from texture masks."""
    width, height = photo.size
    values = np.asarray(photo.convert("RGB"), dtype=np.float32).copy()

    first_mask = load_texture_mask(texture_paths[0], (width, height), 30, rng)
    values += first_mask[:, :, None] * 120

    stain_color = np.array([60, 50, 40], dtype=np.float32)
    stain_weight = first_mask[:, :, None] * 0.2
    values = values * (1 - stain_weight) + stain_color * stain_weight

    if len(texture_paths) > 1:
        second_mask = load_texture_mask(texture_paths[1], (width, height), 35, rng)
        values += second_mask[:, :, None] * 60

        dust_mask = second_mask > 0.5
        values[dust_mask] = np.minimum(values[dust_mask] + 80, 255)

        corner = min(width, height) // 3
        edge_mask = np.zeros((height, width), dtype=np.float32)
        edge_mask[:corner, :corner] = 1
        edge_mask[:corner, -corner:] = 1
        edge_mask[-corner:, :corner] = 1
        edge_mask[-corner:, -corner:] = 1
        values *= 1 - (second_mask * edge_mask * 0.5)[:, :, None]

    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def add_sepia(photo: Image.Image, intensity: float = 0.3) -> Image.Image:
    values = np.asarray(photo, dtype=np.float32)
    matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    sepia = np.clip(np.dot(values, matrix.T), 0, 255)
    result = values * (1 - intensity) + sepia * intensity
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def add_vignette(photo: Image.Image, strength: float = 0.5) -> Image.Image:
    width, height = photo.size
    values = np.asarray(photo, dtype=np.float32).copy()
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = np.clip(1 - (distance / max_distance) ** 2 * strength, 0.2, 1)
    values *= mask[:, :, None]
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def add_grain(
    photo: Image.Image,
    strength: float = 25,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    generator = rng or np.random.default_rng()
    values = np.asarray(photo, dtype=np.float32)
    noise = generator.normal(0, strength, values.shape)
    return Image.fromarray(np.clip(values + noise, 0, 255).astype(np.uint8))


def add_fade(photo: Image.Image, amount: float = 0.15) -> Image.Image:
    values = np.asarray(photo, dtype=np.float32)
    return Image.fromarray(
        np.clip(values * (1 - amount) + 255 * amount, 0, 255).astype(np.uint8)
    )


def age_image(
    photo: Image.Image,
    texture_paths: list[Path],
    seed: int | None = None,
) -> Image.Image:
    """Apply the project's standard synthetic aging pipeline to an image."""
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    result = apply_texture_damage(photo, texture_paths, python_rng)
    result = add_sepia(result, intensity=0.25)
    result = add_fade(result, amount=0.1)
    result = add_vignette(result, strength=0.4)
    return add_grain(result, strength=20, rng=numpy_rng)


def age_photo(
    input_path: Path,
    output_path: Path,
    texture_dir: Path = DEFAULT_TEXTURE_DIR,
    texture_count: int = 2,
    seed: int | None = None,
) -> list[Path]:
    """Age one photo, save it, and return the textures that were used."""
    python_rng = random.Random(seed)
    textures = choose_textures(texture_dir, texture_count, python_rng)
    photo = Image.open(input_path).convert("RGB")
    result = age_image(photo, textures, seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)
    return textures


def create_comparison(
    original_path: Path,
    aged_path: Path,
    output_path: Path,
) -> None:
    """Save an original/aged side-by-side comparison image."""
    original = Image.open(original_path).convert("RGB")
    aged = Image.open(aged_path).convert("RGB")
    width, height = original.size
    aged = aged.resize((width, height), Image.Resampling.LANCZOS)

    comparison = Image.new("RGB", (width * 2 + 20, height + 30), "white")
    comparison.paste(original, (10, 20))
    comparison.paste(aged, (width + 20, 20))
    labels = ImageDraw.Draw(comparison)
    labels.text((10, 2), "Original", fill="black")
    labels.text((width + 20, 2), "Aged", fill="black")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, quality=95)

"""Paths and listing helpers for source texture materials."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
DEFAULT_TEXTURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "textures"
    / "Resource-Boy-Grunge-Textures"
    / "Resource Boy - Grunge Textures"
)


def list_texture_paths(texture_dir: Path = DEFAULT_TEXTURE_DIR) -> list[Path]:
    texture_dir = Path(texture_dir)
    if not texture_dir.is_dir():
        raise FileNotFoundError(f"Texture directory not found: {texture_dir}")
    paths = sorted(
        path
        for path in texture_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No texture images found in: {texture_dir}")
    return paths

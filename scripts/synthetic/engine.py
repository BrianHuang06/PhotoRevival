"""Texture-driven synthetic degradation engine for building paired datasets.

This module generates light/medium/heavy degraded variants of a clean target
image. Degradation is separated into three recorded layers:

- GLOBAL   : fade, color cast, contrast, gamma, vignette, uneven exposure.
- LOCAL    : damage patterns sampled from the project's real grunge texture
             materials (``data/textures/.../Resource Boy - Grunge Textures``),
             plus structural shapes (fold, tear, missing region). All of it is
             composed onto a transparent RGBA damage overlay whose alpha becomes
             the repair mask (0 = clean, 255 = damaged). Masks never come from
             a detector prediction and no texture is invented procedurally.
- IMAGING  : blur, film grain, Poisson noise, scan noise, downscale, JPEG.

Every variant is reproducible: it consumes a single numpy Generator seeded by
the variant seed, and every applied operation (including which texture files
were used and how they were sampled) is recorded in the metadata dict returned
alongside the degraded image and mask.
"""

from __future__ import annotations

import io
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from photo_revival.age_photo import DEFAULT_TEXTURE_DIR, list_texture_paths

try:
    from PIL import __version__ as PILLOW_VERSION
except ImportError:  # pragma: no cover - very old Pillow fallback
    PILLOW_VERSION = "unknown"


# --------------------------------------------------------------------------
# Severity profiles
# --------------------------------------------------------------------------

# ``coverage_target`` is the target local-mask coverage percent window.
# ``texture`` controls how the real texture materials are applied:
#   count       - number of distinct textures sampled
#   opacity     - alpha strength of the damage stamp
#   edge_wear   - how much damage concentrates toward corners/edges (0..1)
#   darken      - how much the texture's own colors are darkened (stain-like)
#   target_cov  - per-texture-layer coverage target (%)
SEVERITY_PROFILE: dict[str, dict] = {
    "light": {
        "global": {
            "fade": {"amount": (0.04, 0.10)},
            "color_cast": {"strength": (0.02, 0.06)},
            "contrast": {"factor": (0.90, 0.97)},
            "gamma": {"gamma": (0.92, 1.06)},
            "vignette": {"strength": (0.10, 0.22)},
            "uneven_exposure": {"strength": (0.04, 0.09)},
        },
        "texture": {
            "count": (1, 1),
            "opacity": (0.40, 0.60),
            "edge_wear": (0.30, 0.50),
            "darken": (0.05, 0.15),
            "target_cov": (2.0, 5.0),
        },
        "structural": {
            "fold": {"count": (0, 1)},
            "tear": {"count": (0, 0)},
            "missing_region": {"count": (0, 0)},
        },
        "imaging": {
            "blur": {"radius": (0.0, 0.6)},
            "grain": {"strength": (3.0, 7.0)},
            "poisson": {"scale": (0.0, 4.0)},
            "scan_noise": {"strength": (2.0, 5.0)},
            "downscale": {"factor": (1.0, 1.0)},
            "jpeg": {"quality": (90, 95)},
        },
        "coverage_target": (1.0, 8.0),
    },
    "medium": {
        "global": {
            "fade": {"amount": (0.10, 0.18)},
            "color_cast": {"strength": (0.06, 0.12)},
            "contrast": {"factor": (0.80, 0.92)},
            "gamma": {"gamma": (0.88, 1.08)},
            "vignette": {"strength": (0.20, 0.36)},
            "uneven_exposure": {"strength": (0.08, 0.16)},
        },
        "texture": {
            "count": (1, 2),
            "opacity": (0.55, 0.75),
            "edge_wear": (0.50, 0.70),
            "darken": (0.10, 0.25),
            "target_cov": (8.0, 14.0),
        },
        "structural": {
            "fold": {"count": (0, 1)},
            "tear": {"count": (0, 1)},
            "missing_region": {"count": (0, 1)},
        },
        "imaging": {
            "blur": {"radius": (0.4, 1.2)},
            "grain": {"strength": (6.0, 12.0)},
            "poisson": {"scale": (3.0, 8.0)},
            "scan_noise": {"strength": (4.0, 9.0)},
            "downscale": {"factor": (1.0, 1.0)},
            "jpeg": {"quality": (80, 90)},
        },
        "coverage_target": (5.0, 20.0),
    },
    "heavy": {
        "global": {
            "fade": {"amount": (0.18, 0.28)},
            "color_cast": {"strength": (0.12, 0.20)},
            "contrast": {"factor": (0.65, 0.82)},
            "gamma": {"gamma": (0.85, 1.10)},
            "vignette": {"strength": (0.32, 0.50)},
            "uneven_exposure": {"strength": (0.12, 0.22)},
        },
        "texture": {
            "count": (2, 3),
            "opacity": (0.70, 0.90),
            "edge_wear": (0.70, 0.90),
            "darken": (0.20, 0.35),
            "target_cov": (15.0, 22.0),
        },
        "structural": {
            "fold": {"count": (1, 2)},
            "tear": {"count": (1, 2)},
            "missing_region": {"count": (1, 2)},
        },
        "imaging": {
            "blur": {"radius": (1.0, 2.0)},
            "grain": {"strength": (10.0, 18.0)},
            "poisson": {"scale": (6.0, 14.0)},
            "scan_noise": {"strength": (7.0, 14.0)},
            "downscale": {"factor": (1.0, 1.2)},
            "jpeg": {"quality": (68, 82)},
        },
        "coverage_target": (15.0, 32.0),
    },
}

# Colors used for the color-cast pass (warm paper, cool scan, greenish, etc.).
CAST_COLORS: list[tuple[int, int, int]] = [
    (250, 244, 220),
    (232, 238, 212),
    (240, 225, 215),
    (225, 232, 240),
]

SEVERITY_ORDER = {"light": 1, "medium": 2, "heavy": 3}

# Structural damage is drawn as shapes (a fold band, an edge tear, a missing
# region). Texture-like damage (scratch/stain/mold/dust/edge wear) comes from
# the real texture materials in ``_texture_damage_layer``, never drawn here.
STRUCTURAL_TYPES = ["fold", "tear", "missing_region"]


# --------------------------------------------------------------------------
# Deterministic RNG helpers
# --------------------------------------------------------------------------

def derive_variant_seed(global_seed: int, source_index: int, severity: str) -> int:
    """Derive a stable per-variant seed from a global seed (splitmix64 avalanche)."""
    key = (
        global_seed * 0x9E3779B97F4A7C15
        + source_index * 0xBF58476D1CE4E5B9
        + SEVERITY_ORDER[severity] * 0x94D049BB133111EB
    ) & 0xFFFFFFFFFFFFFFFF
    key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    key = (key ^ (key >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return key ^ (key >> 31)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Inclusive integer in [lo, hi] using only the numpy Generator."""
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))


def _randfloat(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


# --------------------------------------------------------------------------
# Config and result types
# --------------------------------------------------------------------------

@dataclass
class DegradationConfig:
    min_coverage: float = 1.0
    max_coverage: float = 35.0
    fill_mode: str = "paper"  # "paper" | "blurred_patch"
    texture_dir: Path | None = None  # default: age_photo.DEFAULT_TEXTURE_DIR
    max_adapt_rounds: int = 8
    global_enabled: bool = True
    local_enabled: bool = True
    imaging_enabled: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class DegradedVariant:
    degraded: Image.Image
    mask: Image.Image  # mode "L", values in {0, 255}
    metadata: dict
    coverage: float


# --------------------------------------------------------------------------
# Coverage helpers
# --------------------------------------------------------------------------

def coverage_of(mask: Image.Image | np.ndarray) -> float:
    """Return mask coverage as a percentage of the image."""
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    return 100.0 * float((values > 0).mean())


def binarize_alpha(alpha: np.ndarray) -> Image.Image:
    """Turn an alpha channel into a binary mode-'L' mask (0 / 255)."""
    return Image.fromarray(np.where(alpha > 0, 255, 0).astype(np.uint8), "L")


def _union_elements(
    elements: list[tuple[dict, np.ndarray, np.ndarray | None]],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Union RGBA element layers into combined overlay RGB/alpha and fill RGB/alpha.

    Alpha is the pixelwise maximum (deterministic). RGB is first-draw-wins, so
    overlapping damage keeps the earliest drawn color.
    """
    combined_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    combined_alpha = np.zeros((height, width), dtype=np.uint8)
    fill_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    fill_alpha = np.zeros((height, width), dtype=np.uint8)
    for _, overlay_arr, fill_arr in elements:
        if overlay_arr is not None:
            alpha = overlay_arr[..., 3]
            place = (alpha > 0) & (combined_alpha == 0)
            combined_rgb[place] = overlay_arr[..., :3][place]
            combined_alpha = np.maximum(combined_alpha, alpha)
        if fill_arr is not None:
            fill = fill_arr[..., 3]
            place = (fill > 0) & (fill_alpha == 0)
            fill_rgb[place] = fill_arr[..., :3][place]
            fill_alpha = np.maximum(fill_alpha, fill)
    return combined_rgb, combined_alpha, fill_rgb, fill_alpha


def _union_coverage(
    elements: list[tuple[dict, np.ndarray, np.ndarray | None]],
    width: int,
    height: int,
) -> float:
    alpha = np.zeros((height, width), dtype=np.uint8)
    for _, overlay_arr, _ in elements:
        if overlay_arr is not None:
            alpha = np.maximum(alpha, overlay_arr[..., 3])
    return 100.0 * float((alpha > 0).mean())


def _element_coverage(overlay_arr: np.ndarray) -> float:
    return 100.0 * float((overlay_arr[..., 3] > 0).mean())


def _drop_largest_element(
    elements: list[tuple[dict, np.ndarray, np.ndarray | None]],
) -> list[tuple[dict, np.ndarray, np.ndarray | None]]:
    if len(elements) <= 1:
        return elements
    idx = max(range(len(elements)), key=lambda i: _element_coverage(elements[i][1]))
    return [el for i, el in enumerate(elements) if i != idx]


# --------------------------------------------------------------------------
# Texture-driven local damage
# --------------------------------------------------------------------------

def _fit_texture(texture: Image.Image, rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Crop (or tile) a texture to the target size; returns float32 RGB (H,W,3)."""
    texture_width, texture_height = texture.size
    if texture_width >= width and texture_height >= height:
        x0 = _randint(rng, 0, texture_width - width)
        y0 = _randint(rng, 0, texture_height - height)
        cropped = texture.crop((x0, y0, x0 + width, y0 + height))
        return np.asarray(cropped, dtype=np.float32)
    source = np.asarray(texture.convert("RGB"), dtype=np.float32)
    repeat_y = (height + texture_height - 1) // texture_height
    repeat_x = (width + texture_width - 1) // texture_width
    tiled = np.tile(source, (repeat_y, repeat_x, 1))[:height, :width]
    return tiled


def _texture_damage_layer(
    texture_path: Path,
    rng: np.random.Generator,
    width: int,
    height: int,
    opacity: float,
    target_cov: float,
    edge_wear: float,
    darken: float,
) -> tuple[dict, np.ndarray]:
    """Stamp one real texture onto an RGBA damage layer (alpha becomes the mask).

    A quantile threshold picks which texture pixels count as damage so each
    layer's coverage lands near ``target_cov`` regardless of texture content.
    The texture's own RGB (darkened) provides the visible damage color, and
    ``edge_wear`` boosts damage toward corners/edges like a real old print.
    """
    rgb = _fit_texture(Image.open(texture_path).convert("RGB"), rng, width, height)
    gray = rgb.mean(axis=2)

    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    max_d2 = max(cx * cx + cy * cy, 1.0)
    edge_boost = 1.0 + edge_wear * (d2 / max_d2)  # 1.0 at center -> 1+edge_wear at corners

    quantile = max(0.0, 1.0 - target_cov / 100.0)
    threshold = float(np.quantile(gray, quantile))
    if threshold >= 254.0:
        strength = np.zeros((height, width), dtype=np.float32)
    else:
        strength = np.clip((gray - threshold) / (254.0 - threshold), 0.0, 1.0)

    alpha = np.clip(strength * edge_boost * opacity * 255.0, 0, 255).astype(np.uint8)
    color = np.clip(rgb * (1.0 - darken), 0, 255).astype(np.uint8)
    overlay = np.dstack([color, alpha])

    params = {
        "texture": Path(texture_path).name,
        "opacity": round(opacity, 3),
        "coverage_target": round(target_cov, 2),
        "threshold": round(threshold, 1),
        "edge_wear": round(edge_wear, 3),
        "darken": round(darken, 3),
    }
    return params, overlay


def _sample_texture_paths(
    rng: np.random.Generator, texture_paths: list[Path], count: int
) -> list[Path]:
    """Pick ``count`` distinct textures deterministically (numpy rng only)."""
    pool = list(texture_paths)
    count = max(0, min(count, len(pool)))
    if count == 0:
        return []
    indices = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(index)] for index in indices]


# --------------------------------------------------------------------------
# Structural local damage (shapes, not textures)
# --------------------------------------------------------------------------

def _blank_layer(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def draw_fold(rng: np.random.Generator, width: int, height: int, count: int):
    layer = _blank_layer(width, height)
    draw = ImageDraw.Draw(layer)
    s = min(width, height)
    entries = []
    for _ in range(count):
        horizontal = rng.random() < 0.5
        if horizontal:
            y0 = _randint(rng, int(0.15 * height), int(0.85 * height))
            p1 = (int(0.05 * width), y0)
            p2 = (int(0.95 * width), y0 + _randint(rng, -int(0.1 * s), int(0.1 * s)))
        else:
            x0 = _randint(rng, int(0.15 * width), int(0.85 * width))
            p1 = (x0, int(0.05 * height))
            p2 = (x0 + _randint(rng, -int(0.1 * s), int(0.1 * s)), int(0.95 * height))
        band_width = _randint(rng, int(0.02 * s), int(0.05 * s))
        value = _randint(rng, 120, 200)
        alpha = _randint(rng, 60, 140)
        # 3 overlapping lines give a soft band
        for dw in (band_width, max(1, band_width // 2), 1):
            draw.line([p1, p2], fill=(value, value, value, alpha), width=dw)
        entries.append({"p1": list(p1), "p2": list(p2), "width": band_width, "value": value, "alpha": alpha})
    return {"count": count, "folds": entries}, np.asarray(layer), None


def draw_tear(rng: np.random.Generator, width: int, height: int, count: int):
    overlay = _blank_layer(width, height)
    fill = _blank_layer(width, height)
    od = ImageDraw.Draw(overlay)
    fd = ImageDraw.Draw(fill)
    s = min(width, height)
    placeholder = (255, 0, 255, 255)
    entries = []
    for _ in range(count):
        edge = _randint(rng, 0, 3)  # 0=top, 1=bottom, 2=left, 3=right
        start_frac = _randfloat(rng, 0.1, 0.7)
        length = int(s * _randfloat(rng, 0.2, 0.5))
        depth = int(s * _randfloat(rng, 0.05, 0.15))
        if edge in (0, 1):
            y0 = 0 if edge == 0 else height - 1
            x0 = int(start_frac * width)
            vertices = [(x0, y0)]
            x = x0
            segments = _randint(rng, 3, 5)
            sign = 1 if edge == 0 else -1
            for _ in range(segments):
                x = max(0, min(width - 1, x + _randint(rng, -length // 2, length // 2)))
                y = max(0, min(height - 1, y0 + sign * _randint(rng, 1, depth)))
                vertices.append((x, y))
            vertices.append((max(0, min(width - 1, x0 + _randint(rng, -int(0.1 * s), int(0.1 * s)))), y0))
        else:
            x0 = 0 if edge == 2 else width - 1
            y0 = int(start_frac * height)
            vertices = [(x0, y0)]
            y = y0
            segments = _randint(rng, 3, 5)
            sign = 1 if edge == 2 else -1
            for _ in range(segments):
                y = max(0, min(height - 1, y + _randint(rng, -length // 2, length // 2)))
                x = max(0, min(width - 1, x0 + sign * _randint(rng, 1, depth)))
                vertices.append((x, y))
            vertices.append((x0, max(0, min(height - 1, y0 + _randint(rng, -int(0.1 * s), int(0.1 * s))))))
        fd.polygon(vertices, fill=(255, 255, 255, 255))
        od.polygon(vertices, fill=placeholder)
        entries.append({"edge": edge, "start_frac": round(start_frac, 4), "length": length, "depth": depth, "vertices": vertices})
    return {"count": count, "tears": entries}, np.asarray(overlay), np.asarray(fill)


def draw_missing_region(rng: np.random.Generator, width: int, height: int, count: int):
    overlay = _blank_layer(width, height)
    fill = _blank_layer(width, height)
    od = ImageDraw.Draw(overlay)
    fd = ImageDraw.Draw(fill)
    s = min(width, height)
    placeholder = (255, 0, 255, 255)
    entries = []
    for _ in range(count):
        cx = _randfloat(rng, 0.3, 0.7) * width
        cy = _randfloat(rng, 0.3, 0.7) * height
        radius = _randfloat(rng, 0.06, 0.15) * s
        vertices = []
        n_vertices = _randint(rng, 5, 8)
        for k in range(n_vertices):
            angle = 2.0 * math.pi * k / n_vertices
            rr = radius * _randfloat(rng, 0.6, 1.2)
            x = max(0, min(width - 1, int(cx + rr * math.cos(angle))))
            y = max(0, min(height - 1, int(cy + rr * math.sin(angle))))
            vertices.append((x, y))
        fd.polygon(vertices, fill=(255, 255, 255, 255))
        od.polygon(vertices, fill=placeholder)
        entries.append({"center": [round(cx, 1), round(cy, 1)], "radius": round(radius, 1), "vertices": vertices})
    return {"count": count, "regions": entries}, np.asarray(overlay), np.asarray(fill)


STRUCTURAL_FUNCS = {
    "fold": draw_fold,
    "tear": draw_tear,
    "missing_region": draw_missing_region,
}


# --------------------------------------------------------------------------
# Global degradation
# --------------------------------------------------------------------------

def apply_fade(img: Image.Image, amount: float) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32)
    result = arr * (1.0 - amount) + 255.0 * amount
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"amount": round(amount, 4)}


def apply_color_cast(img: Image.Image, color: tuple[int, int, int], strength: float) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32)
    tint = np.array(color, dtype=np.float32)
    result = arr * (1.0 - strength) + tint * strength
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"color": list(color), "strength": round(strength, 4)}


def apply_contrast(img: Image.Image, factor: float) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32)
    result = (arr - 128.0) * factor + 128.0
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"factor": round(factor, 4)}


def apply_gamma(img: Image.Image, gamma: float) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    result = (arr ** gamma) * 255.0
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"gamma": round(gamma, 4)}


def apply_vignette(img: Image.Image, strength: float) -> tuple[Image.Image, dict]:
    width, height = img.size
    arr = np.asarray(img, dtype=np.float32).copy()
    yy, xx = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    max_distance = np.sqrt(cx**2 + cy**2)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - (distance / max_distance) ** 2 * strength, 0.2, 1.0)
    result = arr * mask[..., None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"strength": round(strength, 4)}


def apply_uneven_exposure(
    img: Image.Image,
    rng: np.random.Generator,
    strength: float,
    axis: int,
    sigma_px: float,
) -> tuple[Image.Image, dict]:
    width, height = img.size
    arr = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    if axis == 0:
        cx = width * _randfloat(rng, 0.1, 0.4)
    elif axis == 1:
        cx = width * _randfloat(rng, 0.6, 0.9)
    else:
        cx = width * _randfloat(rng, 0.2, 0.8)
    cy = height * _randfloat(rng, 0.2, 0.8)
    gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * max(sigma_px, 1.0) ** 2))
    light = 1.0 + strength * (gauss - 0.5) * 2.0
    result = arr * light[..., None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {
        "strength": round(strength, 4),
        "axis": axis,
        "center": [round(cx, 1), round(cy, 1)],
        "sigma_px": round(sigma_px, 1),
    }


# --------------------------------------------------------------------------
# Imaging degradation
# --------------------------------------------------------------------------

def apply_blur(img: Image.Image, radius: float) -> tuple[Image.Image, dict]:
    if radius <= 0:
        return img, {"radius": radius}
    return img.filter(ImageFilter.GaussianBlur(radius)), {"radius": round(radius, 4)}


def apply_grain(img: Image.Image, strength: float, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, strength, arr.shape)
    result = arr + noise
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"strength": round(strength, 4)}


def _poisson(rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    """Sample Poisson counts with Knuth's algorithm from the uniform stream."""
    lam = np.asarray(lam, dtype=np.float64)
    limit = np.exp(-lam)
    result = np.zeros(lam.shape, dtype=np.float64)
    p = np.ones(lam.shape)
    active = np.ones(lam.shape, dtype=bool)
    while active.any():
        u = rng.random(lam.shape)
        p[active] *= u[active]
        finished = p[active] <= limit[active]
        active[active] = ~finished
        result[active] += 1.0
    return result


def apply_poisson(img: Image.Image, scale: float, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    if scale <= 0:
        return img, {"scale": scale}
    arr = np.asarray(img, dtype=np.float32)
    counts = arr / 255.0 * scale
    noisy_counts = _poisson(rng, counts)
    result = noisy_counts / scale * 255.0
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {"scale": round(scale, 4)}


def apply_scan_noise(img: Image.Image, strength: float, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32)
    height, width = arr.shape[:2]
    row_noise = rng.normal(0.0, strength * 0.3, (height, 1, 1))
    # occasional streak bands
    n_streaks = _randint(rng, 0, max(1, height // 200))
    streak_params = []
    for _ in range(n_streaks):
        row = _randint(rng, 0, height - 1)
        band = _randint(rng, 1, 3)
        shift = rng.uniform(strength * 0.5, strength * 2.0)
        for dr in range(band):
            rr = min(height - 1, row + dr)
            arr[rr] += shift
        streak_params.append({"row": row, "band": band, "shift": round(float(shift), 3)})
    # low-frequency mottle from a few gaussian bumps
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    mottle = np.zeros((height, width), dtype=np.float32)
    bump_count = _randint(rng, 2, 5)
    for _ in range(bump_count):
        bx = rng.uniform(0, width)
        by = rng.uniform(0, height)
        sigma = max(10.0, rng.uniform(0.1, 0.3) * max(width, height))
        amp = rng.uniform(-strength * 0.4, strength * 0.4)
        mottle += amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2.0 * sigma**2))
    result = arr + row_noise + mottle[..., None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), {
        "strength": round(strength, 4),
        "streaks": streak_params,
        "bumps": bump_count,
    }


def apply_downscale(img: Image.Image, factor: float, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    del rng  # not needed, kept for a uniform call signature
    if factor <= 1.0:
        return img, {"factor": factor}
    width, height = img.size
    small = img.resize(
        (max(1, int(width / factor)), max(1, int(height / factor))),
        Image.Resampling.LANCZOS,
    )
    up = small.resize((width, height), Image.Resampling.LANCZOS)
    return up, {"factor": round(factor, 4), "resample": "LANCZOS"}


def apply_jpeg(img: Image.Image, quality: int) -> tuple[Image.Image, dict]:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    return result, {"quality": quality, "subsampling": 2}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _build_content_canvas(
    base: Image.Image,
    width: int,
    height: int,
    fill_mode: str,
    rng: np.random.Generator,
) -> Image.Image:
    """Paper-like or blurred local content used to fill tear/missing regions."""
    if fill_mode == "blurred_patch":
        radius = max(8, min(width, height) // 8)
        return base.filter(ImageFilter.GaussianBlur(radius))
    # paper: warm light gray with small deterministic per-pixel jitter
    arr = np.full((height, width, 3), (225, 217, 200), dtype=np.float32)
    jitter = rng.normal(0.0, 6.0, (height, width, 1))
    arr = arr + jitter
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def degrade_target(
    source_image: Image.Image,
    severity: str,
    seed: int,
    cfg: DegradationConfig | None = None,
) -> DegradedVariant:
    """Generate one degraded variant (image + binary mask + metadata)."""
    if severity not in SEVERITY_PROFILE:
        raise ValueError(f"Unknown severity: {severity}")
    cfg = cfg or DegradationConfig()
    rng = np.random.default_rng(seed)
    profile = SEVERITY_PROFILE[severity]
    width, height = source_image.size
    steps: list[dict] = []
    target = source_image.convert("RGB")

    # ---- global pass ---------------------------------------------------
    base = target
    if cfg.global_enabled:
        g = profile["global"]
        fade_amount = _randfloat(rng, *g["fade"]["amount"])
        base, params = apply_fade(base, fade_amount)
        steps.append({"op": "fade", "params": params})

        cast_color = tuple(CAST_COLORS[_randint(rng, 0, len(CAST_COLORS) - 1)])
        cast_strength = _randfloat(rng, *g["color_cast"]["strength"])
        base, params = apply_color_cast(base, cast_color, cast_strength)
        steps.append({"op": "color_cast", "params": params})

        contrast = _randfloat(rng, *g["contrast"]["factor"])
        base, params = apply_contrast(base, contrast)
        steps.append({"op": "contrast", "params": params})

        gamma = _randfloat(rng, *g["gamma"]["gamma"])
        base, params = apply_gamma(base, gamma)
        steps.append({"op": "gamma", "params": params})

        vignette = _randfloat(rng, *g["vignette"]["strength"])
        base, params = apply_vignette(base, vignette)
        steps.append({"op": "vignette", "params": params})

        uneven = _randfloat(rng, *g["uneven_exposure"]["strength"])
        axis = _randint(rng, 0, 2)
        sigma_px = max(10.0, _randfloat(rng, 0.2, 0.5) * min(width, height))
        base, params = apply_uneven_exposure(base, rng, uneven, axis, sigma_px)
        steps.append({"op": "uneven_exposure", "params": params})

    # ---- local damage pass (texture-driven + structural) ---------------
    if cfg.local_enabled:
        texture_dir = Path(cfg.texture_dir) if cfg.texture_dir else DEFAULT_TEXTURE_DIR
        texture_paths = list_texture_paths(texture_dir)
        if not texture_paths:
            raise FileNotFoundError(
                f"No texture materials found in {texture_dir}. "
                "Place the grunge texture files there before generating."
            )

        elements: list[tuple[dict, np.ndarray, np.ndarray | None]] = []
        structural_used: list[str] = []

        # real-texture damage layers
        t = profile["texture"]
        n_textures = _randint(rng, *t["count"])
        sampled = _sample_texture_paths(rng, texture_paths, n_textures)
        opacity = _randfloat(rng, *t["opacity"])
        edge_wear = _randfloat(rng, *t["edge_wear"])
        darken = _randfloat(rng, *t["darken"])
        target_cov = _randfloat(rng, *t["target_cov"])
        for texture_path in sampled:
            params, overlay_arr = _texture_damage_layer(
                texture_path, rng, width, height, opacity, target_cov, edge_wear, darken
            )
            elements.append((params, overlay_arr, None))

        # structural shapes (fold / tear / missing region)
        for damage_type in STRUCTURAL_TYPES:
            count = _randint(rng, *profile["structural"][damage_type]["count"])
            if count > 0:
                params, overlay_arr, fill_arr = STRUCTURAL_FUNCS[damage_type](rng, width, height, count)
                elements.append((params, overlay_arr, fill_arr))
                structural_used.append(damage_type)

        target_lo = max(profile["coverage_target"][0], cfg.min_coverage)
        target_hi = min(profile["coverage_target"][1], cfg.max_coverage)

        # adapt upward until the lower bound is reached (more texture layers)
        rounds = 0
        while _union_coverage(elements, width, height) < target_lo and rounds < cfg.max_adapt_rounds:
            extra = _sample_texture_paths(rng, texture_paths, 1)
            if not extra:
                break
            params, overlay_arr = _texture_damage_layer(
                extra[0], rng, width, height, opacity, target_cov, edge_wear, darken
            )
            elements.append((params, overlay_arr, None))
            rounds += 1

        # trim only when the union exceeds the target ceiling; never drop below the lower bound
        while _union_coverage(elements, width, height) > target_hi and len(elements) > 1:
            candidate = _drop_largest_element(elements)
            if _union_coverage(candidate, width, height) >= target_lo - 0.5:
                elements = candidate
            else:
                break

        combined_rgb, combined_alpha, fill_rgb, fill_alpha = _union_elements(elements, width, height)
        coverage = 100.0 * float((combined_alpha > 0).mean())

        content_canvas = _build_content_canvas(base, width, height, cfg.fill_mode, rng)
        content_rgb = np.asarray(content_canvas, dtype=np.uint8)
        if fill_alpha.any():
            combined_rgb = np.where(fill_alpha[..., None] > 0, content_rgb, combined_rgb)

        overlay_img = Image.fromarray(
            np.dstack([combined_rgb, combined_alpha]).astype(np.uint8),
            "RGBA",
        )
        filled = Image.composite(content_canvas, base, Image.fromarray(fill_alpha, "L"))
        result = Image.alpha_composite(filled.convert("RGBA"), overlay_img).convert("RGB")

        mask = binarize_alpha(combined_alpha)
        steps.append({
            "op": "local_damage",
            "params": {
                "texture_dir": str(texture_dir),
                "textures": [el[0] for el in elements if "texture" in el[0]],
                "structural_types": structural_used,
                "elements": [el[0] for el in elements],
                "coverage": round(coverage, 4),
                "restoration_type": "local_repair",
            },
        })
    else:
        result = base
        mask = Image.new("L", (width, height), 0)
        coverage = 0.0
        steps.append({
            "op": "local_damage",
            "params": {"textures": [], "coverage": 0.0, "restoration_type": "global_restoration"},
        })

    # ---- imaging pass --------------------------------------------------
    final = result
    if cfg.imaging_enabled:
        i = profile["imaging"]
        blur_radius = _randfloat(rng, *i["blur"]["radius"])
        final, params = apply_blur(final, blur_radius)
        steps.append({"op": "blur", "params": params})

        scan = _randfloat(rng, *i["scan_noise"]["strength"])
        final, params = apply_scan_noise(final, scan, rng)
        steps.append({"op": "scan_noise", "params": params})

        grain = _randfloat(rng, *i["grain"]["strength"])
        final, params = apply_grain(final, grain, rng)
        steps.append({"op": "grain", "params": params})

        poisson = _randfloat(rng, *i["poisson"]["scale"])
        final, params = apply_poisson(final, poisson, rng)
        steps.append({"op": "poisson", "params": params})

        downscale = _randfloat(rng, *i["downscale"]["factor"])
        final, params = apply_downscale(final, downscale, rng)
        steps.append({"op": "downscale", "params": params})

        jpeg_quality = _randint(rng, *i["jpeg"]["quality"])
        final, params = apply_jpeg(final, jpeg_quality)
        steps.append({"op": "jpeg", "params": params})

    metadata = {
        "seed": seed,
        "severity": severity,
        "restoration_type": "local_repair" if cfg.local_enabled else "global_restoration",
        "mask_coverage": round(coverage, 4),
        "image_size": [width, height],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "steps": steps,
    }
    return DegradedVariant(degraded=final, mask=mask, metadata=metadata, coverage=coverage)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_variant_files(
    output_root: Path,
    source_id: str,
    severity: str,
    variant: DegradedVariant,
) -> None:
    """Write the degraded image and mask PNGs for one variant."""
    output_root = Path(output_root)
    degraded_path = output_root / "degraded" / f"{source_id}_{severity}.png"
    mask_path = output_root / "masks" / f"{source_id}_{severity}.png"
    degraded_path.parent.mkdir(parents=True, exist_ok=True)
    variant.degraded.save(degraded_path, "PNG")
    variant.mask.save(mask_path, "PNG")


def write_source_metadata(
    output_root: Path,
    source_id: str,
    target_sha256: str,
    variants: dict[str, DegradedVariant],
) -> None:
    """Write a single ``metadata/<source_id>.json`` holding every severity variant.

    Each variant keeps its own seed, restoration type, coverage and full step
    parameters so nothing is lost, while matching the documented single-file
    layout: ``metadata/source_000001.json``.
    """
    output_root = Path(output_root)
    entries: dict[str, dict] = {}
    for severity, variant in variants.items():
        meta = dict(variant.metadata)
        meta["files"] = {
            "degraded": f"degraded/{source_id}_{severity}.png",
            "mask": f"masks/{source_id}_{severity}.png",
        }
        entries[severity] = meta
    doc = {
        "source_id": source_id,
        "target": f"target/{source_id}.png",
        "target_sha256": target_sha256,
        "variants": entries,
    }
    metadata_path = output_root / "metadata" / f"{source_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    # remove any legacy per-variant metadata files for this source
    for legacy in metadata_path.parent.glob(f"{source_id}_*.json"):
        if legacy.name != metadata_path.name:
            legacy.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Collage builder
# --------------------------------------------------------------------------

def build_source_collage(
    target: Path,
    degraded: list[Path],
    masks: list[Path],
    output: Path,
    labels: list[str],
) -> None:
    """2-row collage: row 1 target + degraded variants, row 2 empty + mask overlays."""
    cell_width, cell_height, label_height, pad = 280, 220, 22, 8
    columns = 1 + len(degraded)
    rows = 2
    sheet = Image.new(
        "RGB",
        (columns * cell_width + (columns + 1) * pad, rows * (cell_height + label_height) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    def _paste(image: Image.Image, column: int, row: int, label: str) -> None:
        thumb = ImageOps.contain(image.convert("RGB"), (cell_width, cell_height), Image.Resampling.LANCZOS)
        x = pad + column * (cell_width + pad) + (cell_width - thumb.width) // 2
        y = pad + row * (cell_height + label_height) + (cell_height - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        draw.text((pad + column * (cell_width + pad) + 4, pad + row * (cell_height + label_height) + cell_height + 2), label, fill="black")

    _paste(Image.open(target), 0, 0, "target")
    for col, (degraded_path, mask_path, label) in enumerate(zip(degraded, masks, labels), start=1):
        degraded_img = Image.open(degraded_path).convert("RGB")
        _paste(degraded_img, col, 0, label)
        mask = Image.open(mask_path).convert("L").resize(degraded_img.size, Image.Resampling.NEAREST)
        overlay_red = Image.new("RGB", degraded_img.size, (255, 0, 0))
        mask_overlay = Image.composite(overlay_red, degraded_img, mask)
        _paste(mask_overlay, col, 1, f"{label} mask")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")

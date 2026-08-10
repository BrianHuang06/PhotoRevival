"""Optional SwinIR and BOPBTL preprocessing for LoRA training."""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from photo_revival.paths import THIRD_PARTY_DIR, WEIGHTS_DIR
from photo_revival.training.common import Pair, log

ConditionPair = tuple[Path, Path, str]
TrainingSample = tuple[Path, Path, Path | None, str]


def _load_swinir(weights_path: str | Path | None, device: str):
    source_path = THIRD_PARTY_DIR / "DiffBIR" / "diffbir" / "model" / "swinir.py"
    if not source_path.exists():
        raise FileNotFoundError(f"DiffBIR SwinIR source not found: {source_path}")
    spec = importlib.util.spec_from_file_location("photo_revival_diffbir_swinir", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SwinIR from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6] * 8,
        num_heads=[6] * 8,
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )

    resolved_weights = Path(weights_path) if weights_path else (
        WEIGHTS_DIR / "realesrgan_s4_swinir_100k.pth"
    )
    if not resolved_weights.exists():
        raise FileNotFoundError(f"SwinIR weights not found: {resolved_weights}")
    state = torch.load(resolved_weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def _run_swinir(model, image: Image.Image, resolution: int, device: str) -> Image.Image:
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
        output = model(tensor)
    values = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))


def prepare_condition_images(
    pairs: list[Pair],
    mode: str,
    cache_dir: str | Path,
    weights_path: str | Path | None,
    resolution: int,
    device: str = "cuda",
) -> list[ConditionPair]:
    if mode == "none":
        return pairs.copy()
    if mode != "swinir":
        raise ValueError(f"Unknown preprocessor: {mode}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = [(cache_dir / name, clean_path, name) for _, clean_path, name in pairs]
    pending = [
        (degraded_path, cache_dir / name, name)
        for degraded_path, _, name in pairs
        if not (cache_dir / name).exists()
    ]
    if not pending:
        log(f"Using {len(results)} cached SwinIR images from {cache_dir}")
        return results

    log(f"Generating {len(pending)} SwinIR condition images")
    model = _load_swinir(weights_path, device)
    for input_path, output_path, name in tqdm(pending, desc="SwinIR preprocessing"):
        image = Image.open(input_path).convert("RGB")
        _run_swinir(model, image, resolution, device).save(output_path, quality=95)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _load_bopbtl(device: str):
    project_dir = THIRD_PARTY_DIR / "Bringing-Old-Photos-Back-to-Life" / "Global"
    detection_dir = project_dir / "detection_models"
    for path in (project_dir, detection_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from detection_models import networks

    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )
    checkpoint = project_dir / "checkpoints" / "detection" / "FT_Epoch_latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"BOPBTL checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def _detect_bopbtl(model, image: Image.Image, device: str) -> Image.Image:
    import torchvision.transforms.functional as TF

    original_size = image.size
    tensor = TF.normalize(TF.to_tensor(image.convert("L")), [0.5], [0.5]).unsqueeze(0)
    _, _, height, width = tensor.shape
    scale = 256 / min(height, width)
    scaled_height = max(16, int(round(height * scale / 16) * 16))
    scaled_width = max(16, int(round(width * scale / 16) * 16))
    tensor = F.interpolate(tensor, (scaled_height, scaled_width), mode="bilinear").to(device)
    prediction = torch.sigmoid(model(tensor)).cpu()
    prediction = F.interpolate(prediction, (height, width), mode="nearest")
    mask = (prediction.squeeze().numpy() >= 0.4).astype(np.uint8) * 255
    return Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)


def prepare_masks(
    source_pairs: list[Pair],
    mode: str,
    cache_dir: str | Path,
    device: str = "cuda",
) -> list[Path | None]:
    if mode == "full":
        return [None] * len(source_pairs)
    if mode != "bopbtl":
        raise ValueError(f"Unknown mask mode: {mode}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{Path(name).stem}.png" for _, _, name in source_pairs]
    pending = [
        (degraded_path, mask_path)
        for (degraded_path, _, _), mask_path in zip(source_pairs, paths)
        if not mask_path.exists()
    ]
    if not pending:
        log(f"Using {len(paths)} cached BOPBTL masks from {cache_dir}")
        return paths

    log(f"Generating {len(pending)} BOPBTL masks")
    model = _load_bopbtl(device)
    for input_path, output_path in tqdm(pending, desc="BOPBTL masks"):
        image = Image.open(input_path).convert("RGB")
        _detect_bopbtl(model, image, device).save(output_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return paths


def build_training_samples(
    source_pairs: list[Pair],
    condition_pairs: list[ConditionPair],
    masks: list[Path | None],
) -> list[TrainingSample]:
    if not (len(source_pairs) == len(condition_pairs) == len(masks)):
        raise ValueError("Source, condition, and mask collections must have equal length")
    return [
        (condition_path, clean_path, mask_path, name)
        for (condition_path, clean_path, name), mask_path in zip(condition_pairs, masks)
    ]

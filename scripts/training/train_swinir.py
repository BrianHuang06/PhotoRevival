# -*- coding: utf-8 -*-
"""
Photo Restoration - Real SwinIR Fine-tuning

Uses the official SwinIR with Swin Transformer attention blocks.
Loss: L1 + Perceptual (VGG) for better LPIPS.
"""

import os
import sys
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from photo_revival.paths import CHECKPOINTS_DIR, DATASETS_DIR, LOGS_DIR

CLEAN_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
DEGRADED_DIR = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_Input_Degraded")
OUTPUT_DIR = str(CHECKPOINTS_DIR / "swinir")
LOG_FILE = str(LOGS_DIR / "training_log_swinir.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 128
BATCH_SIZE = 4


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()


class RestorationDataset(Dataset):
    def __init__(self, pairs, patch_size=128, augment=True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dmg_path, clean_path, _ = self.pairs[idx]
        clean = np.array(Image.open(clean_path).convert("RGB"))
        degraded = np.array(Image.open(dmg_path).convert("RGB"))

        h, w = clean.shape[:2]
        ps = self.patch_size

        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            degraded = np.pad(degraded, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = clean.shape[:2]

        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        clean_patch = clean[top:top+ps, left:left+ps]
        dmg_patch = degraded[top:top+ps, left:left+ps]

        if self.augment:
            if np.random.random() < 0.5:
                clean_patch = clean_patch[:, ::-1, :].copy()
                dmg_patch = dmg_patch[:, ::-1, :].copy()
            if np.random.random() < 0.5:
                clean_patch = clean_patch[::-1, :, :].copy()
                dmg_patch = dmg_patch[::-1, :, :].copy()
            if np.random.random() < 0.5:
                k = np.random.choice([1, 2, 3])
                clean_patch = np.rot90(clean_patch, k).copy()
                dmg_patch = np.rot90(dmg_patch, k).copy()

        clean_t = torch.from_numpy(clean_patch.transpose(2, 0, 1)).float() / 255.0
        dmg_t = torch.from_numpy(dmg_patch.transpose(2, 0, 1)).float() / 255.0

        return dmg_t, clean_t


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ]).to(device)
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


def evaluate(model, pairs, device, psnr_fn, ssim_fn, lpips_fn=None, num=10):
    model.eval()
    step = max(1, len(pairs) // num)
    test_pairs = [pairs[i] for i in range(0, len(pairs), step)][:num]

    psnr_before, psnr_after = [], []
    ssim_before, ssim_after = [], []
    lpips_list = []

    with torch.no_grad():
        for idx, (dmg_path, clean_path, fname) in enumerate(test_pairs):
            clean = np.array(Image.open(clean_path).convert("RGB"))
            degraded = np.array(Image.open(dmg_path).convert("RGB"))
            h, w = clean.shape[:2]
            degraded = degraded[:h, :w]

            p_before = psnr_fn(clean, degraded)
            s_before = ssim_fn(clean, degraded, channel_axis=2)

            dmg_t = torch.from_numpy(degraded.transpose(2, 0, 1)).float() / 255.0
            dmg_t = dmg_t.unsqueeze(0).to(device)
            pad_h = (8 - dmg_t.shape[2] % 8) % 8
            pad_w = (8 - dmg_t.shape[3] % 8) % 8
            if pad_h or pad_w:
                dmg_t = F.pad(dmg_t, (0, pad_w, 0, pad_h), mode='reflect')

            restored = model(dmg_t)
            restored = restored[:, :, :h, :w].squeeze(0).permute(1, 2, 0)
            restored = (restored.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            p_after = psnr_fn(clean, restored)
            s_after = ssim_fn(clean, restored, channel_axis=2)

            psnr_before.append(p_before)
            psnr_after.append(p_after)
            ssim_before.append(s_before)
            ssim_after.append(s_after)

            if lpips_fn is not None:
                c_t = torch.from_numpy(clean).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
                r_t = torch.from_numpy(restored).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
                with torch.no_grad():
                    lpips_val = lpips_fn(c_t.to(device), r_t.to(device)).item()
                lpips_list.append(lpips_val)

            if idx < 3:
                Image.fromarray(restored).save(os.path.join(OUTPUT_DIR, f"restored_{fname}"))

    model.train()

    result = {
        "psnr_before": float(np.mean(psnr_before)),
        "psnr_after": float(np.mean(psnr_after)),
        "ssim_before": float(np.mean(ssim_before)),
        "ssim_after": float(np.mean(ssim_after)),
    }
    if lpips_list:
        result["lpips_after"] = float(np.mean(lpips_list))

    log(f"  AVG: PSNR {result['psnr_before']:.2f}->{result['psnr_after']:.2f}, "
        f"SSIM {result['ssim_before']:.4f}->{result['ssim_after']:.4f}"
        + (f", LPIPS {result['lpips_after']:.4f}" if "lpips_after" in result else ""))

    return result


def main():
    log("=" * 60)
    log("Photo Restoration - Real SwinIR (Swin Transformer)")
    log("=" * 60)
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect pairs
    clean_files = set(f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    degraded_files = sorted(f for f in os.listdir(DEGRADED_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    pairs = []
    for df in degraded_files:
        if df in clean_files:
            pairs.append((
                os.path.join(DEGRADED_DIR, df),
                os.path.join(CLEAN_DIR, df),
                df,
            ))
    log(f"Found {len(pairs)} pairs")

    # Split train/test
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    test_size = min(10, len(pairs) // 10)
    test_idx = set(indices[:test_size].tolist())
    test_pairs = [pairs[i] for i in sorted(test_idx)]
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in test_idx]
    log(f"Dataset: {len(pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")

    # Build real SwinIR model
    log("Building real SwinIR (Swin Transformer) model...")
    from photo_revival.network_swinir import SwinIR

    model = SwinIR(
        img_size=PATCH_SIZE,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        upscale=1,
        upsampler='',
        resi_connection='1conv',
    )
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {num_params:,}")
    model = model.to(device)

    # LPIPS
    log("Loading LPIPS...")
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # VGG Perceptual Loss
    log("Loading VGG perceptual loss...")
    perceptual_loss_fn = VGGPerceptualLoss(device)

    # Metrics
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # Training setup
    num_epochs = 100
    lr = 2e-4
    patience = 15
    perceptual_weight = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    l1_criterion = nn.L1Loss()

    dataset = RestorationDataset(train_pairs, patch_size=PATCH_SIZE, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Train
    best_psnr = 0
    best_lpips = float('inf')
    patience_counter = 0
    loss_history = []
    best_state = None

    log(f"\nTraining: {num_epochs} epochs, lr={lr}, batch={BATCH_SIZE}, patch={PATCH_SIZE}")
    log(f"Loss: L1 + {perceptual_weight} * VGG_Perceptual")
    log(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_perc = 0.0

        for dmg_batch, clean_batch in loader:
            dmg_batch = dmg_batch.to(device)
            clean_batch = clean_batch.to(device)

            output = model(dmg_batch)

            l1_loss = l1_criterion(output, clean_batch)
            perc_loss = perceptual_loss_fn(output, clean_batch)
            loss = l1_loss + perceptual_weight * perc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1 += l1_loss.item()
            epoch_perc += perc_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        avg_l1 = epoch_l1 / len(loader)
        avg_perc = epoch_perc / len(loader)
        loss_history.append(avg_loss)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"\nEpoch {epoch+1}/{num_epochs} - loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, Perc: {avg_perc:.6f}), lr: {scheduler.get_last_lr()[0]:.2e}")
            result = evaluate(model, test_pairs, device, psnr, ssim, lpips_fn)

            if result["psnr_after"] > best_psnr:
                best_psnr = result["psnr_after"]
                best_lpips = result.get("lpips_after", float('inf'))
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
                log(f"  New best! PSNR: {best_psnr:.2f}, LPIPS: {best_lpips:.4f}")
            else:
                patience_counter += 1
                log(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                log(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            log(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, Perc: {avg_perc:.6f})")

    # Final evaluation with best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    log(f"\n{'='*60}")
    log("FINAL EVALUATION (best model)")
    log(f"{'='*60}")
    final_result = evaluate(model, test_pairs, device, psnr, ssim, lpips_fn)

    # Save final
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))

    # Save log
    log_data = {
        "final_result": final_result,
        "best_psnr": best_psnr,
        "config": {
            "model": "SwinIR (real Swin Transformer)",
            "num_params": num_params,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "patience": patience,
            "perceptual_weight": perceptual_weight,
            "loss": "L1 + VGG_perceptual",
        },
        "loss_history": loss_history,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)

    log(f"\nTraining complete. Best PSNR: {best_psnr:.2f}, Best LPIPS: {best_lpips:.4f}")
    log(f"Results saved to {OUTPUT_DIR}/ and {LOG_FILE}")


if __name__ == "__main__":
    main()

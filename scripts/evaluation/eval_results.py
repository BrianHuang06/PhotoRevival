import os, numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from photo_revival.paths import DATASETS_DIR, EVALUATIONS_DIR

clean_dir = str(DATASETS_DIR / "archive" / "03_Synthetic_Dataset" / "Train_GT_Clean")
out_dir = str(EVALUATIONS_DIR / "verify_lora")
test_files = ["lrp_img10.jpg", "lrp_img119.jpg", "lrp_img147.jpg"]

checkpoints = ["epoch25", "epoch35", "epoch45", "epoch50", "final"]
strengths = [0.3, 0.5, 0.7]
guidances = [5.0, 7.5]

results = []
for ckpt in checkpoints:
    for s in strengths:
        for g in guidances:
            scores = []
            for f in test_files:
                pred_path = os.path.join(out_dir, f"{ckpt}_s{s}_g{g}_{f}")
                clean_path = os.path.join(clean_dir, f)
                if os.path.exists(pred_path) and os.path.exists(clean_path):
                    pred = np.array(Image.open(pred_path).convert("RGB").resize((512, 512)))
                    gt = np.array(Image.open(clean_path).convert("RGB").resize((512, 512)))
                    scores.append((psnr(gt, pred), ssim(gt, pred, channel_axis=2)))
            if scores:
                avg_p = np.mean([x[0] for x in scores])
                avg_s = np.mean([x[1] for x in scores])
                results.append((ckpt, s, g, avg_p, avg_s))

header = "{:<10} {:<6} {:<6} {:<8} {:<8}".format("Ckpt", "Str", "Guid", "PSNR", "SSIM")
print(header)
print("-" * 40)
for r in sorted(results, key=lambda x: x[3], reverse=True):
    line = "{:<10} {:<6} {:<6} {:<8.2f} {:<8.4f}".format(r[0], r[1], r[2], r[3], r[4])
    print(line)

best = max(results, key=lambda x: x[3])
print("\nBest: {} s={} g={} PSNR={:.2f} SSIM={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))

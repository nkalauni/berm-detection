"""
One clean, presentation-ready summary figure (not the dense diagnostic
plots from plot_diagnostics.py) for status updates: a buffer-width
comparison chart + one strong example detection.

Usage:
    uv run python scripts/make_summary_figure.py --checkpoint-dir outputs/checkpoints/altarvalley_combined
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

BUFFER_RESULTS = {  # from docs/experiments.md
    2: 0.1427, 3: 0.1534, 4: 0.1695, 5: 0.1772,
    6: 0.1704, 7: 0.1498, 8: 0.1792, 10: 0.1596,
}


def find_best_patch(eval_dir: Path, patch_size: int, min_berm_px: int = 150):
    with rasterio.open(eval_dir / "confusion.tif") as src:
        conf = src.read(1)
    H, W = conf.shape
    n_rows, n_cols = H // patch_size, W // patch_size
    best = None
    for i in range(n_rows):
        for j in range(n_cols):
            block = conf[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            tp, fp, fn = (block == 1).sum(), (block == 2).sum(), (block == 3).sum()
            if tp + fn < min_berm_px:
                continue
            iou = tp / (tp + fp + fn + 1e-9)
            if best is None or iou > best[0]:
                best = (iou, i, j)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads((args.checkpoint_dir / "manifest.json").read_text())
    eval_dir = args.checkpoint_dir / "eval_val"
    row_range = tuple(manifest["val_range"])
    patch_size = manifest["patch_size"]
    channels = manifest["channels"]
    stack_path = Path(manifest["stack_path"])
    mask_path = Path(manifest["mask_path"])

    iou, i, j = find_best_patch(eval_dir, patch_size)
    win = Window(j * patch_size, row_range[0] + i * patch_size, patch_size, patch_size)

    hs_idx = channels.index("multidirectional_hillshade") + 1
    with rasterio.open(stack_path) as src:
        hs = src.read(hs_idx, window=win).astype(np.float32)
    hs = np.clip((hs - np.percentile(hs, 2)) / (np.percentile(hs, 98) - np.percentile(hs, 2) + 1e-6), 0, 1)
    with rasterio.open(mask_path) as src:
        true_mask = src.read(1, window=win)
    with rasterio.open(eval_dir / "predictions.tif") as src:
        pred = src.read(1, window=Window(j * patch_size, i * patch_size, patch_size, patch_size))

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 1.3, 0.05, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    widths = sorted(BUFFER_RESULTS)
    ious = [BUFFER_RESULTS[w] for w in widths]
    colors = ["#4c72b0" if w not in (5, 8) else "#dd8452" for w in widths]
    ax0.bar([str(w) for w in widths], ious, color=colors)
    ax0.set_xlabel("label buffer width (m)")
    ax0.set_ylabel("IoU (full-coverage eval)")
    ax0.set_title("Buffer width comparison")
    ax0.grid(axis="y", alpha=0.3)

    ax_gap = fig.add_subplot(gs[2])
    ax_gap.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(hs, cmap="gray")
    ax1.set_title("Terrain (hillshade)")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[3])
    ax2.imshow(hs, cmap="gray")
    ax2.imshow(np.ma.masked_where(true_mask != 1, true_mask), cmap="autumn", alpha=0.85, vmin=0, vmax=1)
    ax2.set_title("Labeled berm")
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[4])
    ax3.imshow(hs, cmap="gray")
    ax3.imshow(np.ma.masked_where(pred <= 0.5, pred), cmap="cool", alpha=0.85, vmin=0, vmax=1)
    ax3.set_title(f"Model prediction (IoU={iou:.2f})")
    ax3.set_xticks([]); ax3.set_yticks([])

    fig.suptitle("Berm Detection — Model Performance Summary", fontsize=13, y=1.02)
    fig.tight_layout()
    out = args.out or (args.checkpoint_dir / "summary_figure.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""
Diagnostic plots for a trained checkpoint, using the outputs of
scripts/evaluate.py. Produces, into <eval-dir>/diagnostics/:

  spatial_confusion.png   TP/FP/FN over hillshade, downsampled for display
  region_iou_heatmap.png  IoU per spatial block -- is performance uniform
                          or concentrated in specific sub-areas?
  threshold_curve.png     precision/recall/F1/IoU vs. decision threshold
  crop_gallery.png        best/typical/worst individual patches, each
                          showing hillshade, true-color NAIP, true mask,
                          predicted probability
  channel_sensitivity.png IoU drop when each channel is zeroed at
                          inference -- which channels does the model
                          actually rely on?
  training_curves.png     loss/IoU/F1 over training epochs

Usage:
    uv run python scripts/plot_diagnostics.py --checkpoint-dir outputs/checkpoints/altarvalley --device cuda:2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_manifest
from src.data.dataset import get_band_indices, load_norm_stats, normalize_stack
from src.models.unet import build_model

NODATA_MASK = 255
CONF_COLORS = {0: (0.85, 0.85, 0.85), 1: (0.10, 0.75, 0.20), 2: (0.90, 0.15, 0.15), 3: (0.15, 0.35, 0.95)}
# TN=light grey, TP=green, FP=red, FN=blue


def decimated_read(path: Path, band: int, max_dim: int = 1600):
    with rasterio.open(path) as src:
        factor = max(1, max(src.width, src.height) // max_dim)
        out_shape = (src.height // factor, src.width // factor)
        data = src.read(band, out_shape=out_shape)
        nodata = src.nodata
    return data, nodata, factor


def plot_spatial_confusion_over_hillshade(eval_dir: Path, stack_path: Path, channels: list, row_range: tuple, out_path: Path):
    conf, _, factor = decimated_read(eval_dir / "confusion.tif", 1)
    hs_band_idx = channels.index("multidirectional_hillshade") + 1 if "multidirectional_hillshade" in channels else None
    with rasterio.open(stack_path) as src:
        h = row_range[1] - row_range[0]
        out_shape = (h // factor, src.width // factor)
        if hs_band_idx is not None:
            hs = src.read(hs_band_idx, window=Window(0, row_range[0], src.width, h), out_shape=out_shape).astype(np.float32)
        else:
            hs = np.full(out_shape, np.nan)

    hs_valid = np.isfinite(hs) & (hs > -30000)
    hs_norm = np.zeros_like(hs)
    if hs_valid.any():
        lo, hi = np.percentile(hs[hs_valid], [2, 98])
        hs_norm[hs_valid] = np.clip((hs[hs_valid] - lo) / (hi - lo + 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(12, 12 * conf.shape[0] / conf.shape[1]))
    ax.imshow(hs_norm, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*conf.shape, 4))
    for code, color in CONF_COLORS.items():
        if code == 0:
            continue  # leave TN transparent so hillshade shows through
        m = conf == code
        overlay[m] = (*color, 0.75)
    ax.imshow(overlay)
    ax.set_title("Confusion over hillshade: TP=green  FP=red  FN=blue")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_region_iou_heatmap(eval_dir: Path, block_size: int, out_path: Path):
    with rasterio.open(eval_dir / "confusion.tif") as src:
        conf = src.read(1)
    H, W = conf.shape
    n_rows, n_cols = H // block_size, W // block_size
    iou_grid = np.full((n_rows, n_cols), np.nan)
    density_grid = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            block = conf[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            tp = (block == 1).sum()
            fp = (block == 2).sum()
            fn = (block == 3).sum()
            valid = (block != NODATA_MASK).sum()
            density_grid[i, j] = (tp + fn) / (valid + 1e-9)  # fraction of block that's actually berm
            if tp + fp + fn > 0:
                iou_grid[i, j] = tp / (tp + fp + fn)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im0 = axes[0].imshow(iou_grid, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_title(f"IoU per {block_size}x{block_size}px block\n(blank = no berm pixels in block)")
    fig.colorbar(im0, ax=axes[0], fraction=0.04)
    im1 = axes[1].imshow(density_grid, cmap="viridis")
    axes[1].set_title("Berm pixel density per block\n(ground truth)")
    fig.colorbar(im1, ax=axes[1], fraction=0.04)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(eval_dir: Path, out_path: Path):
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    sweep = metrics["threshold_sweep"]
    ths = sorted(sweep.keys(), key=float)
    precision = [sweep[t]["precision"] for t in ths]
    recall = [sweep[t]["recall"] for t in ths]
    f1 = [sweep[t]["f1"] for t in ths]
    iou = [sweep[t]["iou"] for t in ths]
    ths_f = [float(t) for t in ths]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ths_f, precision, label="precision", marker=".")
    ax.plot(ths_f, recall, label="recall", marker=".")
    ax.plot(ths_f, f1, label="F1", marker=".")
    ax.plot(ths_f, iou, label="IoU", marker=".", linewidth=2)
    best_th = float(metrics["best_threshold_by_iou"])
    ax.axvline(best_th, color="gray", linestyle="--", label=f"best IoU @ {best_th}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_title("Metrics vs. decision threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_patch_stats(conf: np.ndarray, patch_size: int):
    H, W = conf.shape
    n_rows, n_cols = H // patch_size, W // patch_size
    patch_iou = np.full((n_rows, n_cols), np.nan)
    patch_berm = np.zeros((n_rows, n_cols))
    patch_fp = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            block = conf[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            tp, fp, fn = (block == 1).sum(), (block == 2).sum(), (block == 3).sum()
            patch_berm[i, j] = tp + fn
            patch_fp[i, j] = fp
            if tp + fp + fn > 0:
                patch_iou[i, j] = tp / (tp + fp + fn)
    return patch_iou, patch_berm, patch_fp


def _render_gallery(
    rows_to_plot, stack_path: Path, mask_path: Path, channels: list, row_range: tuple,
    patch_size: int, probs: np.ndarray, label_fn, out_path: Path,
):
    hs_idx = channels.index("multidirectional_hillshade") + 1 if "multidirectional_hillshade" in channels else None
    rgb_idx = [channels.index(c) + 1 for c in ("red", "green", "blue")] if all(c in channels for c in ("red", "green", "blue")) else None

    n_total = sum(len(v) for _, v in rows_to_plot)
    fig, axes = plt.subplots(n_total, 4, figsize=(14, 3.2 * n_total))
    if n_total == 1:
        axes = axes[None, :]

    with rasterio.open(stack_path) as stack_src, rasterio.open(mask_path) as mask_src:
        row_idx = 0
        for label, group in rows_to_plot:
            for i, j in group:
                win = Window(j * patch_size, row_range[0] + i * patch_size, patch_size, patch_size)
                if hs_idx is not None:
                    hs = stack_src.read(hs_idx, window=win).astype(np.float32)
                    hs = np.clip((hs - np.percentile(hs, 2)) / (np.percentile(hs, 98) - np.percentile(hs, 2) + 1e-6), 0, 1)
                else:
                    hs = np.zeros((patch_size, patch_size))
                true_mask = mask_src.read(1, window=win)
                pred_prob = probs[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

                axes[row_idx, 0].imshow(hs, cmap="gray")
                axes[row_idx, 0].set_ylabel(f"{label}\n{label_fn(i, j)}", fontsize=10)
                if rgb_idx is not None:
                    rgb = np.stack([stack_src.read(b, window=win) for b in rgb_idx], axis=-1)
                    rgb = np.clip(rgb / 255.0, 0, 1)
                    axes[row_idx, 1].imshow(rgb)
                else:
                    axes[row_idx, 1].imshow(hs, cmap="gray")
                axes[row_idx, 2].imshow(hs, cmap="gray")
                axes[row_idx, 2].imshow(np.ma.masked_where(true_mask != 1, true_mask), cmap="autumn", alpha=0.8, vmin=0, vmax=1)
                axes[row_idx, 3].imshow(pred_prob, cmap="viridis", vmin=0, vmax=1)

                if row_idx == 0:
                    for c, title in enumerate(["hillshade", "true-color NAIP", "true mask (red)", "predicted prob."]):
                        axes[row_idx, c].set_title(title, fontsize=10)
                for c in range(4):
                    axes[row_idx, c].set_xticks([])
                    axes[row_idx, c].set_yticks([])
                row_idx += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_false_positive_gallery(
    eval_dir: Path, stack_path: Path, mask_path: Path, channels: list, row_range: tuple,
    patch_size: int, out_path: Path, n_examples: int = 6,
):
    """The crop gallery only samples patches that CONTAIN real berms, so it
    can't show the dominant failure mode the region IoU heatmap reveals:
    false positives spread across berm-free terrain. This finds the patches
    with the most false-positive pixels among patches with near-zero true
    berm coverage, to see what's actually triggering those false alarms."""
    with rasterio.open(eval_dir / "confusion.tif") as src:
        conf = src.read(1)
    with rasterio.open(eval_dir / "predictions.tif") as src:
        probs = src.read(1)

    patch_iou, patch_berm, patch_fp = _per_patch_stats(conf, patch_size)
    n_rows, n_cols = patch_iou.shape
    candidates = [(i, j) for i in range(n_rows) for j in range(n_cols)
                  if patch_berm[i, j] < 5 and patch_fp[i, j] > 0]
    if not candidates:
        print("  [fp_gallery] no false-positive-heavy, berm-free patches found -- skipping")
        return
    candidates.sort(key=lambda ij: patch_fp[ij], reverse=True)
    worst_fp = candidates[:n_examples]

    _render_gallery(
        [("worst false-positive", worst_fp)], stack_path, mask_path, channels, row_range, patch_size, probs,
        label_fn=lambda i, j: f"fp_px={int(patch_fp[i, j])}", out_path=out_path,
    )


def plot_crop_gallery(
    eval_dir: Path, stack_path: Path, mask_path: Path, channels: list, row_range: tuple,
    patch_size: int, out_path: Path, n_examples: int = 3,
):
    with rasterio.open(eval_dir / "confusion.tif") as src:
        conf = src.read(1)
    with rasterio.open(eval_dir / "predictions.tif") as src:
        probs = src.read(1)

    patch_iou, patch_berm, _ = _per_patch_stats(conf, patch_size)
    n_rows, n_cols = patch_iou.shape

    # only consider patches that actually contain some berm ground truth --
    # otherwise "best"/"worst" is dominated by trivial all-background patches
    candidates = [(i, j) for i in range(n_rows) for j in range(n_cols) if patch_berm[i, j] > 20]
    if not candidates:
        print("  [crop_gallery] no patches with enough berm pixels found -- skipping")
        return
    candidates.sort(key=lambda ij: patch_iou[ij] if not np.isnan(patch_iou[ij]) else -1)

    worst = candidates[:n_examples]
    best = candidates[-n_examples:]
    mid_idx = len(candidates) // 2
    typical = candidates[max(0, mid_idx - n_examples // 2): mid_idx - n_examples // 2 + n_examples]
    rows_to_plot = [("worst", worst), ("typical", typical), ("best", best)]

    _render_gallery(
        rows_to_plot, stack_path, mask_path, channels, row_range, patch_size, probs,
        label_fn=lambda i, j: f"IoU={patch_iou[i, j]:.2f}", out_path=out_path,
    )


def plot_channel_sensitivity(
    manifest: dict, checkpoint_dir: Path, checkpoint_file: str, device, out_path: Path,
    n_tiles: int = 200,
):
    channels = manifest["channels"]
    stack_path = Path(manifest["stack_path"])
    mask_path = Path(manifest["mask_path"])
    row_range = tuple(manifest["val_range"])
    patch_size = manifest["patch_size"]
    band_indices = get_band_indices(stack_path, channels)
    stats = load_norm_stats(manifest["norm_stats_path"])

    model = build_model(manifest["model_cfg"]).to(device)
    ckpt = torch.load(checkpoint_dir / checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # sample n_tiles random berm-containing patches from the val region for speed
    with rasterio.open(mask_path) as src:
        width = src.width
        mask_region = src.read(1, window=Window(0, row_range[0], width, row_range[1] - row_range[0]))
    berm_rows, berm_cols = np.where(mask_region == 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(berm_rows), size=min(n_tiles, len(berm_rows)), replace=False)

    def iou_with_zeroed_channel(zero_idx):
        tp = fp = fn = 0
        with rasterio.open(stack_path) as src:
            for k in idx:
                cy, cx = berm_rows[k] + row_range[0], berm_cols[k]
                row = int(np.clip(cy - patch_size // 2, row_range[0], row_range[1] - patch_size))
                col = int(np.clip(cx - patch_size // 2, 0, width - patch_size))
                win = Window(col, row, patch_size, patch_size)
                tile = src.read(indexes=band_indices, window=win).astype(np.float32)
                tile = normalize_stack(tile, channels, stats)
                if zero_idx is not None:
                    tile[zero_idx] = 0.0  # 0.0 in normalized space == that channel's own mean
                mask_tile = mask_region[row - row_range[0]: row - row_range[0] + patch_size, col: col + patch_size]

                with torch.no_grad():
                    logit = model(torch.from_numpy(tile).unsqueeze(0).to(device))
                pred = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                valid = mask_tile != NODATA_MASK
                t = mask_tile[valid]
                p = pred[valid]
                tp += int((p & (t == 1)).sum())
                fp += int((p & (t == 0)).sum())
                fn += int(((1 - p) & (t == 1)).sum())
        return tp / (tp + fp + fn + 1e-9)

    baseline_iou = iou_with_zeroed_channel(None)
    print(f"  [channel_sensitivity] baseline IoU on {len(idx)} berm-centered tiles: {baseline_iou:.4f}")
    drops = {}
    for i, name in enumerate(channels):
        iou = iou_with_zeroed_channel(i)
        drops[name] = baseline_iou - iou
        print(f"  [channel_sensitivity] zero {name:<28} IoU={iou:.4f}  drop={drops[name]:+.4f}")

    order = sorted(drops, key=drops.get, reverse=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if drops[n] > 0 else "#2ca02c" for n in order]
    ax.barh(order, [drops[n] for n in order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("IoU drop when this channel is zeroed (positive = channel helps)")
    ax.set_title(f"Channel sensitivity (baseline IoU={baseline_iou:.3f}, n={len(idx)} berm-centered tiles)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(checkpoint_dir: Path, out_path: Path):
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        print("  [training_curves] no history.json found -- skipping")
        return
    history = json.loads(history_path.read_text())
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_iou"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], label="val IoU")
    axes[1].plot(epochs, history["val_f1"], label="val F1")
    axes[1].axvline(best_epoch, color="gray", linestyle="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("Training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Default: <checkpoint-dir>/eval_val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--block-size", type=int, default=1024, help="Block size for the region IoU heatmap")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Channel sensitivity re-runs the model many times; skip for a faster pass")
    args = parser.parse_args()

    manifest = load_manifest(args.checkpoint_dir)
    eval_dir = args.eval_dir or (args.checkpoint_dir / "eval_val")
    if not (eval_dir / "metrics.json").exists():
        print(f"{eval_dir} has no metrics.json -- run scripts/evaluate.py first.")
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = eval_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_path = Path(manifest["stack_path"])
    mask_path = Path(manifest["mask_path"])
    row_range = tuple(manifest["val_range"])

    print("Plotting spatial confusion over hillshade...")
    plot_spatial_confusion_over_hillshade(eval_dir, stack_path, manifest["channels"], row_range, out_dir / "spatial_confusion.png")

    print("Plotting region IoU heatmap...")
    plot_region_iou_heatmap(eval_dir, args.block_size, out_dir / "region_iou_heatmap.png")

    print("Plotting threshold curve...")
    plot_threshold_curve(eval_dir, out_dir / "threshold_curve.png")

    print("Plotting crop gallery...")
    plot_crop_gallery(eval_dir, stack_path, mask_path, manifest["channels"], row_range,
                       manifest["patch_size"], out_dir / "crop_gallery.png")

    print("Plotting false-positive gallery...")
    plot_false_positive_gallery(eval_dir, stack_path, mask_path, manifest["channels"], row_range,
                                 manifest["patch_size"], out_dir / "fp_gallery.png")

    print("Plotting highlight pair (for non-technical summaries)...")
    plot_highlight_pair(eval_dir, stack_path, mask_path, manifest["channels"], row_range,
                         manifest["patch_size"], out_dir / "highlight_pair.png")

    print("Plotting training curves...")
    plot_training_curves(args.checkpoint_dir, out_dir / "training_curves.png")

    if not args.skip_sensitivity:
        print("Running channel sensitivity (re-runs the model per channel, slower)...")
        plot_channel_sensitivity(manifest, args.checkpoint_dir, args.checkpoint, device, out_dir / "channel_sensitivity.png")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()


def plot_highlight_pair(
    eval_dir: Path, stack_path: Path, mask_path: Path, channels: list, row_range: tuple,
    patch_size: int, out_path: Path,
):
    """Two real rows for a non-technical audience: one solid working
    detection, and one MEDIAN-severity false positive representing the
    dominant, typical failure mode (widespread false positives, not
    missed detections -- see docs/experiments.md finding #4) rather than
    an extreme outlier. Same real pixel data as the other galleries,
    just distilled to the two rows that tell the actual story."""
    with rasterio.open(eval_dir / "confusion.tif") as src:
        conf = src.read(1)
    with rasterio.open(eval_dir / "predictions.tif") as src:
        probs = src.read(1)

    patch_iou, patch_berm, patch_fp = _per_patch_stats(conf, patch_size)
    n_rows, n_cols = patch_iou.shape

    berm_candidates = [(i, j) for i in range(n_rows) for j in range(n_cols) if patch_berm[i, j] > 20]
    berm_candidates.sort(key=lambda ij: patch_iou[ij] if not np.isnan(patch_iou[ij]) else -1)
    good = berm_candidates[-1:]

    fp_candidates = [(i, j) for i in range(n_rows) for j in range(n_cols)
                      if patch_berm[i, j] < 5 and patch_fp[i, j] > 0]
    fp_candidates.sort(key=lambda ij: patch_fp[ij])
    typical_fp = fp_candidates[len(fp_candidates) // 2: len(fp_candidates) // 2 + 1]

    rows_to_plot = [
        ("working example", good),
        ("typical false positive", typical_fp),
    ]

    def label_fn(i, j):
        if (i, j) in good:
            return f"IoU={patch_iou[i, j]:.2f}"
        return f"fp_px={int(patch_fp[i, j])}"

    _render_gallery(
        rows_to_plot, stack_path, mask_path, channels, row_range, patch_size, probs,
        label_fn=label_fn, out_path=out_path,
    )

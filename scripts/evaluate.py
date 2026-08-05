"""
Run a trained checkpoint over its (or a chosen) row range and produce:
  - predictions.tif   float32 predicted probability, same grid as the stack
  - confusion.tif     uint8: 0=TN 1=TP 2=FP 3=FN 255=ignore (vs. the default
                      0.5 threshold; re-derivable at any threshold from
                      predictions.tif + the mask)
  - metrics.json      precision/recall/F1/IoU swept across thresholds

Reads everything it needs (channels, model architecture, row ranges) from
the checkpoint directory's manifest.json (written by train.py), so it
doesn't depend on the config file being unchanged since training.

Usage:
    uv run python scripts/evaluate.py --checkpoint-dir outputs/checkpoints/altarvalley --device cuda:2
    uv run python scripts/evaluate.py --checkpoint-dir outputs/checkpoints/altarvalley --split full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_band_indices, load_norm_stats, normalize_stack
from src.models.unet import build_model

NODATA_MASK = 255


def load_manifest(checkpoint_dir: Path) -> dict:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- evaluate.py needs the manifest train.py writes "
            "alongside checkpoints to know which channels/architecture were used."
        )
    return json.loads(manifest_path.read_text())


def tiled_inference(
    model, stack_path: Path, band_indices: list, stats: dict, channel_names: list,
    row_range: tuple, width: int, patch_size: int, device, batch_size: int = 16,
) -> np.ndarray:
    """Non-overlapping tiled inference over row_range x [0, width). Partial
    edge tiles are zero-padded up to patch_size before the model (a U-Net's
    skip connections need dims divisible by the encoder's downsampling
    factor) and cropped back afterward."""
    row_start, row_end = row_range
    height = row_end - row_start
    probs = np.zeros((height, width), dtype=np.float32)

    tile_coords = []
    for r in range(0, height, patch_size):
        for c in range(0, width, patch_size):
            tile_coords.append((r, c))

    with rasterio.open(stack_path) as src:
        model.eval()
        with torch.no_grad():
            for batch_start in range(0, len(tile_coords), batch_size):
                batch_coords = tile_coords[batch_start: batch_start + batch_size]
                batch_imgs = []
                for r, c in batch_coords:
                    h = min(patch_size, height - r)
                    w = min(patch_size, width - c)
                    window = Window(c, row_start + r, w, h)
                    tile = src.read(indexes=band_indices, window=window).astype(np.float32)
                    tile = normalize_stack(tile, channel_names, stats)
                    if h < patch_size or w < patch_size:
                        padded = np.zeros((tile.shape[0], patch_size, patch_size), dtype=np.float32)
                        padded[:, :h, :w] = tile
                        tile = padded
                    batch_imgs.append(tile)

                batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
                logits = model(batch_tensor)
                batch_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                for (r, c), p in zip(batch_coords, batch_probs):
                    h = min(patch_size, height - r)
                    w = min(patch_size, width - c)
                    probs[r:r + h, c:c + w] = p[:h, :w]

                done = batch_start + len(batch_coords)
                if done % (batch_size * 20) == 0 or done == len(tile_coords):
                    print(f"  inference: {done}/{len(tile_coords)} tiles", flush=True)

    return probs


def sweep_thresholds(probs: np.ndarray, mask: np.ndarray, thresholds: list) -> dict:
    valid = mask != NODATA_MASK
    p_valid = probs[valid]
    t_valid = mask[valid].astype(np.int64)
    results = {}
    for th in thresholds:
        pred = (p_valid > th).astype(np.int64)
        tp = int((pred & (t_valid == 1)).sum())
        fp = int((pred & (t_valid == 0)).sum())
        fn = int(((1 - pred) & (t_valid == 1)).sum())
        tn = int(((1 - pred) & (t_valid == 0)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        results[f"{th:.2f}"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pth", help="Filename within checkpoint-dir")
    parser.add_argument("--split", choices=["val", "train", "full"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <checkpoint-dir>/eval_<split>")
    args = parser.parse_args()

    manifest = load_manifest(args.checkpoint_dir)
    stack_path = Path(manifest["stack_path"])
    mask_path = Path(manifest["mask_path"])
    channels = manifest["channels"]
    patch_size = manifest["patch_size"]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    with rasterio.open(stack_path) as src:
        width, total_height = src.width, src.height
        stack_profile = src.profile

    if args.split == "val":
        row_range = tuple(manifest["val_range"])
    elif args.split == "train":
        row_range = tuple(manifest["train_range"])
    else:
        row_range = (0, total_height)
    print(f"Split: {args.split}  Row range: {row_range}")

    model = build_model(manifest["model_cfg"]).to(device)
    ckpt = torch.load(args.checkpoint_dir / args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, val_iou={ckpt.get('val_iou'):.4f} at save time)")

    band_indices = get_band_indices(stack_path, channels)
    stats = load_norm_stats(manifest["norm_stats_path"])

    probs = tiled_inference(
        model, stack_path, band_indices, stats, channels, row_range, width, patch_size, device,
        batch_size=args.batch_size,
    )

    with rasterio.open(mask_path) as src:
        mask = src.read(1, window=Window(0, row_range[0], width, row_range[1] - row_range[0]))

    out_dir = args.out_dir or (args.checkpoint_dir / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    row_transform = stack_profile["transform"] * rasterio.Affine.translation(0, row_range[0])
    out_profile = dict(stack_profile)
    out_profile.update(count=1, height=row_range[1] - row_range[0], width=width, transform=row_transform)

    pred_profile = dict(out_profile, dtype="float32", nodata=-1.0)
    probs_out = np.where(mask == NODATA_MASK, -1.0, probs).astype(np.float32)
    with rasterio.open(out_dir / "predictions.tif", "w", **pred_profile) as dst:
        dst.write(probs_out, 1)

    pred_binary = (probs > 0.5).astype(np.uint8)
    valid = mask != NODATA_MASK
    confusion = np.full(mask.shape, NODATA_MASK, dtype=np.uint8)
    t = mask.astype(np.uint8)
    confusion[valid & (pred_binary == 0) & (t == 0)] = 0  # TN
    confusion[valid & (pred_binary == 1) & (t == 1)] = 1  # TP
    confusion[valid & (pred_binary == 1) & (t == 0)] = 2  # FP
    confusion[valid & (pred_binary == 0) & (t == 1)] = 3  # FN
    conf_profile = dict(out_profile, dtype="uint8", nodata=NODATA_MASK, compress="lzw")
    with rasterio.open(out_dir / "confusion.tif", "w", **conf_profile) as dst:
        dst.write(confusion, 1)

    thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    metrics = sweep_thresholds(probs, mask, thresholds)
    best_th = max(metrics, key=lambda k: metrics[k]["iou"])
    summary = {
        "checkpoint": str(args.checkpoint_dir / args.checkpoint),
        "split": args.split,
        "row_range": list(row_range),
        "best_threshold_by_iou": best_th,
        "metrics_at_0.50": metrics["0.50"],
        "metrics_at_best": metrics[best_th],
        "threshold_sweep": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAt threshold 0.50: IoU={metrics['0.50']['iou']:.4f}  F1={metrics['0.50']['f1']:.4f}  "
          f"precision={metrics['0.50']['precision']:.4f}  recall={metrics['0.50']['recall']:.4f}")
    print(f"Best threshold {best_th}: IoU={metrics[best_th]['iou']:.4f}  F1={metrics[best_th]['f1']:.4f}")
    print(f"Wrote predictions.tif, confusion.tif, metrics.json to {out_dir}")


if __name__ == "__main__":
    main()

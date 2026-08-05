"""
Small-scale hyperparameter sweep, for picking lr/batch_size/loss before
committing to a full training run on the whole study area.

Trains many short runs on a small ROW-RANGE SUBSET of the stack (not the
full raster) so each run finishes in a couple minutes instead of a couple
hours. The subset gets its own train/val split (via split_row_range), so
sweep results still reflect held-out performance, just on less data.

Usage:
    uv run python scripts/sweep_hyperparams.py --config configs/train_altarvalley.yaml --device cuda:2
    uv run python scripts/sweep_hyperparams.py --config configs/train_altarvalley.yaml --device cuda:2 \
        --rows 20000 --epochs 10
"""

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import rasterio
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BermDataset, split_row_range
from src.models.unet import build_model
from src.training.losses import get_loss
from src.training.trainer import Trainer

# Kept small and simple on purpose -- this is meant to be a fast, cheap scan
# to rule bad regions of hyperparameter space out, not an exhaustive search.
SEARCH_SPACE = {
    "lr": [1e-3, 3e-4, 1e-4],
    "batch_size": [8, 16],
    "loss": ["dice", "bce_dice"],
}


def build_datasets(data_cfg: dict, channels: list, row_start: int, n_rows: int):
    # split_row_range's offsets are relative to 0 -- shift by row_start so
    # the caller can target a specific sub-region, not always rows [0:n_rows].
    local_train, local_val = split_row_range(
        n_rows, val_split=data_cfg.get("val_split", 0.2), patch_size=data_cfg["patch_size"]
    )
    train_range = (row_start + local_train[0], row_start + local_train[1])
    val_range = (row_start + local_val[0], row_start + local_val[1])
    train_ds = BermDataset(
        data_cfg["stack_path"], data_cfg["mask_path"], data_cfg["norm_stats_path"], channels, train_range,
        patch_size=data_cfg["patch_size"], augment=True,
        pos_fraction=data_cfg.get("pos_fraction", 0.5),
        samples_per_epoch=data_cfg.get("sweep_samples_per_epoch", 400),
    )
    val_ds = BermDataset(
        data_cfg["stack_path"], data_cfg["mask_path"], data_cfg["norm_stats_path"], channels, val_range,
        # NOT pos_fraction=0.0 -- see train.py for why: pure-random crops
        # essentially never contain a berm pixel given how rare/thin they
        # are, which makes IoU stuck at 0.0 regardless of the model.
        patch_size=data_cfg["patch_size"], augment=False,
        pos_fraction=data_cfg.get("val_pos_fraction", data_cfg.get("pos_fraction", 0.5)),
        samples_per_epoch=data_cfg.get("sweep_val_samples_per_epoch", 100),
    )
    return train_ds, val_ds, train_range, val_range


def run_one(model_cfg: dict, train_ds, val_ds, device, lr, batch_size, loss_name, epochs, checkpoint_dir):
    model = build_model(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model_cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = get_loss(loss_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    trainer = Trainer(model, optimizer, scheduler, loss_fn, device, checkpoint_dir)
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader, epochs)
    elapsed = time.time() - t0
    return {
        "lr": lr, "batch_size": batch_size, "loss": loss_name,
        "best_val_iou": trainer.best_val_iou,
        "final_val_f1": history["val_f1"][-1],
        "final_train_loss": history["train_loss"][-1],
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--row-start", type=int, default=0, help="First row of the subset (must fall within the labeled/berm region -- check before running)")
    parser.add_argument("--rows", type=int, default=12000, help="Row-range subset height for the sweep")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per sweep run (kept short)")
    parser.add_argument("--out", type=Path, default=Path("outputs/sweep/results.csv"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    channels = data_cfg["channels"]
    device = torch.device(args.device)
    print(f"Device: {device}")

    with rasterio.open(data_cfg["stack_path"]) as src:
        total_height = src.height
    n_rows = min(args.rows, total_height - args.row_start)
    print(f"Sweeping on rows [{args.row_start}:{args.row_start + n_rows}] of {total_height} total "
          f"(~{100*n_rows/total_height:.1f}% of the raster)")

    train_ds, val_ds, train_range, val_range = build_datasets(data_cfg, channels, args.row_start, n_rows)
    print(f"Train rows: {train_range} ({len(train_ds._berm_locs):,} berm px)  "
          f"Val rows: {val_range} ({len(val_ds._berm_locs):,} berm px)")
    if len(train_ds._berm_locs) == 0 or len(val_ds._berm_locs) == 0:
        print("WARNING: train or val subset has ZERO berm pixels -- IoU will be meaningless. "
              "Pick a --row-start within the labeled region.")
    print(f"Train samples/epoch: {len(train_ds)}  Val samples/epoch: {len(val_ds)}")

    model_cfg = dict(cfg["model"])
    model_cfg["in_channels"] = len(channels)
    rgb_names = ("red", "green", "blue")
    if all(c in channels for c in rgb_names):
        model_cfg["rgb_channel_indices"] = [channels.index(c) for c in rgb_names]
    else:
        model_cfg["rgb_channel_indices"] = None
        model_cfg["encoder_weights"] = None

    combos = list(itertools.product(SEARCH_SPACE["lr"], SEARCH_SPACE["batch_size"], SEARCH_SPACE["loss"]))
    print(f"Running {len(combos)} configs x {args.epochs} epochs each\n")

    results = []
    for i, (lr, batch_size, loss_name) in enumerate(combos, start=1):
        print(f"--- [{i}/{len(combos)}] lr={lr} batch_size={batch_size} loss={loss_name} ---")
        ckpt_dir = Path(f"outputs/sweep/run_{i:02d}")
        r = run_one(model_cfg, train_ds, val_ds, device, lr, batch_size, loss_name, args.epochs, ckpt_dir)
        results.append(r)
        print(f"  best_val_iou={r['best_val_iou']:.4f}  ({r['elapsed_s']:.0f}s)\n")

    results.sort(key=lambda r: r["best_val_iou"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print("SWEEP RESULTS (best first)")
    print("=" * 70)
    for r in results:
        print(f"  lr={r['lr']:<8} batch_size={r['batch_size']:<3} loss={r['loss']:<9} "
              f"best_val_iou={r['best_val_iou']:.4f}  final_val_f1={r['final_val_f1']:.4f}")
    print(f"\nBest config: lr={results[0]['lr']} batch_size={results[0]['batch_size']} loss={results[0]['loss']}")
    print(f"Full results: {args.out}")


if __name__ == "__main__":
    main()

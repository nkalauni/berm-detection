"""
Train the berm detection U-Net.

The set of input channels is entirely a config choice -- see the
`data.channels` list in the YAML. The model's in_channels and, if using
ImageNet weights, which channels get the pretrained R/G/B slots
(first-conv inflation) are both derived automatically from that list.
Changing how many/which channels a run uses means editing the YAML, not
this script, dataset.py, or the model.

Usage:
    uv run python scripts/train.py --config configs/train_altarvalley.yaml
    uv run python scripts/train.py --config configs/train_altarvalley.yaml --device mps
    uv run python scripts/train.py --config configs/train_altarvalley.yaml --resume outputs/checkpoints/latest.pth
"""

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cuda:N | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- data ---
    data_cfg = cfg["data"]
    stack_path = Path(data_cfg["stack_path"])
    mask_path = Path(data_cfg["mask_path"])
    norm_stats_path = Path(data_cfg["norm_stats_path"])
    channels = data_cfg["channels"]

    for p, label in [(stack_path, "stack_path"), (mask_path, "mask_path"), (norm_stats_path, "norm_stats_path")]:
        if not p.exists():
            print(f"{label} not found: {p}")
            print("Run scripts/build_feature_stack.py and scripts/rasterize_labels.py first.")
            sys.exit(1)

    with rasterio.open(stack_path) as src:
        height = src.height
    print(f"Channels ({len(channels)}): {channels}")

    patch_size = data_cfg.get("patch_size", 256)
    train_range, val_range = split_row_range(height, val_split=data_cfg.get("val_split", 0.2), patch_size=patch_size)
    print(f"Train rows: {train_range}  Val rows: {val_range}")

    train_ds = BermDataset(
        stack_path, mask_path, norm_stats_path, channels, train_range,
        patch_size=patch_size, augment=True,
        pos_fraction=data_cfg.get("pos_fraction", 0.5),
        samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
    )
    val_ds = BermDataset(
        stack_path, mask_path, norm_stats_path, channels, val_range,
        # NOT pos_fraction=0.0: berms cover ~0.02% of pixels as thin linear
        # features, so pure-random val crops essentially never contain one,
        # making tp/fn always 0 and IoU stuck at 0.0 regardless of the model
        # (found via a hyperparameter sweep where every single config showed
        # val_iou=0.0000 even as train loss dropped meaningfully). Match
        # train's oversampling so the val metric is actually informative.
        patch_size=patch_size, augment=False,
        pos_fraction=data_cfg.get("val_pos_fraction", data_cfg.get("pos_fraction", 0.5)),
        samples_per_epoch=data_cfg.get("val_samples_per_epoch", 400),
    )
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        if len(ds._berm_locs) == 0:
            print(f"WARNING: {name} split has ZERO berm pixels in its row range {ds.row_start, ds.row_end} -- "
                  f"IoU will be meaningless for this split.")

    train_cfg = cfg["training"]
    batch_size = train_cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- model ---
    model_cfg = dict(cfg["model"])
    model_cfg["in_channels"] = len(channels)
    rgb_names = ("red", "green", "blue")
    if all(c in channels for c in rgb_names):
        model_cfg["rgb_channel_indices"] = [channels.index(c) for c in rgb_names]
    else:
        model_cfg["rgb_channel_indices"] = None
        if model_cfg.get("encoder_weights"):
            print("No raw R/G/B in `data.channels` -- ignoring encoder_weights, training encoder from random init.")
            model_cfg["encoder_weights"] = None
    model = build_model(model_cfg)

    # --- optimizer & scheduler ---
    lr = train_cfg.get("lr", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- loss ---
    loss_fn = get_loss(train_cfg.get("loss", "bce_dice"))

    # --- trainer ---
    out_cfg = cfg.get("output", {})
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "outputs/checkpoints"))

    trainer = Trainer(model, optimizer, scheduler, loss_fn, device, checkpoint_dir)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}, best val IoU: {trainer.best_val_iou:.4f}")

    history = trainer.fit(train_loader, val_loader, epochs)
    print(f"\nDone. Best val IoU: {trainer.best_val_iou:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

"""
Train the berm detection U-Net.

Usage:
    uv run python scripts/train.py --config configs/train_altarvalley.yaml
    uv run python scripts/train.py --config configs/train_altarvalley.yaml --device mps
    uv run python scripts/train.py --config configs/train_altarvalley.yaml --resume outputs/checkpoints/latest.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BermDataset, build_tile_pairs, split_tiles
from src.models.unet import build_model
from src.training.losses import get_loss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | mps | cpu (auto-detected if omitted)")
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
    tile_pairs = build_tile_pairs(
        dem_dir=Path(data_cfg["dem_dir"]),
        mask_dir=Path(data_cfg["mask_dir"]),
    )
    if not tile_pairs:
        print(f"No (dem, mask) pairs found. Run scripts/rasterize_labels.py first.")
        sys.exit(1)
    print(f"Found {len(tile_pairs)} tile pair(s)")

    train_pairs, val_pairs = split_tiles(
        tile_pairs, val_split=data_cfg.get("val_split", 0.2)
    )
    print(f"Train tiles: {len(train_pairs)}  Val tiles: {len(val_pairs)}")

    patch_size = data_cfg.get("patch_size", 256)
    samples_per_tile = data_cfg.get("samples_per_tile", 200)
    pos_fraction = data_cfg.get("pos_fraction", 0.5)

    train_ds = BermDataset(train_pairs, patch_size=patch_size,
                           augment=True, pos_fraction=pos_fraction,
                           samples_per_tile=samples_per_tile)
    val_ds = BermDataset(val_pairs, patch_size=patch_size,
                         augment=False, pos_fraction=0.0,
                         samples_per_tile=samples_per_tile)

    train_cfg = cfg["training"]
    batch_size = train_cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- model ---
    model = build_model(cfg["model"])

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

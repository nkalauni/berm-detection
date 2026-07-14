"""
PyTorch Dataset for berm detection.

Loads DEM tiles + binary mask tiles, computes 4-channel DEM-derived features
(elevation, slope, hillshade, aspect), and returns random crops suitable for
training a semantic segmentation model.

Mask values:
  0   = background
  1   = berm
  255 = ignore (DEM nodata region)
"""

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

DEM_NODATA_THRESHOLD = -999998.0
SAMPLES_PER_TILE     = 200       # random crops sampled per tile per epoch


# ---------------------------------------------------------------------------
# DEM feature helpers  (pure numpy, no geo dependencies)
# ---------------------------------------------------------------------------

def _compute_slope(dem: np.ndarray, res: float = 1.0) -> np.ndarray:
    dy, dx = np.gradient(dem, res, res)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))   # radians [0, pi/2]
    return slope.astype(np.float32)


def _compute_hillshade(dem: np.ndarray, res: float = 1.0,
                       azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    az_rad  = np.deg2rad(360 - azimuth + 90)
    alt_rad = np.deg2rad(altitude)
    dy, dx  = np.gradient(dem, res, res)
    slope   = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect  = np.arctan2(-dy, dx)
    hs = (np.cos(alt_rad) * np.cos(slope) +
          np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1).astype(np.float32)


def _compute_aspect(dem: np.ndarray, res: float = 1.0) -> np.ndarray:
    dy, dx = np.gradient(dem, res, res)
    aspect = np.arctan2(-dy, dx)                     # radians [-pi, pi]
    return ((aspect + np.pi) / (2 * np.pi)).astype(np.float32)  # [0, 1]


def compute_features(dem: np.ndarray) -> np.ndarray:
    """
    Return (4, H, W) float32 array: [elevation, slope, hillshade, aspect].
    Input dem is (H, W) float32 with nan where nodata.
    """
    valid = ~np.isnan(dem)
    mean  = float(dem[valid].mean()) if valid.any() else 0.0
    std   = float(dem[valid].std())  if valid.any() else 1.0
    std   = std if std > 1e-6 else 1.0

    dem_filled = np.where(valid, dem, mean)          # fill nan for gradient ops

    elev      = ((dem_filled - mean) / std).astype(np.float32)
    slope     = _compute_slope(dem_filled) / (np.pi / 2)   # normalise to [0,1]
    hillshade = _compute_hillshade(dem_filled)
    aspect    = _compute_aspect(dem_filled)

    features = np.stack([elev, slope, hillshade, aspect], axis=0)   # (4, H, W)

    # zero out nodata locations so they don't bleed into valid pixels
    features[:, ~valid] = 0.0
    return features


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _build_augmentations() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BermDataset(Dataset):
    """
    Random-crop dataset over a list of (dem_path, mask_path) tile pairs.

    Args:
        tile_pairs:    List of (dem_path, mask_path) Path tuples.
        patch_size:    Square crop size in pixels (default 256).
        augment:       Apply spatial augmentations (training only).
        pos_fraction:  Fraction of crops centred on a berm pixel (oversample
                       positives to combat class imbalance).
    """

    def __init__(
        self,
        tile_pairs: list[tuple[Path, Path]],
        patch_size: int = 256,
        augment: bool = False,
        pos_fraction: float = 0.5,
        samples_per_tile: int = SAMPLES_PER_TILE,
    ):
        self.tile_pairs       = tile_pairs
        self.patch_size       = patch_size
        self.augment          = augment
        self.pos_fraction     = pos_fraction
        self.samples_per_tile = samples_per_tile
        self.aug              = _build_augmentations() if augment else None
        self._cache: dict     = {}     # (path, path) → (features, mask) — LRU would be better for large sets

    def __len__(self) -> int:
        return len(self.tile_pairs) * self.samples_per_tile

    def _load_tile(self, dem_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = (dem_path, mask_path)
        if key in self._cache:
            return self._cache[key]

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)

        # Replace nodata with nan
        dem[dem <= DEM_NODATA_THRESHOLD] = np.nan

        features = compute_features(dem)              # (4, H, W)

        result = (features, mask)
        # Only cache if tile is reasonably small (< 500 MB tensor equivalent)
        if features.nbytes < 500 * 1024 * 1024:
            self._cache[key] = result
        return result

    def _random_crop(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, H, W = features.shape
        p = self.patch_size

        if H < p or W < p:
            # Tile smaller than patch — pad and return full tile
            pad_h = max(0, p - H)
            pad_w = max(0, p - W)
            features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)))
            mask     = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=255)
            return features[:, :p, :p], mask[:p, :p]

        # Try positive crop (berm pixel in centre region)
        if rng.random() < self.pos_fraction:
            berm_ys, berm_xs = np.where(mask == 1)
            if len(berm_ys) > 0:
                pick = rng.integers(len(berm_ys))
                cy, cx = int(berm_ys[pick]), int(berm_xs[pick])
                row = int(np.clip(cy - p // 2, 0, H - p))
                col = int(np.clip(cx - p // 2, 0, W - p))
                return features[:, row:row + p, col:col + p], mask[row:row + p, col:col + p]

        # Random crop
        row = rng.integers(0, H - p + 1)
        col = rng.integers(0, W - p + 1)
        return features[:, row:row + p, col:col + p], mask[row:row + p, col:col + p]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_idx = idx // self.samples_per_tile
        dem_path, mask_path = self.tile_pairs[tile_idx]

        features, mask = self._load_tile(dem_path, mask_path)

        rng = np.random.default_rng(seed=idx)      # deterministic per idx for reproducibility
        patch_feat, patch_mask = self._random_crop(features, mask, rng)

        # albumentations expects (H, W, C) image
        if self.aug is not None:
            img_hwc = patch_feat.transpose(1, 2, 0)   # (H, W, 4)
            augmented = self.aug(image=img_hwc, mask=patch_mask)
            patch_feat  = augmented["image"].transpose(2, 0, 1)  # back to (4, H, W)
            patch_mask  = augmented["mask"]

        image_tensor = torch.from_numpy(patch_feat.copy())                          # (4, H, W)
        mask_tensor  = torch.from_numpy(patch_mask.copy()).unsqueeze(0).float()     # (1, H, W)
        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_tile_pairs(dem_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """
    Scan dem_dir for *.tif files and find matching masks in mask_dir.
    Mask filename convention: <dem_stem>_mask.tif
    Returns only pairs where both files exist.
    """
    pairs = []
    for dem_path in sorted(dem_dir.glob("*.tif")):
        mask_path = mask_dir / f"{dem_path.stem}_mask.tif"
        if mask_path.exists():
            pairs.append((dem_path, mask_path))
    return pairs


def split_tiles(
    tile_pairs: list[tuple[Path, Path]],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Spatial split: sort tiles by filename (which encodes grid position),
    take the last val_split fraction as validation.  This avoids spatially
    adjacent tiles appearing in both train and val.
    """
    sorted_pairs = sorted(tile_pairs, key=lambda p: p[0].name)
    n_val = max(1, int(len(sorted_pairs) * val_split))
    train_pairs = sorted_pairs[:-n_val]
    val_pairs   = sorted_pairs[-n_val:]
    return train_pairs, val_pairs

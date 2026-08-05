"""
PyTorch Dataset for berm detection.

Loads a precomputed multi-channel feature stack (built by
scripts/build_feature_stack.py -- up to the full 12-channel plan: residual
relief at two scales, openness, profile curvature, multidirectional
hillshade, depression depth, flow-orthogonality, log flow accumulation,
raw NAIP R/G/B, and SAVI anomaly) plus its matching binary mask, and returns
random crops for training.

Which channels a given training run actually uses is a config choice, not
a code choice: the stack GeoTIFF always carries all the channels that have
been built for a dataset (each band tagged with its channel name via
set_band_description), and `channel_names` here is just the subset/order a
particular config asks for. Changing channel count/selection is a matter of
editing a YAML's `data.channels` list -- see configs/train_altarvalley.yaml
-- not touching this file, dataset.py, or the model.

Normalization uses WHOLE-STUDY-AREA statistics (1st/99th percentile clip +
mean/std), saved to norm_stats.json by build_feature_stack.py. This is
deliberately not per-tile normalization: per-tile z-scoring destroys the
absolute-magnitude information that distinguishes a real 1m berm from a
10cm ripple, since both would get rescaled to look the same within their
own tile's statistics.

There is a single merged raster covering the whole study area (needed
anyway for hydrologic channels that require full-watershed context), so the
train/val split is a spatial split BY ROW RANGE within that one raster, not
a list of tiles.

Mask values:
  0   = background
  1   = berm
  255 = ignore (DEM nodata region)
"""

import json
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


def load_norm_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def get_band_indices(stack_path: Path, channel_names: list) -> list:
    """Map channel names to their 1-indexed band numbers in the stack GeoTIFF."""
    with rasterio.open(stack_path) as src:
        descriptions = list(src.descriptions)
    indices = []
    for name in channel_names:
        if name not in descriptions:
            raise ValueError(
                f"Channel '{name}' not found in {stack_path} (has: {descriptions}). "
                f"Run scripts/build_feature_stack.py --channels ... to add it."
            )
        indices.append(descriptions.index(name) + 1)
    return indices


def normalize_stack(stack: np.ndarray, channel_names: list, stats: dict) -> np.ndarray:
    """stack: (C, H, W) float32. Per-channel clip to [p1, p99], then standardize."""
    out = np.empty_like(stack, dtype=np.float32)
    for i, name in enumerate(channel_names):
        s = stats[name]
        clipped = np.clip(stack[i], s["p1"], s["p99"])
        sigma = s["sigma"] if s["sigma"] > 1e-6 else 1.0
        out[i] = (clipped - s["mu"]) / sigma
    return out


def _build_augmentations() -> A.Compose:
    # Safe for every channel in the 12-channel plan: each one is an
    # already-computed intrinsic scalar (a height difference, a curvature,
    # an angle-difference magnitude, a raw pixel intensity, ...), not a raw
    # field with a baked-in EXTERNAL absolute azimuth. Multidirectional
    # hillshade and openness are explicitly azimuth-invariant by
    # construction (averaged over all directions); profile curvature, flow
    # accumulation, and omega are intrinsic/geometric, computed from
    # directions the terrain itself determines, so a rigid flip/rotation of
    # the whole raster rotates those local directions consistently too and
    # the scalar values stay correct -- flipping/rotating the grid just
    # relocates already-correct values to new pixel positions.
    # RandomRotate90 (90-degree multiples only, not a free angle) also avoids
    # resampling/interpolation artifacts at the crop edge.
    # CAUTION if extending beyond these 12: a channel with a baked-in FIXED
    # EXTERNAL azimuth (e.g. a single-direction hillshade at a fixed sun
    # angle, or a raw compass-bearing channel that isn't turned into an
    # angle-difference first) would NOT be safe under this augmentation set.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
    ])


def split_row_range(height: int, val_split: float = 0.2, patch_size: int = 256) -> tuple:
    """
    Spatial split: hold out the southernmost val_split fraction of rows as
    validation, with a patch_size-row gap so no crop can straddle the
    train/val boundary.

    Returns ((train_start, train_end), (val_start, val_end)).
    """
    n_val = max(patch_size, int(height * val_split))
    val_start = height - n_val
    train_end = max(patch_size, val_start - patch_size)
    return (0, train_end), (val_start, height)


class BermDataset(Dataset):
    """
    Random-crop dataset over a row-range of a single (stack, mask) pair.

    Args:
        stack_path:      Path to the multiband feature stack GeoTIFF.
        mask_path:       Path to the matching binary mask GeoTIFF.
        norm_stats_path: Path to norm_stats.json (whole-area stats).
        channel_names:   Band order of the stack (must match stack.tif's
                          band order, e.g. ["resid15","openness","omega","savi_anomaly"]).
        row_range:       (row_start, row_end) region of the raster this
                          dataset instance may sample from.
        patch_size:      Square crop size in pixels.
        augment:         Apply spatial augmentations (training only).
        pos_fraction:    Fraction of crops centred on a berm pixel.
        samples_per_epoch: Number of crops __len__ reports per epoch.
    """

    def __init__(
        self,
        stack_path: Path,
        mask_path: Path,
        norm_stats_path: Path,
        channel_names: list,
        row_range: tuple,
        patch_size: int = 256,
        augment: bool = False,
        pos_fraction: float = 0.5,
        samples_per_epoch: int = 2000,
        seed: int = 42,
    ):
        self.stack_path = Path(stack_path)
        self.mask_path = Path(mask_path)
        self.channel_names = channel_names
        self.band_indices = get_band_indices(self.stack_path, channel_names)
        self.stats = load_norm_stats(norm_stats_path)
        self.row_start, self.row_end = row_range
        self.patch_size = patch_size
        self.augment = augment
        self.pos_fraction = pos_fraction
        self.samples_per_epoch = samples_per_epoch
        self.aug = _build_augmentations() if augment else None

        with rasterio.open(self.mask_path) as src:
            self._width = src.width
            mask_region = src.read(
                1, window=Window(0, self.row_start, self._width, self.row_end - self.row_start)
            )
        berm_rows, berm_cols = np.where(mask_region == 1)
        self._berm_locs = np.stack([berm_rows + self.row_start, berm_cols], axis=1)

        self._seed = seed
        self._stack_src = None
        self._mask_src = None

    def _ensure_open(self):
        # Open file handles lazily, per worker process -- rasterio datasets
        # must not be opened in the parent and pickled to DataLoader workers.
        if self._stack_src is None:
            self._stack_src = rasterio.open(self.stack_path)
            self._mask_src = rasterio.open(self.mask_path)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_location(self, rng: np.random.Generator) -> tuple:
        p = self.patch_size
        if self.pos_fraction > 0 and len(self._berm_locs) > 0 and rng.random() < self.pos_fraction:
            idx = rng.integers(len(self._berm_locs))
            cy, cx = self._berm_locs[idx]
        else:
            cy = rng.integers(self.row_start, self.row_end)
            cx = rng.integers(0, self._width)
        row = int(np.clip(cy - p // 2, self.row_start, max(self.row_start, self.row_end - p)))
        col = int(np.clip(cx - p // 2, 0, max(0, self._width - p)))
        return row, col

    def __getitem__(self, idx: int) -> tuple:
        self._ensure_open()
        # deterministic per idx for val (reproducible metrics); fresh randomness for train
        rng = np.random.default_rng(seed=None if self.augment else self._seed + idx)
        row, col = self._sample_location(rng)
        p = self.patch_size
        window = Window(col, row, p, p)

        stack = self._stack_src.read(indexes=self.band_indices, window=window).astype(np.float32)  # (C, H, W)
        mask = self._mask_src.read(1, window=window).astype(np.uint8)

        stack = normalize_stack(stack, self.channel_names, self.stats)
        stack[:, mask == 255] = 0.0  # zero out DEM-nodata / ignore region

        if self.aug is not None:
            img_hwc = stack.transpose(1, 2, 0)
            augmented = self.aug(image=img_hwc, mask=mask)
            stack = augmented["image"].transpose(2, 0, 1)
            mask = augmented["mask"]

        image_tensor = torch.from_numpy(stack.copy())
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return image_tensor, mask_tensor

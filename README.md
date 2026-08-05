# Berm Detection

Deep learning pipeline for automated detection and mapping of berms and legacy earthworks in dryland watersheds using U-Net semantic segmentation on LiDAR-derived DEMs and NAIP aerial imagery.

**Study areas:** Altar Valley, Safford, Big Chino, Cochise, Upper Gila (Arizona)

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url>
cd berm-detection
uv sync          # creates .venv and installs all dependencies
```

To run any script, prefix with `uv run`:

```bash
uv run python scripts/download_dem_api.py --help
```

---

## End-to-end pipeline

This is the order to run things in, from a fresh clone to a trained model.
Each step name links to its section below.

1. [Download DEM + NAIP](#1-download-dem--naip)
2. [Merge DEM tiles](#2-merge-dem-tiles) into one per-study-area mosaic
3. [Prepare / snap / buffer labels](#3-labels) into polygon training masks
4. [Rasterize labels](#4-rasterize-labels-onto-the-merged-grid) onto the merged DEM's grid
5. [Build the feature stack](#5-build-the-feature-stack) (terrain + hydrology + NAIP channels)
6. [(Optional) Hyperparameter sweep](#6-optional-hyperparameter-sweep) on a small region
7. [Train](#7-train)

---

### 1. Download DEM + NAIP

Large rasters (DEMs, NAIP) are not tracked in git. Shapefiles (labels, snapped lines, polygon masks) already are — no download needed for those.

**DEM — 1m LiDAR-derived elevation tiles (USGS)**

```bash
# Derive AOI from an existing label shapefile
uv run python scripts/download_dem_api.py \
    --shp data/raw/labels/altarvalley_longberms.shp \
    --out data/raw/dem/altar_valley

# Or specify a bounding box directly (minlon,minlat,maxlon,maxlat)
uv run python scripts/download_dem_api.py \
    --bbox -111.6,31.4,-111.0,32.2 \
    --out data/raw/dem/altar_valley
```

Options: `--shp` / `--bbox` (mutually exclusive), `--out` (default `data/raw/dem/`), `--workers`, `--dry-run`.

For the Safford AOI there's a pre-built URL list instead:

```bash
uv run python scripts/download_dem.py   # tiles go to data/raw/dem/ by default
```

All tiles are 1m resolution, float32, EPSG:26912, nodata = `-999999.0` (a few older Safford tiles use `-3.4e+38`).

**NAIP imagery — 1m aerial imagery, 4-band RGBNIR (USDA)**

NAIP is downloaded manually from the [USDA NAIP gateway](https://helpcenter.agvance.net/home/download-naip-imagery) using the county FIPS code matching your study area, then placed in `data/raw/naip/`.

| Study Area   | County       | FIPS  | LiDAR Year |
|--------------|--------------|-------|------------|
| Altar Valley | Pima, AZ     | 04019 | 2018–2021  |
| Safford      | Graham, AZ   | 04009 | 2016       |
| Cochise      | Cochise, AZ  | 04003 | 2020       |

NAIP is distributed as MrSID (`.sid`) files. **This will not open with a normal GDAL/rasterio install** — MrSID is a proprietary format, and open-source GDAL builds exclude its decode driver for licensing reasons. `gdalinfo`/`rasterio.open()` on a `.sid` fails with a generic "not recognized as being in a supported file format" error that looks like a corrupt file but isn't.

Fix — install LizardTech/Extensis's free MrSID Decode SDK (free for decode-only use, no registration required as of writing):

```bash
# Download the SDK for your platform from https://www.lizardtech.com/developer
# (Linux build example, RHEL9/x86-64 -- check the site for the current version/link)
mkdir -p ~/mrsid_sdk && cd ~/mrsid_sdk
curl -L -o mrsid_dsdk.zip "<url from the developer page>"
unzip mrsid_dsdk.zip
chmod +x MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode   # ships without the executable bit

# Decode a NAIP tile (or just the window you need -- see --coord geo below)
export LD_LIBRARY_PATH=~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/lib:~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin:$LD_LIBRARY_PATH
~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode \
    -i data/raw/naip/<tile>/<tile>.sid -o /tmp/naip_check.tif -of tifg
```

You don't need to run `mrsiddecode` by hand for the actual pipeline -- `scripts/build_feature_stack.py` (step 5) calls it for you via `--mrsiddecode-bin`, decoding just the window it needs rather than the whole (often 5-10GB) tile. This manual step above is just to confirm the SDK is working before relying on it.

Notes if this trips you up:
- The tool is named `mrsiddecode` in the Decode SDK, not `mrsidgeodecode` (that name shows up in some LizardTech docs for GeoExpress, a different/commercial product bundling a differently-named CLI).
- `-of tifg` gives a georeferenced GeoTIFF; plain `-of tif` doesn't carry the CRS/transform.
- NAIP band order is R, G, B, Infrared (confirmed from each tile's own `.xml` metadata, not assumed) plus sometimes a 5th alpha/validity band.

---

### 2. Merge DEM tiles

Needed because several feature channels (flow direction/accumulation, openness) require full-watershed context, not per-tile.

```bash
uv run python scripts/merge_dem.py --dataset altarvalley
# or, for a study area not in the pre-registered list:
uv run python scripts/merge_dem.py --dem-dir data/raw/dem/some_area --out data/processed/dem/SomeAreaMerged.tif
```

**Gotcha that will bite you if you touch this file with another tool:** the merged GeoTIFF must NOT be written with a floating-point LZW predictor. GDAL/rasterio read predictor=3 fine, but WhiteboxTools' own Rust GeoTIFF reader panics on it -- *silently*, meaning its Python wrapper still returns exit code 0 and every downstream step in `build_feature_stack.py` will quietly produce empty output. `merge_dem.py` already avoids this; just don't add a predictor if you re-merge by hand.

---

### 3. Labels

Cleaned and reprojected label shapefiles are already committed to this repo under `data/raw/labels/`.

```bash
# Step 1 -- snap lines to the DEM crest (corrects manual digitization offset)
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley_structures

# Step 2 -- buffer snapped lines into polygon masks (N metres each side)
uv run python scripts/buffer_labels.py --dataset altarvalley --buffer 5
uv run python scripts/buffer_labels.py --dataset altarvalley_structures --buffer 5
```

Available datasets: `safford`, `cochise`, `bigchino`, `uppergila`, `altarvalley`, `altarvalley_structures`. The buffer width is a real modeling choice worth comparing (2m/5m/10m all exist for Altar Valley in `data/processed/labels_buffered/`) -- 5m is what the current model configs use.

If you have new raw label files from the shared project folder:

```bash
uv run python scripts/prepare_labels.py --source-dir "/path/to/shared/BermIdentification/02 ExistingDatasets"
```

---

### 4. Rasterize labels onto the merged grid

`scripts/rasterize_labels.py` normally rasterizes per original DEM tile, but training now uses one merged raster per study area, so point it at a directory containing just the merged DEM:

```bash
mkdir -p /tmp/merged_dem_only && ln -s "$PWD/data/processed/dem/AltarValleyMerged.tif" /tmp/merged_dem_only/
uv run python scripts/rasterize_labels.py --dataset altarvalley --buffer 5 --dem-dir /tmp/merged_dem_only
```

Output: `data/processed/masks/altarvalley_buf5m/AltarValleyMerged_mask.tif` -- same grid as the merged DEM, values `0`=background, `1`=berm, `255`=ignore (DEM nodata region).

---

### 5. Build the feature stack

`scripts/build_feature_stack.py` computes the full 12-channel plan and writes one multiband GeoTIFF + a normalization-stats JSON:

| Group | Channels |
|---|---|
| Terrain | residual relief (15m, 45m windows), positive openness, profile curvature, multidirectional hillshade, depression depth |
| Hydrologic | flow-orthogonality (Ω), log(1 + flow accumulation) |
| Optical (NAIP) | raw R, G, B, SAVI anomaly |

```bash
uv run python scripts/build_feature_stack.py \
    --dataset altarvalley \
    --naip data/raw/naip/<tile>/<tile>.sid \
    --mrsiddecode-bin ~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode
```

Output goes to `data/processed/features/<dataset>/stack.tif` (12 bands, each tagged with its channel name -- `rasterio`'s `descriptions`) and `norm_stats.json` (per-channel 1st/99th percentile clip bounds + mean/std, computed over the *whole study area*, not per-tile -- per-tile normalization would destroy the absolute-magnitude information that distinguishes a real 1m berm from a 10cm ripple).

Every step is idempotent -- if you already have some of the intermediate rasters (say, from a previous run), it skips straight to whatever's missing. Positive openness has no WhiteboxTools equivalent in the free/open-source build (it's a paid-extension tool there), so it's reimplemented from scratch in `src/utils/openness.py`, GPU-accelerated via torch since it's an expensive 8-direction search over the whole raster.

Want fewer channels? `--channels resid15 openness omega savi_anomaly` builds just the 4-channel core (no NAIP needed). Which channels an actual *training run* uses is a separate, later choice -- see Training below -- so you generally want to build the full 12-channel stack once and pick subsets of it per config, rather than rebuilding a smaller stack file.

---

### 6. (Optional) Hyperparameter sweep

Before committing to a full run (which can take an hour+), scan lr/batch_size/loss on a small, high-berm-density slice of the raster:

```bash
uv run python scripts/sweep_hyperparams.py --config configs/train_altarvalley.yaml --device cuda:2 \
    --row-start 58000 --rows 12000 --epochs 8
```

**Picking `--row-start` matters.** Berms aren't distributed evenly across the raster -- check where they actually are before picking a slice, or you'll silently sweep over a region with zero labeled positives (IoU is then mathematically stuck at 0 regardless of hyperparameters, which looks like "nothing works" but is actually "there was nothing to measure"). The script warns if either the train or val slice it computes has zero berm pixels, but that warning only fires after you've already picked `--row-start`:

```python
import rasterio, numpy as np
with rasterio.open("data/processed/masks/altarvalley_buf5m/AltarValleyMerged_mask.tif") as src:
    counts = (src.read(1) == 1).sum(axis=1)
print(np.where(counts > 0)[0].min(), np.where(counts > 0)[0].max())  # row range with any berm pixels
```

Results (ranked by val IoU) print to the console and save to `outputs/sweep/results.csv`.

---

### 7. Train

```bash
uv run python scripts/train.py --config configs/train_altarvalley.yaml --device cuda
uv run python scripts/train.py --config configs/train_altarvalley.yaml --device mps
uv run python scripts/train.py --config configs/train_altarvalley.yaml --resume outputs/checkpoints/altarvalley/latest.pth
```

**Which channels a run uses is entirely a YAML choice, not a code choice.** `configs/train_altarvalley.yaml` has a `data.channels` list; `in_channels` and, if using ImageNet weights, which channels get the pretrained R/G/B first-conv slots are both derived from that list automatically. To try a different channel subset, copy the YAML and edit the list -- see `configs/train_altarvalley_core4.yaml` for the 4-channel-core example. Don't touch `train.py`, `dataset.py`, or `unet.py` for this.

If `data.channels` doesn't include all of `red`, `green`, `blue`, `encoder_weights: imagenet` is ignored and the encoder trains from random init instead -- ImageNet pretraining doesn't transfer meaningfully onto channels with no raw RGB in them.

The train/val split is spatial (by row range within the merged raster, not a list of tiles), and validation sampling oversamples toward berm pixels the same way training does (`pos_fraction`) -- pure-random validation crops essentially never contain a berm given how rare and thin they are, which otherwise makes val IoU stuck at 0 regardless of the model (this bit us during the hyperparameter sweep; fixed in both `train.py` and `sweep_hyperparams.py`).

Checkpoints go to `output.checkpoint_dir` in the config (`latest.pth` every epoch, `best.pth` on new best val IoU -- use `best.pth`, since val performance can plateau or regress well before training loss does).

---

## Project Structure

```
berm-detection/
├── data/
│   ├── raw/
│   │   ├── dem/                  # USGS 1m DEM GeoTIFFs — not in git, download above
│   │   ├── naip/                 # NAIP 4-band imagery (.sid) — not in git, download above
│   │   ├── labels/               # Cleaned berm polyline shapefiles (in git)
│   │   └── labels_snapped/       # DEM-snapped centrelines (in git)
│   └── processed/
│       ├── dem/                  # Per-study-area merged DEM mosaics — not in git
│       ├── labels_buffered/      # Polygon training masks, 2m/5m/10m (in git)
│       ├── masks/                # Rasterized binary masks on the merged grid — not in git
│       └── features/             # Multiband feature stacks + norm_stats.json (stats in git, rasters not)
├── notebooks/                     # Exploratory analysis and label/DEM QC
├── scripts/
│   ├── prepare_labels.py         # Reproject + QC raw label shapefiles
│   ├── download_dem.py           # Bulk DEM download from URL list
│   ├── download_dem_api.py       # DEM download via USGS TNM API
│   ├── dem_urls.txt              # Pre-built URL list for Safford AOI
│   ├── snap_labels_to_dem.py     # Snap polylines to DEM ridge
│   ├── buffer_labels.py          # Buffer lines into polygon masks
│   ├── merge_dem.py              # Merge DEM tiles into one per-study-area mosaic
│   ├── rasterize_labels.py       # Rasterize buffered polygons onto a DEM grid
│   ├── build_feature_stack.py    # Build the 12-channel terrain/hydro/NAIP feature stack
│   ├── sweep_hyperparams.py      # Small-scale lr/batch_size/loss sweep
│   └── train.py                  # Train the U-Net
├── src/
│   ├── data/                     # BermDataset (loads the feature stack + mask)
│   ├── models/                   # U-Net (with ImageNet first-conv inflation onto raw RGB)
│   ├── training/                 # Trainer, loss functions
│   └── utils/                    # openness.py, flow_orthogonality.py (from-scratch terrain features)
├── configs/                       # Training configs (YAML) -- channel selection lives here
└── tests/
```

---

## References

- Li et al. (2023) — U-Net + OBIA for check dam detection, Yellow River basin
- Xia & Tonooka (2024) — DL-based earthwork detection
- D-LinkNet + clDice loss for topology-aware linear feature segmentation
- Yokoyama, Kikuchi & Ohuchi (2002) — topographic openness

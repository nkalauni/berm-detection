# Berm Detection

Deep learning pipeline for automated detection and mapping of berms and legacy earthworks in dryland watersheds using U-Net semantic segmentation on LiDAR-derived DEMs and NAIP aerial imagery.

**Study areas:** Altar Valley, Safford, Big Chino, Cochise, Upper Gila (Arizona)

Results and findings from actual runs are tracked in [`docs/experiments.md`](docs/experiments.md) — check there before re-running something that's already been tried.

## Setup

```bash
git clone <repo-url>
cd berm-detection
uv sync                              # creates .venv, installs deps
uv run python scripts/download_dem_api.py --help   # prefix any script with `uv run`
```

## Pipeline

```
download DEM + NAIP  →  merge DEM tiles  →  buffer + rasterize labels
→  build feature stack  →  (optional) hyperparameter sweep  →  train  →  evaluate
```

### 1. Download DEM

```bash
# From a label shapefile's bounding box
uv run python scripts/download_dem_api.py \
    --shp data/raw/labels/altarvalley_longberms.shp \
    --out data/raw/dem/altar_valley

# Or a bounding box directly
uv run python scripts/download_dem_api.py --bbox -111.6,31.4,-111.0,32.2 --out data/raw/dem/altar_valley

# Safford has a pre-built URL list instead:
uv run python scripts/download_dem.py
```

1m resolution, float32, EPSG:26912, nodata `-999999.0`.

### 2. Download NAIP

Manual download from the [USDA NAIP gateway](https://helpcenter.agvance.net/home/download-naip-imagery) → `data/raw/naip/`. Match county FIPS to LiDAR year:

| Study Area | County | FIPS | LiDAR Year |
|---|---|---|---|
| Altar Valley | Pima, AZ | 04019 | 2018–2021 |
| Safford | Graham, AZ | 04009 | 2016 |
| Cochise | Cochise, AZ | 04003 | 2020 |

**NAIP ships as MrSID (`.sid`) — this needs one manual setup step, or it'll look like a corrupt-file error.** Open-source GDAL/rasterio can't read `.sid` at all (the decode driver is proprietary and excluded from open builds), so `rasterio.open()` fails with a generic "not recognized" error that looks like corruption but isn't. Fix once, free, no registration:

```bash
mkdir -p ~/mrsid_sdk && cd ~/mrsid_sdk
curl -L -o sdk.zip "<current SDK url from https://www.lizardtech.com/developer>"
unzip sdk.zip
chmod +x MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode   # ships without the executable bit

# sanity check:
export LD_LIBRARY_PATH=~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/lib:~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin:$LD_LIBRARY_PATH
~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode -i data/raw/naip/<tile>/<tile>.sid -o /tmp/check.tif -of tifg
```

The pipeline calls `mrsiddecode` for you (`--mrsiddecode-bin` in step 5) — this is just to confirm the SDK works first. Two things that'll trip you up if you skip this note: the tool is named `mrsiddecode` here, not `mrsidgeodecode` (a different LizardTech product uses that name); and NAIP band order is R, G, B, Infrared (confirmed per-tile from each `.xml`, not assumed).

### 3. Merge DEM tiles

```bash
uv run python scripts/merge_dem.py --dataset altarvalley
# other areas:
uv run python scripts/merge_dem.py --dem-dir data/raw/dem/some_area --out data/processed/dem/SomeAreaMerged.tif
```

Needed because flow direction/accumulation and openness require full-watershed context, not per-tile.

**Don't add a floating-point predictor if you re-merge by hand.** GDAL/rasterio read `predictor=3` fine; WhiteboxTools' own GeoTIFF reader panics on it *silently* (its Python wrapper still returns exit code 0), so every downstream step quietly produces empty output. `merge_dem.py` already avoids this.

### 4. Labels

```bash
# snap digitized lines to the DEM crest
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley_structures

# buffer into polygon masks (metres each side)
uv run python scripts/buffer_labels.py --dataset altarvalley --buffer 5
uv run python scripts/buffer_labels.py --dataset altarvalley_structures --buffer 5
```

Datasets: `safford`, `cochise`, `bigchino`, `uppergila`, `altarvalley`, `altarvalley_structures`. Altar Valley has two label sets (long berm centrelines + shorter structure features) — use both, see step 5. Buffer width is a real modeling choice; see `docs/experiments.md` for the 2–10m comparison (5m and 8m are both reasonable, 2–3m measurably worse).

New raw labels from the shared project folder:

```bash
uv run python scripts/prepare_labels.py --source-dir "/path/to/shared/BermIdentification/02 ExistingDatasets"
```

### 5. Rasterize labels onto the merged grid

```bash
mkdir -p /tmp/merged_dem_only && ln -s "$PWD/data/processed/dem/AltarValleyMerged.tif" /tmp/merged_dem_only/
uv run python scripts/rasterize_labels.py --dataset altarvalley_combined --buffer 5 --dem-dir /tmp/merged_dem_only
```

Use `altarvalley_combined`, not `altarvalley` — the latter only rasterizes the long-berms shapefile and silently drops the structures labels (a real bug that cost real accuracy, see `docs/experiments.md` finding #2). `altarvalley_combined`'s shapefile list is defined in `rasterize_labels.py:DATASETS`.

Output: `data/processed/masks/altarvalley_combined_buf5m/AltarValleyMerged_mask.tif` — `0`=background, `1`=berm, `255`=ignore (DEM nodata).

### 6. Build the feature stack

```bash
uv run python scripts/build_feature_stack.py \
    --dataset altarvalley \
    --naip data/raw/naip/<tile>/<tile>.sid \
    --mrsiddecode-bin ~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode
```

Produces `data/processed/features/<dataset>/stack.tif` (12 bands, named via `rasterio` band descriptions) + `norm_stats.json` (whole-area 1st/99th percentile + mean/std — not per-tile, which would erase the absolute-magnitude difference between a real 1m berm and a 10cm ripple). Idempotent — reruns skip whatever's already built.

| Group | Channels |
|---|---|
| Terrain | resid15, resid45, openness, profile_curvature, multidirectional_hillshade, depression_depth |
| Hydrologic | omega (flow-orthogonality), log_flowacc |
| Optical (NAIP) | red, green, blue, savi_anomaly |

Fewer channels: `--channels resid15 openness omega savi_anomaly` (no NAIP needed). Which channels a *training run* uses is a separate, later choice (step 8) — build the full stack once, select subsets per config.

### 7. (Optional) Hyperparameter sweep

```bash
uv run python scripts/sweep_hyperparams.py --config configs/train_altarvalley.yaml --device cuda:2 \
    --row-start 58000 --rows 12000 --epochs 8
```

Check where the real berm pixels are before picking `--row-start`, or you'll sweep over an unlabeled region and get IoU=0 everywhere (looks like "nothing works," is actually "nothing to measure"):

```python
import rasterio, numpy as np
with rasterio.open("data/processed/masks/altarvalley_combined_buf5m/AltarValleyMerged_mask.tif") as src:
    counts = (src.read(1) == 1).sum(axis=1)
print(np.where(counts > 0)[0].min(), np.where(counts > 0)[0].max())
```

Results → console + `outputs/sweep/results.csv`.

### 8. Train

```bash
uv run python scripts/train.py --config configs/train_altarvalley.yaml --device cuda
uv run python scripts/train.py --config configs/train_altarvalley.yaml --resume outputs/checkpoints/altarvalley_combined/latest.pth
```

Channel selection is a YAML edit, not a code change: `data.channels` in the config drives `in_channels` and (if using ImageNet weights) which channels get the pretrained R/G/B first-conv slots, both derived automatically. Compare `configs/train_altarvalley.yaml` (12ch) vs `configs/train_altarvalley_core4.yaml` (4ch) — same stack file, different subset. `encoder_weights: imagenet` is ignored (falls back to random init, with a printed note) unless `red`/`green`/`blue` are all in the channel list.

Checkpoints: `latest.pth` every epoch, `best.pth` on new best val IoU (use `best.pth` — val performance can plateau/regress before train loss does). `manifest.json` alongside them records exactly what was used (channels, architecture, row ranges) so evaluation doesn't depend on the config file staying unchanged.

### 9. Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint-dir outputs/checkpoints/altarvalley_combined --device cuda
uv run python scripts/plot_diagnostics.py --checkpoint-dir outputs/checkpoints/altarvalley_combined --device cuda
```

`evaluate.py` runs full tiled inference over the val region (not oversampled crops — training's own reported val_iou reads much higher than true full-coverage performance, don't use it to compare runs) and writes a prediction raster, a TP/FP/FN/TN confusion raster, and a threshold sweep. `plot_diagnostics.py` turns those into: a spatial confusion map over hillshade, a per-region IoU heatmap, a precision/recall curve, best/typical/worst and false-positive crop galleries, channel-sensitivity (IoU drop when each channel is zeroed), and training curves.

## Project Structure

```
berm-detection/
├── data/
│   ├── raw/{dem,naip,labels,labels_snapped}/   # naip/dem not in git, see steps 1-2
│   └── processed/
│       ├── dem/              # merged mosaics — not in git
│       ├── labels_buffered/  # polygon masks, various widths (in git)
│       ├── masks/            # rasterized binary masks — not in git
│       └── features/         # feature stacks + norm_stats.json (stats in git, rasters not)
├── notebooks/                 # exploratory analysis, label/DEM QC
├── scripts/                   # one script per pipeline step, see above
├── src/{data,models,training,utils}/
├── configs/                   # training configs — channel selection lives here
├── docs/experiments.md        # results log, keep this updated
└── outputs/                    # checkpoints, sweep results, eval diagnostics — not in git
```

## References

- Li et al. (2023) — U-Net + OBIA for check dam detection, Yellow River basin
- Xia & Tonooka (2024) — DL-based earthwork detection
- D-LinkNet + clDice loss for topology-aware linear feature segmentation
- Yokoyama, Kikuchi & Ohuchi (2002) — topographic openness

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

## Data

Large rasters (DEMs, NAIP) are not tracked in git. After cloning, download them using the steps below. Shapefiles (labels, snapped lines, polygon masks) are already included in the repo under `data/`.

### 1. DEM — 1m LiDAR-derived elevation tiles (USGS)

There are two download methods depending on the study area.

**Method A — API download (recommended for any area)**

Queries the USGS TNM API automatically from a shapefile bounding box:

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

Options:
```
--shp        Shapefile to derive bounding box from (mutually exclusive with --bbox)
--bbox       minlon,minlat,maxlon,maxlat in WGS84
--out        Output directory (default: data/raw/dem/)
--workers    Parallel download threads, default 3
--dry-run    List matching tiles without downloading
```

**Method B — URL list download (Safford AOI pre-configured)**

`scripts/dem_urls.txt` contains direct download URLs for the Safford study area tiles:

```bash
uv run python scripts/download_dem.py
# tiles go to data/raw/dem/ by default
```

Options:
```
--urls       Path to URL list (default: scripts/dem_urls.txt)
--out        Output directory (default: data/raw/dem/)
--workers    Parallel download threads, default 2
--dry-run    Print URLs without downloading
```

To use this method for other areas, generate a URL list from the [USGS National Map Downloader](https://apps.nationalmap.gov/downloader/) and pass it with `--urls`.

**Expected DEM layout after download:**

```
data/raw/dem/
├── altar_valley/          # Altar Valley tiles
│   ├── USGS_1M_12_x44y349_AZ_CochiseCounty_2020_B20.tif
│   └── ...
├── USGS_one_meter_x59y363_AZ_Safford_QL2_2016.tif   # Safford tiles (flat)
└── ...
```

All tiles are 1m resolution, float32, EPSG:26912, nodata = `-999999.0` (a few older Safford tiles use `-3.4e+38`).

### 2. NAIP imagery — 1m aerial imagery, 4-band RGBNIR (USDA)

NAIP is downloaded manually. Use the county FIPS codes below to select the correct tiles. Match the NAIP year to the LiDAR collection year for temporal consistency.

| Study Area   | County       | FIPS  | LiDAR Year |
|--------------|--------------|-------|------------|
| Altar Valley | Pima, AZ     | 04019 | 2018–2021  |
| Safford      | Graham, AZ   | 04009 | 2016       |
| Cochise      | Cochise, AZ  | 04003 | 2020       |

Download from the [USDA NAIP gateway](https://helpcenter.agvance.net/home/download-naip-imagery), then place files in `data/raw/naip/`.

### 3. Labels — berm polyline shapefiles

Cleaned and reprojected label shapefiles are already committed to this repo under `data/raw/labels/`. No manual copy needed.

If you have new raw label files from the shared project folder, run the preparation script to reproject to EPSG:26912 and QC geometry:

```bash
uv run python scripts/prepare_labels.py \
    --source-dir "/path/to/shared/BermIdentification/02 ExistingDatasets"
```

---

## Label Processing Pipeline

Once DEMs and labels are in place, run the following steps in order to produce polygon training masks.

### Step 1 — Snap lines to DEM crest

Corrects manual digitization offsets by moving each vertex to the local ridge peak in a Relative Elevation Model (REM):

```bash
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley
uv run python scripts/snap_labels_to_dem.py --dataset altarvalley_structures
```

Output goes to `data/raw/labels_snapped/`. Each feature gets `snap_shift` (mean offset in metres), `snap_flags` (fraction of unsnapped vertices), and `needs_qc` (True if >50% vertices could not be snapped) attributes.

Available datasets: `safford`, `cochise`, `bigchino`, `uppergila`, `altarvalley`, `altarvalley_structures`

### Step 2 — Buffer lines into polygon masks

Applies a symmetric buffer around each snapped centreline to produce polygon shapefiles for use as binary training masks:

```bash
uv run python scripts/buffer_labels.py --dataset altarvalley --buffer 2
uv run python scripts/buffer_labels.py --dataset altarvalley_structures --buffer 2
# --buffer N means N metres each side (2N total width)
```

Output goes to `data/processed/labels_buffered/` with filenames encoding the buffer size (e.g. `altarvalley_longberms_snapped_buf2m.shp`).

---

## Project Structure

```
berm-detection/
├── data/
│   ├── raw/
│   │   ├── dem/                  # USGS 1m DEM GeoTIFFs — not in git, download above
│   │   ├── naip/                 # NAIP 4-band imagery — not in git, download above
│   │   ├── labels/               # Cleaned berm polyline shapefiles (in git)
│   │   └── labels_snapped/       # DEM-snapped centrelines (in git)
│   └── processed/
│       └── labels_buffered/      # Polygon training masks (in git)
├── notebooks/                    # Exploratory analysis
├── scripts/
│   ├── prepare_labels.py         # Reproject + QC raw label shapefiles
│   ├── download_dem.py           # Bulk DEM download from URL list
│   ├── download_dem_api.py       # DEM download via USGS TNM API
│   ├── dem_urls.txt              # Pre-built URL list for Safford AOI
│   ├── snap_labels_to_dem.py     # Snap polylines to DEM ridge
│   └── buffer_labels.py          # Buffer lines into polygon masks
├── src/
│   ├── data/                     # Dataset classes, transforms, tiling
│   ├── models/                   # U-Net and related architectures
│   ├── training/                 # Training loop, loss functions
│   └── utils/                    # Geo utilities, visualization
├── configs/                      # Training and data configs (YAML)
└── tests/
```

---

## Pipeline (planned)

1. **Download** DEM + NAIP (`download_dem_api.py`)
2. **Prepare labels** — reproject, QC (`prepare_labels.py`)
3. **Snap labels** — align to DEM ridge (`snap_labels_to_dem.py`)
4. **Buffer labels** — create polygon masks (`buffer_labels.py`)
5. **Tile** — generate 256×256 px patches with 32px overlap
6. **Train** — U-Net with ResNet-34 encoder, Dice + clDice loss
7. **Evaluate** — IoU, F1, precision, recall per study area
8. **Transfer** — test generalization across watersheds

---

## References

- Li et al. (2023) — U-Net + OBIA for check dam detection, Yellow River basin
- Xia & Tonooka (2024) — DL-based earthwork detection
- D-LinkNet + clDice loss for topology-aware linear feature segmentation

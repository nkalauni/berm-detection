# Berm Detection

Deep learning pipeline for automated detection and mapping of berms and legacy earthworks in dryland watersheds using U-Net semantic segmentation on LiDAR-derived DEMs and NAIP aerial imagery.

**Study areas:** Safford, Big Chino, Cochise, Upper Gila (Arizona)

---

## Setup

```bash
git clone <repo-url>
cd berm-detection
pip install -r requirements.txt
```

---

## Data

Data is not tracked in this repo. After cloning, download it using the steps below.

### 1. DEM — 1m LiDAR-derived elevation tiles (USGS)

The file `scripts/dem_urls.txt` contains 117 direct download URLs for the Safford study area, sourced from the [USGS National Map](https://apps.nationalmap.gov/downloader/).

```bash
python scripts/download_dem.py
```

Options:
```
--urls     Path to URL list (default: scripts/dem_urls.txt)
--out      Output directory (default: data/raw/dem/)
--workers  Parallel download threads, default 2 (keep low)
--dry-run  Print URLs without downloading
```

For other study areas, generate a new URL list from the USGS National Map downloader and pass it with `--urls`.

### 2. NAIP imagery — 1m aerial imagery, 4-band RGBNIR (USDA)

NAIP is downloaded manually through the USDA gateway. Use the county FIPS codes below to select the correct tiles. Match the NAIP year to the LiDAR collection year for your study area.

| Study Area | County       | FIPS  | LiDAR Year |
|------------|-------------|-------|------------|
| Safford    | Graham, AZ  | 04009 | 2017       |
| Cochise    | Cochise, AZ | 04003 | —          |
| Pima       | Pima, AZ    | 04019 | —          |

Download steps:
1. Go to the [USDA NAIP download page](https://helpcenter.agvance.net/home/download-naip-imagery)
2. Enter the county FIPS code
3. Download tiles and place them in `data/raw/naip/`

### 3. Labels — existing berm polylines

Copy the shapefiles from the shared project folder into `data/raw/labels/`:

```
data/raw/labels/
├── Safford/
├── BigChino/
├── CochiseReference/
└── UpperGila_Erosion_Control_Inventory/
```

Note: datasets use different coordinate systems. Reproject all to **EPSG:26912** (UTM Zone 12N) before use.

---

## Project Structure

```
berm-detection/
├── configs/            # Training and data configs (YAML)
├── data/               # NOT in git — download using scripts above
│   ├── raw/
│   │   ├── dem/        # USGS 1m DEM GeoTIFFs
│   │   ├── naip/       # NAIP 4-band imagery
│   │   └── labels/     # Berm polyline shapefiles
│   └── processed/      # Tiled patches ready for training
├── notebooks/          # Exploratory analysis
├── scripts/
│   ├── download_dem.py # Bulk DEM download
│   └── dem_urls.txt    # USGS tile URLs for Safford AOI
├── src/
│   ├── data/           # Dataset classes, transforms, tiling
│   ├── models/         # U-Net and related architectures
│   ├── training/       # Training loop, loss functions
│   └── utils/          # Geo utilities, visualization
└── tests/
```

---

## Pipeline (planned)

1. **Download** DEM + NAIP + labels
2. **Preprocess** — reproject to EPSG:26912, compute hillshade/slope/aspect from DEM
3. **Tile** — generate 256×256 px patches with 32px overlap
4. **Train** — U-Net with ResNet-34 encoder, Dice + clDice loss
5. **Evaluate** — IoU, F1, precision, recall per study area
6. **Transfer** — test generalization across watersheds

---

## References

- Li et al. (2023) — U-Net + OBIA for check dam detection, Yellow River basin
- Xia & Tonooka (2024) — DL-based earthwork detection
- D-LinkNet + clDice loss for topology-aware linear feature segmentation

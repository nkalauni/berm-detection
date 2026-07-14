"""
Rasterize buffered berm polygon shapefiles into binary mask GeoTIFFs.

For each DEM tile that overlaps the shapefile extent, produces a matching
mask GeoTIFF with the same transform, shape, and CRS.  Mask values:
  0   = background
  1   = berm (polygon interior)
  255 = nodata / ignore (DEM nodata pixels)

Usage:
    python scripts/rasterize_labels.py --dataset altarvalley
    python scripts/rasterize_labels.py --dataset altarvalley --buffer 2.0
    python scripts/rasterize_labels.py --dataset all --buffer 2.0 --dry-run
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset registry  —  keys match buffer_labels.py
# ---------------------------------------------------------------------------
DEM_BASE      = Path(__file__).parent.parent / "data" / "raw" / "dem"
BUF_BASE      = Path(__file__).parent.parent / "data" / "processed" / "labels_buffered"
MASKS_BASE    = Path(__file__).parent.parent / "data" / "processed" / "masks"

DATASETS = {
    "altarvalley":            (DEM_BASE / "altar_valley",
                               "altarvalley_longberms_snapped_buf{buf}m.shp"),
    "altarvalley_structures": (DEM_BASE / "altar_valley",
                               "altarvalley_structures_snapped_buf{buf}m.shp"),
    "safford":                (DEM_BASE,
                               "safford_berms_snapped_buf{buf}m.shp"),
    "cochise":                (DEM_BASE,
                               "cochise_berms_snapped_buf{buf}m.shp"),
    "bigchino":               (DEM_BASE,
                               "bigchino_berms_snapped_buf{buf}m.shp"),
    "uppergila":              (DEM_BASE,
                               "uppergila_berms_snapped_buf{buf}m.shp"),
}

DEM_NODATA_THRESHOLD = -999998.0  # values <= this are treated as nodata


def tile_overlaps(tile_path: Path, shp_bounds: tuple) -> bool:
    """Return True if tile extent intersects shp_bounds (xmin,ymin,xmax,ymax)."""
    with rasterio.open(tile_path) as src:
        b = src.bounds
    sx0, sy0, sx1, sy1 = shp_bounds
    return b.left < sx1 and b.right > sx0 and b.bottom < sy1 and b.top > sy0


def rasterize_tile(tile_path: Path, gdf: gpd.GeoDataFrame, out_path: Path) -> int:
    """
    Rasterize gdf polygons onto the grid of tile_path and write to out_path.
    Returns the number of berm pixels written.
    """
    with rasterio.open(tile_path) as src:
        transform = src.transform
        width     = src.width
        height    = src.height
        crs       = src.crs
        dem       = src.read(1)
        nodata    = src.nodata

    # Build nodata mask from DEM
    if nodata is not None:
        nodata_mask = (dem <= DEM_NODATA_THRESHOLD) | (dem == nodata)
    else:
        nodata_mask = dem <= DEM_NODATA_THRESHOLD

    # Reproject shapefile to tile CRS if needed
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Clip to tile extent to speed up rasterization
    tile_box = box(
        transform.c,
        transform.f + transform.e * height,
        transform.c + transform.a * width,
        transform.f,
    )
    gdf_clip = gdf[gdf.intersects(tile_box)]

    # Rasterize: 1 inside any polygon, 0 elsewhere
    if len(gdf_clip) == 0:
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        shapes = ((geom, 1) for geom in gdf_clip.geometry if geom is not None and not geom.is_empty)
        mask = rio_rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

    # Mark DEM nodata as ignore (255)
    mask[nodata_mask] = 255

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":   "GTiff",
        "dtype":    "uint8",
        "width":    width,
        "height":   height,
        "count":    1,
        "crs":      crs,
        "transform": transform,
        "nodata":   255,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return int(mask.sum())


def process_dataset(
    key: str,
    buffer_m: float,
    dem_dir_override: Path | None,
    out_dir_override: Path | None,
    dry_run: bool,
) -> dict:
    dem_dir_default, shp_template = DATASETS[key]
    dem_dir = dem_dir_override or dem_dir_default

    buf_int = int(buffer_m) if buffer_m == int(buffer_m) else buffer_m
    shp_name = shp_template.format(buf=buf_int)
    shp_path = BUF_BASE / shp_name

    if not shp_path.exists():
        print(f"\n[SKIP] Shapefile not found: {shp_path}")
        print("       Run scripts/buffer_labels.py first.")
        return {}

    if not dem_dir.exists():
        print(f"\n[SKIP] DEM directory not found: {dem_dir}")
        return {}

    print(f"\nDataset      : {key}")
    print(f"Shapefile    : {shp_path.name}")
    print(f"DEM dir      : {dem_dir}")

    gdf = gpd.read_file(shp_path)
    shp_bounds = tuple(gdf.to_crs("EPSG:26912").total_bounds)  # (xmin,ymin,xmax,ymax)

    tiles = sorted(dem_dir.glob("*.tif"))
    overlapping = [t for t in tiles if tile_overlaps(t, shp_bounds)]
    print(f"DEM tiles    : {len(tiles)} found, {len(overlapping)} overlap shapefile extent")

    if dry_run:
        for t in overlapping:
            print(f"  {t.name}")
        return {}

    out_dir = out_dir_override or (MASKS_BASE / f"{key}_buf{buf_int}m")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_berm_px = 0
    for tile_path in tqdm(overlapping, desc=f"Rasterizing {key}"):
        out_path = out_dir / f"{tile_path.stem}_mask.tif"
        n_berm_px += rasterize_tile(tile_path, gdf, out_path)

    print(f"Masks saved  : {out_dir}")
    print(f"Berm pixels  : {n_berm_px:,}")
    return {"dataset": key, "n_tiles": len(overlapping), "n_berm_px": n_berm_px}


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize buffered berm polygons into per-tile binary mask GeoTIFFs"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to process, or 'all'",
    )
    parser.add_argument(
        "--buffer", type=float, default=2.0, metavar="METRES",
        help="Buffer half-width used when building the shapefile (default: 2.0)",
    )
    parser.add_argument("--dem-dir",  type=Path, default=None,
                        help="Override DEM directory")
    parser.add_argument("--out-dir",  type=Path, default=None,
                        help="Override mask output directory")
    parser.add_argument("--dry-run",  action="store_true",
                        help="List overlapping tiles without writing masks")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    results = []
    for key in keys:
        r = process_dataset(key, args.buffer, args.dem_dir, args.out_dir, args.dry_run)
        if r:
            results.append(r)

    if results:
        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        for r in results:
            print(f"  {r['dataset']:<28} {r['n_tiles']:>4} tiles  "
                  f"{r['n_berm_px']:>12,} berm px")
        print(f"\nNext: run scripts/train.py to start model training.")


if __name__ == "__main__":
    main()

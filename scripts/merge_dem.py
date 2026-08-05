"""
Merge individual USGS DEM tiles into a single per-study-area mosaic.

Needed for hydrologic operations (flow direction/accumulation) that must run
on the full watershed extent rather than per-tile.

Usage:
    python scripts/merge_dem.py --dataset altarvalley
    python scripts/merge_dem.py --dem-dir data/raw/dem/altar_valley --out data/processed/dem/AltarValleyMerged.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

DEM_BASE = Path(__file__).parent.parent / "data" / "raw" / "dem"
OUT_BASE = Path(__file__).parent.parent / "data" / "processed" / "dem"
DEM_NODATA = -999999.0

# key -> (dem_dir, output filename)
DATASETS = {
    "altarvalley": (DEM_BASE / "altar_valley", "AltarValleyMerged.tif"),
}


def merge_dataset(dem_dir: Path, out_path: Path) -> None:
    tiles = sorted(dem_dir.glob("*.tif"))
    if not tiles:
        print(f"[SKIP] No .tif tiles found in {dem_dir}")
        return

    print(f"Merging {len(tiles)} tile(s) from {dem_dir}")
    srcs = [rasterio.open(t) for t in tiles]
    try:
        mosaic, transform = rio_merge(srcs, nodata=DEM_NODATA, method="first")
    finally:
        for s in srcs:
            s.close()

    crs = rasterio.open(tiles[0]).crs
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": mosaic.shape[2],
        "height": mosaic.shape[1],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": DEM_NODATA,
        "compress": "lzw",
        "tiled": True,
        "bigtiff": "YES",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    valid = mosaic[0] != DEM_NODATA
    print(f"Wrote {out_path}")
    print(f"  shape: {mosaic.shape[1]} x {mosaic.shape[2]}")
    print(f"  valid pixels: {valid.sum():,} / {valid.size:,} ({100 * valid.mean():.1f}%)")
    if valid.any():
        print(f"  elevation range: {mosaic[0][valid].min():.2f} to {mosaic[0][valid].max():.2f} m")
    else:
        print("  WARNING: no valid (non-nodata) pixels in merged output!")


def main():
    parser = argparse.ArgumentParser(description="Merge DEM tiles into a study-area mosaic")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=list(DATASETS.keys()), help="Pre-registered dataset")
    src.add_argument("--dem-dir", type=Path, help="Directory of .tif tiles to merge")
    parser.add_argument("--out", type=Path, default=None, help="Output path (required with --dem-dir)")
    args = parser.parse_args()

    if args.dataset:
        dem_dir, out_name = DATASETS[args.dataset]
        out_path = args.out or (OUT_BASE / out_name)
    else:
        dem_dir = args.dem_dir
        if args.out is None:
            parser.error("--out is required when using --dem-dir")
        out_path = args.out

    merge_dataset(dem_dir, out_path)


if __name__ == "__main__":
    main()

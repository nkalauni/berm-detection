"""
Build a road-buffer mask from OpenStreetMap data, on the same grid as a
study area's merged DEM. Used to test whether suppressing model
predictions that overlap a known road reduces the road-vs-berm false
positives seen in the diagnostic galleries (see docs/experiments.md).

Queries the public Overpass API for all highway=* ways within the DEM's
bounding box (reprojected to WGS84), buffers them, and rasterizes onto
the DEM's grid. Caches the raw Overpass response so re-runs (e.g. with a
different buffer width) don't re-query.

Usage:
    python scripts/build_road_mask.py --dataset altarvalley --buffer 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize as rio_rasterize

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.osm import query_overpass_highways, ways_to_geodataframe  # noqa: E402

DEM_BASE = Path(__file__).parent.parent / "data" / "processed" / "dem"
OUT_BASE = Path(__file__).parent.parent / "data" / "processed" / "roads"

DATASETS = {
    "altarvalley": DEM_BASE / "AltarValleyMerged.tif",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--buffer", type=float, default=5.0, metavar="METRES",
                         help="Half-width buffer around each road centreline")
    args = parser.parse_args()

    dem_path = DATASETS[args.dataset]
    with rasterio.open(dem_path) as src:
        transform, width, height, crs = src.transform, src.width, src.height, src.crs
        bounds = src.bounds

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs84.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = to_wgs84.transform(bounds.right, bounds.top)

    out_dir = OUT_BASE / args.dataset
    cache_path = out_dir / "osm_roads_raw.json"
    osm_data = query_overpass_highways((lon_min, lat_min, lon_max, lat_max), cache_path)

    gdf = ways_to_geodataframe(osm_data)
    print(f"Parsed {len(gdf)} road ways, types: {gdf['highway'].value_counts().to_dict()}")
    gdf = gdf.to_crs(crs)
    gdf["geometry"] = gdf.buffer(args.buffer)

    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    mask = rio_rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

    out_path = out_dir / f"road_mask_buf{int(args.buffer)}m.tif"
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": width, "height": height,
        "count": 1, "crs": crs, "transform": transform, "nodata": 255,
        "compress": "lzw", "tiled": True, "bigtiff": "YES",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"Wrote {out_path}")
    print(f"Road-buffer pixels: {int(mask.sum()):,} / {mask.size:,} ({100*mask.mean():.3f}%)")


if __name__ == "__main__":
    main()

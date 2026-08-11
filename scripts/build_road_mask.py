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
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import LineString

DEM_BASE = Path(__file__).parent.parent / "data" / "processed" / "dem"
OUT_BASE = Path(__file__).parent.parent / "data" / "processed" / "roads"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DATASETS = {
    "altarvalley": DEM_BASE / "AltarValleyMerged.tif",
}


def query_overpass(bbox_wgs84: tuple, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"Using cached Overpass response: {cache_path}")
        return json.loads(cache_path.read_text())

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    query = (
        "[out:json][timeout:120];\n"
        f'way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});\n'
        "out geom;"
    )
    print("Querying Overpass API...")
    t0 = time.time()
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=150)
    r.raise_for_status()
    data = r.json()
    print(f"  {len(data['elements'])} ways ({time.time()-t0:.0f}s)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


def ways_to_geodataframe(osm_data: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in osm_data["elements"]:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        rows.append({"highway": el.get("tags", {}).get("highway", "unknown"), "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


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
    osm_data = query_overpass((lon_min, lat_min, lon_max, lat_max), cache_path)

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

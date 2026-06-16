"""
Programmatically download 1m LiDAR DEM tiles from the USGS TNM API
for any study area, derived from a shapefile bounding box.

Usage:
    python scripts/download_dem_api.py --shp data/raw/labels/altarvalley_longberms.shp
    python scripts/download_dem_api.py --bbox -111.6,31.4,-111.0,32.2
    python scripts/download_dem_api.py --shp data/raw/labels/safford_berms.shp --dry-run
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import requests
from pyproj import Transformer
from tqdm import tqdm

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "raw" / "dem"


def bbox_from_shp(shp_path: Path, buffer_m: float = 500) -> tuple:
    """Return (minlon, minlat, maxlon, maxlat) from a shapefile with a small buffer."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.is_geographic:
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf
    else:
        gdf = gdf.buffer(buffer_m)
        gdf = gpd.GeoDataFrame(geometry=gdf, crs=gdf.crs if hasattr(gdf, "crs") else "EPSG:26912")

    gdf_geo = gdf.to_crs("EPSG:4326") if hasattr(gdf, "to_crs") else gdf
    xmin, ymin, xmax, ymax = gdf_geo.total_bounds
    return xmin, ymin, xmax, ymax


def query_tnm(bbox: tuple, resolution: str = "1 meter", max_items: int = 500) -> list:
    """Query TNM API and return list of (title, downloadURL)."""
    minlon, minlat, maxlon, maxlat = bbox
    params = {
        "datasets": f"Digital Elevation Model (DEM) {resolution}",
        "bbox": f"{minlon:.5f},{minlat:.5f},{maxlon:.5f},{maxlat:.5f}",
        "prodFormats": "GeoTIFF",
        "max": max_items,
        "outputFormat": "JSON",
    }
    r = requests.get(TNM_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    total = data.get("total", 0)
    items = data.get("items", [])
    if total > max_items:
        print(f"  WARNING: {total} products found but only {max_items} returned — increase --max")
    # deduplicate: same tile grid position appears in multiple LiDAR projects;
    # keep the most recent collection year (higher year in project name wins)
    import re
    seen: dict[str, tuple] = {}
    for item in items:
        fname = Path(item["downloadURL"]).stem  # e.g. USGS_1M_12_x44y349_AZ_BrawleyRillito_2018_D19
        # extract grid key: everything up to the project name (xNNyNNN)
        m = re.search(r"(x\d+y\d+)", fname)
        grid_key = m.group(1) if m else fname
        year_m = re.search(r"_(\d{4})_", fname)
        year = int(year_m.group(1)) if year_m else 0
        if grid_key not in seen or year > seen[grid_key][0]:
            seen[grid_key] = (year, item["title"], item["downloadURL"])
    deduped = [(title, url) for _, title, url in seen.values()]
    if len(deduped) < len(items):
        print(f"  Deduplicated {len(items)} → {len(deduped)} tiles (kept newest collection per grid cell)")
    return deduped


def download_tile(url: str, out_dir: Path, session: requests.Session) -> tuple:
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename
    if dest.exists():
        return filename, True, "skipped (already exists)"
    try:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=filename, leave=False,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        return filename, True, "downloaded"
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download USGS 1m DEM via TNM API")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--shp", type=Path, help="Shapefile to derive bounding box from")
    src.add_argument("--bbox", type=str, help="minlon,minlat,maxlon,maxlat in WGS84")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--resolution", default="1 meter", help="DEM resolution string")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max", type=int, default=500, dest="max_items")
    parser.add_argument("--dry-run", action="store_true", help="List tiles without downloading")
    args = parser.parse_args()

    # --- resolve bbox ---
    if args.shp:
        if not args.shp.exists():
            print(f"Shapefile not found: {args.shp}")
            sys.exit(1)
        bbox = bbox_from_shp(args.shp)
        print(f"AOI from {args.shp.name}: "
              f"lon {bbox[0]:.4f} to {bbox[2]:.4f}, lat {bbox[1]:.4f} to {bbox[3]:.4f}")
    else:
        bbox = tuple(float(x) for x in args.bbox.split(","))

    # --- query ---
    print(f"Querying TNM API for {args.resolution} DEM...")
    products = query_tnm(bbox, resolution=args.resolution, max_items=args.max_items)

    if not products:
        print(f"No {args.resolution} DEM products found for this area.")
        print("Try --resolution '3m' or '1/3 arc-second' if 1m coverage is unavailable.")
        sys.exit(0)

    print(f"Found {len(products)} tile(s):")
    for title, url in products:
        size_info = ""
        print(f"  {Path(url).name}")

    if args.dry_run:
        return

    # --- download ---
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []

    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_tile, url, args.out, session): url
                   for _, url in products}
        with tqdm(total=len(products), desc="Total tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                filename, ok, msg = future.result()
                pbar.set_postfix_str(f"{filename}: {msg}")
                pbar.update(1)
                if not ok:
                    failed.append((filename, msg))

    print(f"\nDone. {len(products) - len(failed)}/{len(products)} tiles saved to {args.out}")
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

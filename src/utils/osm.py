"""Shared OpenStreetMap fetch/parse helpers, used by build_road_mask.py and
build_feature_stack.py's dist_to_road channel."""

import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def query_overpass_highways(bbox_wgs84: tuple, cache_path: Path) -> dict:
    """bbox_wgs84: (lon_min, lat_min, lon_max, lat_max). Caches the raw
    response so repeat calls (e.g. building multiple channels) don't
    re-query."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    query = (
        "[out:json][timeout:120];\n"
        f'way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});\n'
        "out geom;"
    )
    t0 = time.time()
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=150)
    r.raise_for_status()
    data = r.json()
    print(f"  Overpass: {len(data['elements'])} ways ({time.time()-t0:.0f}s)", flush=True)
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

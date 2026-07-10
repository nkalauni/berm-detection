"""
Buffer snapped berm polylines into polygon training masks.

Reads snapped line shapefiles from data/raw/labels_snapped/ and applies a
configurable symmetric buffer to produce polygon shapefiles. The output
polygons can be rasterized directly as binary training masks for the berm
detection model.

The buffer is applied in projected metres (EPSG:26912), so --buffer 3 means
3 m either side of the centreline (6 m total width).

Usage:
    python scripts/buffer_labels.py --dataset altarvalley
    python scripts/buffer_labels.py --dataset altarvalley --buffer 3.0
    python scripts/buffer_labels.py --dataset all
    python scripts/buffer_labels.py --dataset all --buffer 2.5 --dissolve
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BUFFER_M = 3.0   # metres each side of the centreline (half-width)
CAP_STYLE_FLAT   = 2     # shapely cap_style: 1=round, 2=flat, 3=square
JOIN_STYLE_ROUND = 1     # shapely join_style: 1=round, 2=mitre, 3=bevel

# ---------------------------------------------------------------------------
# Dataset registry  —  keys and snapped stems match snap_labels_to_dem.py
# ---------------------------------------------------------------------------
SNAPPED_DIR = Path(__file__).parent.parent / "data" / "raw" / "labels_snapped"
OUT_BASE    = Path(__file__).parent.parent / "data" / "processed" / "labels_buffered"

# dataset key → snapped shapefile stem (produced by snap_labels_to_dem.py)
DATASETS = {
    "safford":                "safford_berms_snapped",
    "cochise":                "cochise_berms_snapped",
    "bigchino":               "bigchino_berms_snapped",
    "uppergila":              "uppergila_berms_snapped",
    "altarvalley":            "altarvalley_longberms_snapped",
    "altarvalley_structures": "altarvalley_structures_snapped",
}


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def buffer_geodataframe(
    gdf: gpd.GeoDataFrame,
    buffer_m: float,
    cap_style: int,
    join_style: int,
) -> gpd.GeoDataFrame:
    """
    Apply a symmetric buffer to every geometry and return a new GeoDataFrame
    with Polygon / MultiPolygon geometries.  Non-polygon results (e.g. from
    degenerate lines) are dropped with a warning.
    """
    buffered_geoms = gdf.geometry.buffer(
        buffer_m,
        cap_style=cap_style,
        join_style=join_style,
    )

    gdf_out = gdf.copy()
    gdf_out["geometry"] = buffered_geoms

    # drop any rows that did not produce a valid polygon (e.g. point-like lines)
    valid_mask = gdf_out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  [!] {n_dropped} feature(s) did not produce a polygon — dropped")
    gdf_out = gdf_out[valid_mask].reset_index(drop=True)

    return gdf_out


def process_dataset(
    key: str,
    snapped_dir: Path,
    out_dir: Path,
    buffer_m: float,
    cap_style: int,
    join_style: int,
    dissolve: bool,
    dry_run: bool,
) -> dict:
    stem = DATASETS[key]
    in_path = snapped_dir / f"{stem}.shp"

    if not in_path.exists():
        print(f"\n[SKIP] Snapped file not found: {in_path}")
        print("       Run scripts/snap_labels_to_dem.py first.")
        return {}

    print(f"\nLoading  : {in_path.name}")
    gdf = gpd.read_file(in_path)

    if dry_run:
        gdf = gdf.head(10)
        print("  [dry-run] processing first 10 features only")

    print(f"  {len(gdf)} features, CRS: {gdf.crs}")

    # --- buffer ---
    print(f"  Buffering {buffer_m}m each side ...")
    gdf_poly = buffer_geodataframe(gdf, buffer_m, cap_style, join_style)
    gdf_poly["buffer_m"] = buffer_m   # record parameter used

    # --- optional dissolve ---
    if dissolve:
        print("  Dissolving overlapping polygons ...")
        dissolved = gdf_poly.dissolve().reset_index(drop=True)
        # dissolve loses per-feature attributes; keep only geometry + buffer_m
        dissolved["buffer_m"] = buffer_m
        gdf_poly = dissolved

    # --- area stats ---
    areas_m2 = gdf_poly.geometry.area
    print(f"  Polygon area (m²): min={areas_m2.min():.1f}  "
          f"max={areas_m2.max():.1f}  mean={areas_m2.mean():.1f}  "
          f"total={areas_m2.sum()/1e4:.2f} ha")

    # --- save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_buf{int(buffer_m)}m"
    if dissolve:
        suffix += "_dissolved"
    out_name = f"{stem}{suffix}.shp"
    out_path = out_dir / out_name
    gdf_poly.to_file(out_path)
    print(f"  Saved  : {out_path}")

    return {
        "dataset":    key,
        "n_in":       len(gdf),
        "n_out":      len(gdf_poly),
        "buffer_m":   buffer_m,
        "total_ha":   round(areas_m2.sum() / 1e4, 3),
        "out_file":   out_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Buffer snapped berm polylines into polygon training masks"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to process, or 'all' to run every registered dataset",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=DEFAULT_BUFFER_M,
        metavar="METRES",
        help=f"Buffer half-width in metres, applied each side of the centreline "
             f"(default: {DEFAULT_BUFFER_M})",
    )
    parser.add_argument(
        "--cap-style",
        type=int,
        default=CAP_STYLE_FLAT,
        choices=[1, 2, 3],
        metavar="{1|2|3}",
        help="Line end-cap style: 1=round, 2=flat (default), 3=square",
    )
    parser.add_argument(
        "--join-style",
        type=int,
        default=JOIN_STYLE_ROUND,
        choices=[1, 2, 3],
        metavar="{1|2|3}",
        help="Corner join style: 1=round (default), 2=mitre, 3=bevel",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="Dissolve overlapping polygons into a single geometry per dataset",
    )
    parser.add_argument(
        "--snapped-dir",
        type=Path,
        default=SNAPPED_DIR,
        help="Directory containing snapped shapefiles (default: data/raw/labels_snapped)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_BASE,
        help="Output directory (default: data/processed/labels_buffered)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 10 features per dataset",
    )
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    results = []
    for key in keys:
        result = process_dataset(
            key=key,
            snapped_dir=args.snapped_dir,
            out_dir=args.out_dir,
            buffer_m=args.buffer,
            cap_style=args.cap_style,
            join_style=args.join_style,
            dissolve=args.dissolve,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    # --- summary ---
    if results:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  {'Dataset':<28} {'n_in':>5} {'n_out':>5} {'buf_m':>5} {'total_ha':>9}")
        print(f"  {'-'*28} {'-'*5} {'-'*5} {'-'*5} {'-'*9}")
        for r in results:
            print(f"  {r['dataset']:<28} {r['n_in']:>5} {r['n_out']:>5} "
                  f"{r['buffer_m']:>5.1f} {r['total_ha']:>9.3f}")
        print(f"\nOutput: {args.out_dir}")
        print("Next: rasterize these polygons against a DEM tile grid to produce "
              "binary training masks (1=berm, 0=background).")


if __name__ == "__main__":
    main()

"""
Reproject all berm label shapefiles to a common CRS (EPSG:26912) and run
geometry QC checks. Cleaned shapefiles are written to data/raw/labels/.

Usage:
    python scripts/prepare_labels.py
    python scripts/prepare_labels.py --source-dir /path/to/shared/folder
    python scripts/prepare_labels.py --min-length 5 --verbose
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_CRS = "EPSG:26912"
MIN_LENGTH_M = 10  # polylines shorter than this are flagged as suspected artifacts

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: (relative path from source_dir, output name, geometry type expected)
DATASETS = [
    # --- primary training candidates ---
    (
        "miscellaneous berms/Safford/Safford.shp",
        "safford_berms",
        "LineString",
    ),
    (
        "miscellaneous berms/BigChino/HX_022423_big_chino_reference_addon.shp",
        "bigchino_berms",
        "LineString",
    ),
    (
        "miscellaneous berms/CochiseReference/Haiqing_CochiseReference.shp",
        "cochise_berms",
        "LineString",
    ),
    (
        "miscellaneous berms/UpperGila_Erosion_Control_Inventory/UpperGilaRiverBasin_lines.shp",
        "uppergila_berms",
        "LineString",
    ),
    # --- Altar Valley (new, described as cleanest) ---
    (
        "AltarValley_USDA-ARS/AltarValley_USDA-ARS/LongBerms_Imagery.shp",
        "altarvalley_longberms",
        "LineString",
    ),
    (
        "AltarValley_USDA-ARS/AltarValley_USDA-ARS/Structures_Imagery.shp",
        "altarvalley_structures",
        "LineString",
    ),
    # --- point datasets (stock tanks / earthworks) ---
    (
        "miscellaneous berms/UpperGila_Erosion_Control_Inventory/UpperGilaRiverBasin_pts.shp",
        "uppergila_stocktanks",
        "Point",
    ),
    (
        "AltarValley_USDA-ARS/AltarValley_USDA-ARS/BrawleyWash_Hillshade.shp",
        "brawleywash_pts",
        "Point",
    ),
    (
        "AltarValley_USDA-ARS/AltarValley_USDA-ARS/RioDeLaConcepcion_Hillshade.shp",
        "riodelaconception_pts",
        "Point",
    ),
    # --- known problem flags (BigChino) ---
    (
        "miscellaneous berms/BigChino/HX_022423_big_chino_reference_problems.shp",
        "bigchino_problems",
        "Point",
    ),
]


# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------

def qc_geometry(gdf: gpd.GeoDataFrame, geom_type: str, min_length: float) -> dict:
    """Run geometry checks and return a dict of flagged row indices."""
    flags = {
        "empty": [],
        "invalid": [],
        "wrong_type": [],
        "short": [],       # polylines only
        "coord_error": [], # infinite or null coordinates
    }

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            flags["empty"].append(idx)
            continue

        if not geom.is_valid:
            flags["invalid"].append(idx)

        # geometry type check (allow Multi variants)
        actual = geom.geom_type
        if geom_type not in actual:
            flags["wrong_type"].append(idx)

        # short segment check for lines
        if geom_type == "LineString" and hasattr(geom, "length"):
            if geom.length < min_length:
                flags["short"].append(idx)

        # infinite coordinate check
        try:
            bounds = geom.bounds
            if any(abs(v) == float("inf") or v != v for v in bounds):
                flags["coord_error"].append(idx)
        except Exception:
            flags["coord_error"].append(idx)

    return flags


def fix_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attempt to repair invalid geometries using shapely make_valid."""
    mask = ~gdf.geometry.is_valid
    if mask.any():
        gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(make_valid)
    return gdf


def print_qc_report(name: str, original_crs, gdf_raw: gpd.GeoDataFrame,
                    gdf_clean: gpd.GeoDataFrame, flags: dict, dropped: list):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Source CRS     : {original_crs}")
    print(f"  Output CRS     : {TARGET_CRS}")
    print(f"  Features (in)  : {len(gdf_raw)}")
    print(f"  Features (out) : {len(gdf_clean)}")
    print(f"  Dropped        : {len(dropped)}")

    flag_labels = {
        "empty":       "Empty geometries",
        "invalid":     "Invalid geometries (fixed)",
        "wrong_type":  "Wrong geometry type",
        "short":       f"Short segments < {MIN_LENGTH_M}m (flagged)",
        "coord_error": "Coordinate errors (inf/nan)",
    }
    for key, label in flag_labels.items():
        n = len(flags.get(key, []))
        if n:
            marker = "  [!]" if key != "invalid" else "  [~]"
            print(f"{marker} {label}: {n}")

    if gdf_clean.geometry.geom_type.iloc[0] in ("LineString", "MultiLineString"):
        lengths = gdf_clean.geometry.length
        print(f"  Length (m)     : min={lengths.min():.1f}  max={lengths.max():.1f}  "
              f"mean={lengths.mean():.1f}  total={lengths.sum()/1000:.2f} km")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproject and QC berm label shapefiles")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Users/nkalauni/Documents/Chaulagain, Smriti - (smritichaulagain)'s files"
                     " - BermIdentification/02 ExisitingDatasets"),
        help="Root of the shared datasets folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw" / "labels",
        help="Output directory for cleaned shapefiles",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=MIN_LENGTH_M,
        help="Minimum polyline length in metres (shorter segments are flagged)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for rel_path, out_name, geom_type in DATASETS:
        src_path = args.source_dir / rel_path
        if not src_path.exists():
            print(f"\n[SKIP] Not found: {src_path}")
            continue

        # --- load ---
        gdf = gpd.read_file(src_path)
        original_crs = gdf.crs
        n_in = len(gdf)

        # --- fix invalid geometries before reprojection ---
        gdf = fix_geometry(gdf)

        # --- reproject ---
        if gdf.crs is None:
            print(f"\n[WARN] {out_name}: no CRS defined, assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")
        if str(gdf.crs) != TARGET_CRS:
            gdf = gdf.to_crs(TARGET_CRS)

        # --- QC ---
        flags = qc_geometry(gdf, geom_type, args.min_length)

        # drop features with empty geometries or coordinate errors (unfixable)
        drop_idx = set(flags["empty"] + flags["coord_error"])
        gdf_clean = gdf.drop(index=list(drop_idx)).reset_index(drop=True)

        # add a flag column for short segments so they're visible in QGIS
        if geom_type == "LineString" and flags["short"]:
            short_in_clean = [i for i in flags["short"] if i not in drop_idx]
            gdf_clean["qc_short"] = False
            gdf_clean.loc[
                gdf_clean.index.isin(short_in_clean), "qc_short"
            ] = True

        # --- save ---
        out_path = args.out_dir / f"{out_name}.shp"
        gdf_clean.to_file(out_path)

        if args.verbose or len(drop_idx) > 0 or any(len(v) for v in flags.values()):
            print_qc_report(out_name, original_crs, gdf, gdf_clean, flags, list(drop_idx))

        summary_rows.append({
            "dataset": out_name,
            "source_crs": str(original_crs),
            "n_in": n_in,
            "n_out": len(gdf_clean),
            "n_dropped": len(drop_idx),
            "n_invalid_fixed": len(flags["invalid"]),
            "n_short_flagged": len(flags["short"]),
            "n_wrong_type": len(flags["wrong_type"]),
        })

    # --- summary table ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    print(f"\nCleaned shapefiles written to: {args.out_dir}")


if __name__ == "__main__":
    main()

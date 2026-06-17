"""
Snap berm polyline labels to DEM crest lines to fix manual digitization offsets.

Algorithm (from berm_methods.docx / Weekly Goals Week 3):
  1. Build a Relative Elevation Model (REM = DEM - low-pass filtered DEM) so
     snapping targets the berm ridge above the regional slope, not absolute elevation.
  2. Densify polyline vertices every VERTEX_SPACING metres.
  3. At each vertex, sample REM along a perpendicular transect (+/- TRANSECT_HALF_WIDTH m).
  4. Move the vertex to the local REM maximum on that transect.
  5. Apply a moving-average smooth to remove jitter from DEM noise.
  6. Flag lines where no clear ridge is found (prominence below threshold) for manual QC.

Usage:
    python scripts/snap_labels_to_dem.py --dataset safford
    python scripts/snap_labels_to_dem.py --dataset altarvalley --dem-dir data/raw/dem/altar_valley
    python scripts/snap_labels_to_dem.py --dataset cochise --dry-run
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import uniform_filter
from shapely.geometry import LineString, MultiLineString
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Tunable parameters (from weekly goals doc)
# ---------------------------------------------------------------------------
VERTEX_SPACING_M    = 1.0    # densify interval along each line
TRANSECT_HALF_WIDTH = 12     # metres each side of vertex to search
TRANSECT_SAMPLES    = 25     # number of points along transect (odd)
REM_WINDOW_M        = 50     # low-pass window size for REM (metres)
MAX_SHIFT_M         = 13     # reject snap if proposed shift > this
PROMINENCE_THRESH   = 0.05   # min REM peak value (m above local mean) to accept snap
SMOOTH_WINDOW       = 5      # moving-average window for post-snap smoothing

# ---------------------------------------------------------------------------
# Dataset registry — (labels path, DEM directory)
# ---------------------------------------------------------------------------
LABELS_DIR = Path(__file__).parent.parent / "data" / "raw" / "labels"
DEM_BASE   = Path(__file__).parent.parent / "data" / "raw" / "dem"

DATASETS = {
    "safford":     (LABELS_DIR / "safford_berms.shp",              DEM_BASE),
    "cochise":     (LABELS_DIR / "cochise_berms.shp",              DEM_BASE),
    "bigchino":    (LABELS_DIR / "bigchino_berms.shp",             DEM_BASE),
    "uppergila":   (LABELS_DIR / "uppergila_berms.shp",            DEM_BASE),
    "altarvalley": (LABELS_DIR / "altarvalley_longberms.shp",      DEM_BASE / "altar_valley"),
    "altarvalley_structures": (
                   LABELS_DIR / "altarvalley_structures.shp",      DEM_BASE / "altar_valley"),
}


# ---------------------------------------------------------------------------
# DEM helpers
# ---------------------------------------------------------------------------

def find_overlapping_tiles(tiles: list[Path], bounds_26912: tuple) -> list[Path]:
    """Return only tiles whose extent intersects the given bounds."""
    xmin, ymin, xmax, ymax = bounds_26912
    overlapping = []
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            if b.left < xmax and b.right > xmin and b.bottom < ymax and b.top > ymin:
                overlapping.append(t)
    return overlapping


def load_dem_for_bounds(dem_dir: Path, bounds: tuple, buffer_m: float = 200) -> tuple:
    """
    Merge DEM tiles covering bounds (+ buffer). Returns (array, transform, res).
    bounds is (xmin, ymin, xmax, ymax) in EPSG:26912.
    """
    xmin, ymin, xmax, ymax = bounds
    buffered = (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)

    tiles = sorted(dem_dir.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No .tif files found in {dem_dir}")

    overlapping = find_overlapping_tiles(tiles, buffered)
    if not overlapping:
        return None, None, None

    datasets = [rasterio.open(t) for t in overlapping]
    mosaic, transform = merge(datasets)
    for ds in datasets:
        ds.close()

    arr = mosaic[0].astype(np.float32)
    nodata = datasets[0].nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan

    res = abs(transform.a)  # pixel size in metres
    return arr, transform, res


def build_rem(dem: np.ndarray, window_m: float, res: float) -> np.ndarray:
    """
    Relative Elevation Model: DEM minus a low-pass filtered version.
    Removes regional hillslope trend so berm crests appear as local highs.
    """
    window_px = max(3, int(window_m / res))
    if window_px % 2 == 0:
        window_px += 1
    trend = uniform_filter(np.where(np.isnan(dem), 0, dem), size=window_px)
    return dem - trend


def sample_raster(arr: np.ndarray, transform, xy: np.ndarray) -> np.ndarray:
    """
    Sample raster array at (N, 2) array of (x, y) world coordinates.
    Returns array of float values (nan where out of bounds).
    """
    cols = (xy[:, 0] - transform.c) / transform.a
    rows = (xy[:, 1] - transform.f) / transform.e
    c = np.round(cols).astype(int)
    r = np.round(rows).astype(int)
    h, w = arr.shape
    valid = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    out = np.full(len(xy), np.nan)
    out[valid] = arr[r[valid], c[valid]]
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def densify(line: LineString, spacing: float) -> np.ndarray:
    """Return (N, 2) array of points along line at `spacing` metre intervals."""
    # drop Z if present
    if line.has_z:
        from shapely.ops import transform as shp_transform
        line = shp_transform(lambda x, y, z=None: (x, y), line)
    total = line.length
    distances = np.arange(0, total, spacing)
    if len(distances) == 0 or distances[-1] < total - 1e-6:
        distances = np.append(distances, total)
    pts = np.array([line.interpolate(d).coords[0] for d in distances])
    return pts[:, :2]  # ensure 2D


def perpendicular_dirs(pts: np.ndarray) -> np.ndarray:
    """
    Unit perpendicular vectors at each vertex. Uses forward/backward difference
    for interior points, forward/backward only at endpoints.
    """
    n = len(pts)
    tangents = np.zeros((n, 2))
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0]    = pts[1]  - pts[0]
    tangents[-1]   = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit_tangents = tangents / norms
    # rotate 90°: (tx, ty) -> (-ty, tx)
    perps = np.column_stack([-unit_tangents[:, 1], unit_tangents[:, 0]])
    return perps


def snap_line(pts: np.ndarray, rem: np.ndarray, transform,
              half_width: float, n_samples: int,
              max_shift: float, prominence_thresh: float) -> tuple:
    """
    Snap each vertex to REM local maximum along its perpendicular transect.

    Returns:
        snapped_pts : (N, 2) array of moved vertices
        shifts      : (N,) array of shift distances in metres
        flagged     : (N,) bool array — True where no clear ridge found
    """
    offsets = np.linspace(-half_width, half_width, n_samples)
    perps = perpendicular_dirs(pts)
    snapped = pts.copy()
    shifts = np.zeros(len(pts))
    flagged = np.zeros(len(pts), dtype=bool)

    # build transect sample points: shape (N, n_samples, 2)
    transect_pts = pts[:, np.newaxis, :] + perps[:, np.newaxis, :] * offsets[np.newaxis, :, np.newaxis]

    for i in range(len(pts)):
        xy = transect_pts[i]          # (n_samples, 2)
        vals = sample_raster(rem, transform, xy)

        if np.all(np.isnan(vals)):
            flagged[i] = True
            continue

        valid_mask = ~np.isnan(vals)
        valid_vals = vals[valid_mask]
        valid_offsets = offsets[valid_mask]

        peak_local_idx = np.argmax(valid_vals)
        peak_val = valid_vals[peak_local_idx]
        prominence = peak_val - np.median(valid_vals)

        if prominence < prominence_thresh:
            flagged[i] = True
            continue

        best_offset = valid_offsets[peak_local_idx]
        if abs(best_offset) > max_shift:
            flagged[i] = True
            continue

        snapped[i] = pts[i] + perps[i] * best_offset
        shifts[i] = abs(best_offset)

    return snapped, shifts, flagged


def smooth_vertices(pts: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth on vertex coordinates, preserving endpoints."""
    if len(pts) < window:
        return pts
    smoothed = pts.copy()
    half = window // 2
    for i in range(half, len(pts) - half):
        smoothed[i] = pts[i - half: i + half + 1].mean(axis=0)
    return smoothed


def pts_to_linestring(pts: np.ndarray) -> LineString | None:
    if len(pts) < 2:
        return None
    # keep points where coordinates change (diff has N-1 rows, prepend True for first point)
    keep = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    unique = pts[keep]
    if len(unique) < 2:
        return None
    return LineString(unique)


# ---------------------------------------------------------------------------
# Per-feature processing
# ---------------------------------------------------------------------------

def process_feature(geom, rem, transform, params):
    if geom is None or geom.is_empty:
        return geom, 0.0, 0.0, True

    # handle MultiLineString by processing each part
    if geom.geom_type == "MultiLineString":
        parts, all_shifts, all_flags = [], [], []
        for part in geom.geoms:
            new_geom, mean_s, flag_frac, _ = process_feature(part, rem, transform, params)
            if new_geom is not None:
                parts.append(new_geom)
            all_shifts.append(mean_s)
            all_flags.append(flag_frac)
        if not parts:
            return geom, 0.0, 1.0, True
        return (MultiLineString(parts),
                float(np.mean(all_shifts)),
                float(np.mean(all_flags)),
                False)

    pts = densify(geom, params["vertex_spacing"])
    if len(pts) < 2:
        return geom, 0.0, 1.0, True

    snapped, shifts, flagged = snap_line(
        pts, rem, transform,
        half_width=params["transect_half_width"],
        n_samples=params["transect_samples"],
        max_shift=params["max_shift"],
        prominence_thresh=params["prominence_thresh"],
    )
    smoothed = smooth_vertices(snapped, params["smooth_window"])
    new_line = pts_to_linestring(smoothed)

    if new_line is None:
        return geom, 0.0, 1.0, True

    mean_shift = float(np.mean(shifts[~flagged])) if np.any(~flagged) else 0.0
    flag_frac = float(flagged.mean())
    return new_line, mean_shift, flag_frac, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Snap berm polylines to DEM crest lines")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Which label dataset to process")
    parser.add_argument("--dem-dir", type=Path, default=None,
                        help="Override DEM directory")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "raw" / "labels_snapped")
    parser.add_argument("--vertex-spacing",     type=float, default=VERTEX_SPACING_M)
    parser.add_argument("--transect-half-width",type=float, default=TRANSECT_HALF_WIDTH)
    parser.add_argument("--max-shift",          type=float, default=MAX_SHIFT_M)
    parser.add_argument("--prominence-thresh",  type=float, default=PROMINENCE_THRESH)
    parser.add_argument("--rem-window",         type=float, default=REM_WINDOW_M)
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 10 features only")
    args = parser.parse_args()

    labels_path, default_dem_dir = DATASETS[args.dataset]
    dem_dir = args.dem_dir or default_dem_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        print("Run scripts/prepare_labels.py first.")
        return

    params = {
        "vertex_spacing":      args.vertex_spacing,
        "transect_half_width": args.transect_half_width,
        "transect_samples":    TRANSECT_SAMPLES,
        "max_shift":           args.max_shift,
        "prominence_thresh":   args.prominence_thresh,
        "smooth_window":       SMOOTH_WINDOW,
    }

    print(f"Loading labels: {labels_path.name}")
    gdf = gpd.read_file(labels_path)
    if args.dry_run:
        gdf = gdf.head(10)
        print("  [dry-run] processing first 10 features only")

    print(f"  {len(gdf)} features, CRS: {gdf.crs}")

    # load DEM once for the full dataset extent
    xmin, ymin, xmax, ymax = gdf.total_bounds
    print(f"Loading DEM tiles from {dem_dir} ...")
    dem_arr, dem_tf, dem_res = load_dem_for_bounds(dem_dir, (xmin, ymin, xmax, ymax))

    if dem_arr is None:
        print(f"No DEM tiles overlap this dataset. Check {dem_dir}")
        return

    print(f"  DEM shape: {dem_arr.shape}, resolution: {dem_res}m")

    print(f"Computing REM (window={args.rem_window}m) ...")
    rem = build_rem(dem_arr, args.rem_window, dem_res)

    # process features
    new_geoms, mean_shifts, flag_fracs = [], [], []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Snapping"):
        new_geom, mean_s, flag_f, failed = process_feature(row.geometry, rem, dem_tf, params)
        new_geoms.append(new_geom if not failed else row.geometry)
        mean_shifts.append(mean_s)
        flag_fracs.append(flag_f)

    gdf_out = gdf.copy()
    gdf_out["geometry"]   = new_geoms
    gdf_out["snap_shift"] = np.round(mean_shifts, 2)  # mean shift per line (m)
    gdf_out["snap_flags"] = np.round(flag_fracs, 2)   # fraction of unfixed vertices
    # flag for manual review: >50% of vertices could not be snapped
    gdf_out["needs_qc"]   = gdf_out["snap_flags"] > 0.5

    out_path = args.out_dir / labels_path.name
    gdf_out.to_file(out_path)

    # report
    n_qc    = gdf_out["needs_qc"].sum()
    shifted = gdf_out[gdf_out["snap_shift"] > 0]
    print(f"\n--- {args.dataset} snap results ---")
    print(f"  Features processed  : {len(gdf_out)}")
    print(f"  Features snapped    : {len(shifted)} ({100*len(shifted)/len(gdf_out):.1f}%)")
    print(f"  Mean shift (m)      : {gdf_out['snap_shift'].mean():.2f}")
    print(f"  Max shift (m)       : {gdf_out['snap_shift'].max():.2f}")
    print(f"  Flagged for QC      : {n_qc} ({100*n_qc/len(gdf_out):.1f}%)")
    print(f"\nSaved: {out_path}")
    print("Open in QGIS: load original + snapped + DEM hillshade, spot-check ~5% of features.")


if __name__ == "__main__":
    main()

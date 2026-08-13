"""
Build the multi-channel feature stack used for training, from a merged DEM
and NAIP imagery. Reproduces the full 12-channel design:

Terrain (6):
    1. resid15    - residual relief, 15m window   (whitebox diff_from_mean_elev)
    2. resid45    - residual relief, 45m window   (whitebox diff_from_mean_elev)
    3. openness   - positive topographic openness (src/utils/openness.py --
                     reimplemented from scratch; the free WhiteboxTools build
                     doesn't include this tool)
    4. profile_curvature - profile curvature, log-transformed, on a lightly
                     Gaussian-smoothed DEM (whitebox profile_curvature)
    5. multidirectional_hillshade                 (whitebox multidirectional_hillshade)
    6. depression_depth - filled DEM minus DEM     (whitebox fill_depressions,
                     then simple subtraction)
Hydrologic (2):
    7. omega      - flow-orthogonality: |sin(theta_struct - theta_flow)|
                     (src/utils/flow_orthogonality.py -- Hessian of resid15
                     x D-infinity flow direction)
    8. log_flowacc - log(1 + D-infinity flow accumulation) (whitebox
                     d_inf_flow_accumulation, log=True)
Optical/NAIP (4):
    9-11. red, green, blue - raw NAIP bands, cubic-resampled to the DEM's
                     1m grid (kept raw so ImageNet-pretrained encoder
                     weights transfer meaningfully into the first 3 slots)
    12. savi_anomaly - SAVI minus its ~75m local median background

Optional 13th channel:
    dist_to_road - distance (metres) to the nearest OpenStreetMap
                     highway=* centreline. A hard version of this idea
                     (suppress predictions overlapping a road buffer) was
                     tested as a post-filter and made things worse, not
                     better -- see docs/experiments.md finding #6. This is
                     the soft/learned version: a continuous distance value
                     the model can weigh against other evidence, instead
                     of a binary cutoff.

Every step is idempotent (skips recomputation if its output file already
exists), so re-running after a partial failure or to extend the channel
list is cheap.

Reading NAIP .sid files requires LizardTech's free MrSID Decode SDK (not
pip-installable -- GDAL's own MrSID driver is excluded from open-source
builds for licensing reasons). Download the Linux build from
https://www.lizardtech.com/developer, extract it anywhere, then point
--mrsiddecode-bin at Raster_DSDK/bin/mrsiddecode (chmod +x it first -- it
ships without the executable bit).

IMPORTANT: GeoTIFFs fed to WhiteboxTools must NOT use a floating-point LZW
predictor -- its Rust GeoTIFF reader panics on predictor=3, SILENTLY (the
Python wrapper returns exit code 0 regardless of the subprocess crash).
Plain LZW (no predictor) or uncompressed is fine. This script's own writes
never set a predictor; if you feed it a DEM merged by some other tool,
strip any float predictor from it first.

Usage:
    python scripts/build_feature_stack.py --dataset altarvalley \
        --naip data/raw/naip/ortho_1-1_hm_s_az019_2025_1/ortho_1-1_hm_s_az019_2025_1.sid \
        --mrsiddecode-bin ~/mrsid_sdk/MrSID_DSDK-*/Raster_DSDK/bin/mrsiddecode

    # Just the 4-channel core (no NAIP needed):
    python scripts/build_feature_stack.py --dataset altarvalley \
        --channels resid15 openness omega
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter, zoom

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.flow_orthogonality import flow_orthogonality  # noqa: E402
from utils.openness import positive_openness_chunked  # noqa: E402
from utils.osm import query_overpass_highways, ways_to_geodataframe  # noqa: E402

DEM_BASE = Path(__file__).parent.parent / "data" / "processed" / "dem"
FEATURES_BASE = Path(__file__).parent.parent / "data" / "processed" / "features"
NODATA = -999999.0

DATASETS = {
    "altarvalley": DEM_BASE / "AltarValleyMerged.tif",
}

CORE_4_CHANNELS = ["resid15", "openness", "omega", "savi_anomaly"]
FULL_12_CHANNELS = [
    "resid15", "resid45", "openness", "profile_curvature",
    "multidirectional_hillshade", "depression_depth",
    "omega", "log_flowacc",
    "red", "green", "blue", "savi_anomaly",
]
# dist_to_road: soft (continuous) road-proximity signal, added after a hard
# post-filter version of the same idea tested worse than baseline (see
# docs/experiments.md finding #6) -- a learned channel lets the model weigh
# road proximity against other evidence instead of a blanket cutoff.
FULL_CHANNELS = FULL_12_CHANNELS + ["dist_to_road"]
NAIP_CHANNELS = {"red", "green", "blue", "savi_anomaly"}
ROADS_BASE = Path(__file__).parent.parent / "data" / "processed" / "roads"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _check(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Expected output missing/empty: {path}. "
            "Check for a silent WhiteboxTools panic (e.g. unsupported GeoTIFF predictor)."
        )


def _write_like(ref_path: Path, out_path: Path, data: np.ndarray) -> None:
    with rasterio.open(ref_path) as src:
        profile = src.profile
    profile.update(dtype="float32", nodata=NODATA, bigtiff="YES")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


class WhiteboxRunner:
    """Thin wrapper so every terrain tool call goes through one working dir + idempotency check."""

    def __init__(self, dem_path: Path, work_dir: Path):
        import whitebox
        self.dem_path = dem_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.wbt = whitebox.WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.set_working_dir(str(dem_path.parent))

    def run(self, name: str, out_name: str, fn) -> Path:
        out_path = self.work_dir / out_name
        if not out_path.exists():
            t0 = time.time()
            fn(self.wbt, out_path)
            log(f"{name} done ({time.time()-t0:.0f}s)")
        _check(out_path)
        return out_path


def compute_hydro_terrain(dem_path: Path, work_dir: Path) -> dict:
    """Core WhiteboxTools chain shared by several channels: resid15, resid45,
    fill_depressions, D-infinity flow direction + accumulation."""
    wb = WhiteboxRunner(dem_path, work_dir)
    paths = {}
    paths["resid15"] = wb.run(
        "resid15", "resid15.tif",
        lambda w, o: w.diff_from_mean_elev(dem_path.name, str(o), filterx=15, filtery=15),
    )
    paths["resid45"] = wb.run(
        "resid45", "resid45.tif",
        lambda w, o: w.diff_from_mean_elev(dem_path.name, str(o), filterx=45, filtery=45),
    )
    paths["filled"] = wb.run(
        "fill_depressions", "filled.tif",
        lambda w, o: w.fill_depressions(dem_path.name, str(o)),
    )
    paths["flowdir"] = wb.run(
        "d_inf_pointer", "flowdir.tif",
        lambda w, o: w.d_inf_pointer(str(paths["filled"]), str(o)),
    )
    paths["log_flowacc"] = wb.run(
        "d_inf_flow_accumulation", "flowacc.tif",
        lambda w, o: w.d_inf_flow_accumulation(str(paths["filled"]), str(o), out_type="cells", log=True),
    )
    return paths


def compute_profile_curvature(dem_path: Path, work_dir: Path, smooth_sigma: float = 1.5) -> Path:
    out_path = work_dir / "profile_curvature.tif"
    if out_path.exists():
        _check(out_path)
        return out_path

    smoothed_path = work_dir / "dem_smoothed.tif"
    if not smoothed_path.exists():
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            profile = src.profile
        valid = dem != NODATA
        dem_filled = np.where(valid, dem, np.nanmean(np.where(valid, dem, np.nan)))
        smoothed = gaussian_filter(dem_filled, sigma=smooth_sigma)
        smoothed[~valid] = NODATA
        profile.update(bigtiff="YES")
        with rasterio.open(smoothed_path, "w", **profile) as dst:
            dst.write(smoothed.astype(np.float32), 1)

    wb = WhiteboxRunner(dem_path, work_dir)
    wb.wbt.set_working_dir(str(work_dir))
    return wb.run(
        "profile_curvature", "profile_curvature.tif",
        lambda w, o: w.profile_curvature(smoothed_path.name, str(o), log=True),
    )


def compute_multidirectional_hillshade(dem_path: Path, work_dir: Path) -> Path:
    wb = WhiteboxRunner(dem_path, work_dir)
    return wb.run(
        "multidirectional_hillshade", "multidirectional_hillshade.tif",
        lambda w, o: w.multidirectional_hillshade(dem_path.name, str(o), altitude=45),
    )


def compute_depression_depth(dem_path: Path, filled_path: Path, work_dir: Path) -> Path:
    out_path = work_dir / "depression_depth.tif"
    if out_path.exists():
        _check(out_path)
        return out_path
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    with rasterio.open(filled_path) as src:
        filled = src.read(1)
    valid = (dem != NODATA) & (filled != NODATA)
    depth = np.where(valid, filled - dem, NODATA)
    _write_like(dem_path, out_path, depth)
    return out_path


def compute_openness(dem_path: Path, out_path: Path, device: str = "cuda:0") -> Path:
    if out_path.exists():
        _check(out_path)
        return out_path
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
    valid = dem != NODATA
    t0 = time.time()
    result = positive_openness_chunked(dem, valid, radius_cells=30, cell_size=1.0, device=device, chunk_rows=4000)
    log(f"openness done ({time.time()-t0:.0f}s)")
    result_out = np.where(valid, result, NODATA).astype(np.float32)
    profile.update(dtype="float32", nodata=NODATA, bigtiff="YES")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(result_out, 1)
    return out_path


def compute_omega(resid15_path: Path, flowdir_path: Path, out_path: Path) -> Path:
    if out_path.exists():
        _check(out_path)
        return out_path
    with rasterio.open(resid15_path) as src:
        resid = src.read(1)
        profile = src.profile
    with rasterio.open(flowdir_path) as src:
        flowdir = src.read(1)
    valid = (resid != NODATA) & (flowdir != NODATA)
    resid_f = np.where(valid, resid, 0.0).astype(np.float32)
    flowdir_f = np.where(valid, flowdir, 0.0).astype(np.float32)
    t0 = time.time()
    omega = flow_orthogonality(resid_f, flowdir_f, sigma=3.0)
    log(f"omega done ({time.time()-t0:.0f}s)")
    omega[~valid] = NODATA
    profile.update(dtype="float32", nodata=NODATA, bigtiff="YES")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(omega, 1)
    return out_path


def _decode_naip_window(naip_sid: Path, mrsiddecode_bin: Path, dem_bounds, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    sdk_lib = mrsiddecode_bin.parent.parent / "lib"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{sdk_lib}:{mrsiddecode_bin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    left, bottom, right, top = dem_bounds
    cmd = [
        str(mrsiddecode_bin), "-i", str(naip_sid), "-o", str(out_path),
        "-of", "tifg", "-coord", "geo",
        "-ulxy", str(left), str(top), "-lrxy", str(right), str(bottom),
    ]
    log(f"decoding NAIP window: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    return out_path


def _resample_naip_band(naip_window: Path, band_idx: int, dst_transform, dst_crs, dst_shape) -> np.ndarray:
    with rasterio.open(naip_window) as src:
        src_band = src.read(band_idx).astype(np.float32)
        src_transform, src_crs = src.transform, src.crs
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=src_band, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.cubic,
    )
    # cubic resampling can overshoot the valid 8-bit range near sharp edges --
    # clip it back, or SAVI's denominator can go near zero and blow up.
    return np.clip(dst, 0.0, 255.0)


# NAIP band order confirmed from the tile's own .xml metadata: R, G, B, Infrared
NAIP_BAND_INDEX = {"red": 1, "green": 2, "blue": 3}


def compute_raw_naip_band(
    band_name: str, dem_path: Path, naip_sid: Path, mrsiddecode_bin: Path, work_dir: Path, out_path: Path,
) -> Path:
    if out_path.exists():
        _check(out_path)
        return out_path
    with rasterio.open(dem_path) as dem_src:
        dst_transform, dst_crs = dem_src.transform, dem_src.crs
        dst_shape = (dem_src.height, dem_src.width)
        bounds = dem_src.bounds
        dem_valid = dem_src.read(1) != NODATA

    naip_window = work_dir / "naip_window.tif"
    _decode_naip_window(naip_sid, mrsiddecode_bin, bounds, naip_window)

    t0 = time.time()
    band = _resample_naip_band(naip_window, NAIP_BAND_INDEX[band_name], dst_transform, dst_crs, dst_shape)
    log(f"NAIP {band_name} resampled ({time.time()-t0:.0f}s)")
    band_out = np.where(dem_valid, band, NODATA)
    _write_like(dem_path, out_path, band_out)
    return out_path


def compute_savi_anomaly(
    dem_path: Path, naip_sid: Path, mrsiddecode_bin: Path, work_dir: Path, out_path: Path,
) -> Path:
    if out_path.exists():
        _check(out_path)
        return out_path

    with rasterio.open(dem_path) as dem_src:
        dem_valid = dem_src.read(1) != NODATA
        dst_transform, dst_crs = dem_src.transform, dem_src.crs
        dst_shape = (dem_src.height, dem_src.width)
        bounds = dem_src.bounds

    naip_window = work_dir / "naip_window.tif"
    _decode_naip_window(naip_sid, mrsiddecode_bin, bounds, naip_window)

    t0 = time.time()
    red = _resample_naip_band(naip_window, 1, dst_transform, dst_crs, dst_shape)
    nir = _resample_naip_band(naip_window, 4, dst_transform, dst_crs, dst_shape)
    log(f"NAIP Red/NIR resampled to DEM grid ({time.time()-t0:.0f}s)")

    savi = ((nir - red) / (nir + red + 0.5)) * 1.5
    savi[~dem_valid] = np.nan

    t0 = time.time()
    DS = 8  # downsample factor for the ~75m median background (full-res
    # median_filter over a ~150px window is too slow at this raster size)
    savi_filled = np.nan_to_num(savi, nan=0.0)
    small = zoom(savi_filled, 1.0 / DS, order=1)
    small_med = median_filter(small, size=max(3, round(75 / DS)))
    background = zoom(small_med, savi.shape[0] / small_med.shape[0], order=1)
    bh, bw = background.shape
    th, tw = savi.shape
    background = background[:th, :tw] if bh >= th and bw >= tw else np.pad(
        background, ((0, max(0, th - bh)), (0, max(0, tw - bw))), mode="edge"
    )[:th, :tw]
    savi_anom = savi - background
    log(f"SAVI anomaly computed ({time.time()-t0:.0f}s)")

    savi_anom[~dem_valid] = NODATA
    _write_like(dem_path, out_path, savi_anom)
    return out_path


def compute_dist_to_road(dataset: str, dem_path: Path, work_dir: Path, out_path: Path) -> Path:
    """Distance (metres) to the nearest OpenStreetMap highway=* centreline,
    NOT a buffered/binary mask -- lets the model weigh road proximity as
    continuous evidence rather than a hard cutoff (see FULL_CHANNELS note)."""
    if out_path.exists():
        _check(out_path)
        return out_path

    with rasterio.open(dem_path) as src:
        transform, width, height, crs = src.transform, src.width, src.height, src.crs
        dem = src.read(1)
        bounds = src.bounds
    valid = dem != NODATA

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs84.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = to_wgs84.transform(bounds.right, bounds.top)
    cache_path = ROADS_BASE / dataset / "osm_roads_raw.json"
    osm_data = query_overpass_highways((lon_min, lat_min, lon_max, lat_max), cache_path)

    gdf = ways_to_geodataframe(osm_data).to_crs(crs)
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    road_lines = rio_rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

    t0 = time.time()
    dist = distance_transform_edt(road_lines == 0).astype(np.float32)  # pixels == metres on this 1m grid
    log(f"dist_to_road computed ({time.time()-t0:.0f}s)")

    dist[~valid] = NODATA
    _write_like(dem_path, out_path, dist)
    return out_path


def build_stack(channel_paths: dict, channel_order: list, out_path: Path) -> None:
    ref_profile = rasterio.open(channel_paths[channel_order[0]]).profile
    profile = dict(ref_profile)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(count=len(channel_order), dtype="float32", nodata=NODATA,
                   compress="lzw", tiled=True, blockxsize=256, blockysize=256,
                   bigtiff="YES")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, name in enumerate(channel_order, start=1):
            with rasterio.open(channel_paths[name]) as src:
                band = src.read(1).astype(np.float32)
                # remap this source's own nodata sentinel (if different from
                # the pipeline-wide NODATA -- e.g. WhiteboxTools' int16
                # outputs use -32768) so every band in the stack shares one
                # consistent nodata value.
                if src.nodata is not None and src.nodata != NODATA:
                    band[src.read(1) == src.nodata] = NODATA
                dst.write(band, i)
            dst.set_band_description(i, name)
    log(f"wrote stack: {out_path} ({len(channel_order)} bands: {channel_order})")


def compute_norm_stats(channel_paths: dict, channel_order: list, out_json: Path) -> None:
    stats = {}
    for name in channel_order:
        with rasterio.open(channel_paths[name]) as src:
            data = src.read(1)
            # Use this file's OWN nodata tag, not the pipeline-wide NODATA
            # constant -- some WhiteboxTools outputs (e.g. multidirectional
            # hillshade, written as int16) use their own nodata convention
            # (-32768) regardless of the input DEM's nodata value. Blindly
            # assuming NODATA here let ~943M sentinel pixels contaminate
            # that channel's percentile/mean/std the first time around.
            file_nodata = src.nodata if src.nodata is not None else NODATA
            valid = data != file_nodata
            v = data[valid]
        lo, hi = np.percentile(v, [1, 99])
        clipped = np.clip(v, lo, hi)
        stats[name] = {
            "p1": float(lo), "p99": float(hi),
            "mu": float(clipped.mean()), "sigma": float(clipped.std()),
        }
        log(f"  stats[{name}]: p1={lo:.3f} p99={hi:.3f} mu={clipped.mean():.3f} sigma={clipped.std():.3f}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2))
    log(f"wrote normalization stats: {out_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--naip", type=Path, default=None, help="Path to the NAIP .sid covering this dataset's extent")
    parser.add_argument("--mrsiddecode-bin", type=Path, default=None, help="Path to the mrsiddecode binary (MrSID Decode SDK)")
    parser.add_argument("--device", default="cuda:0", help="torch device for the openness computation")
    parser.add_argument("--channels", nargs="+", default=FULL_CHANNELS, choices=FULL_CHANNELS)
    args = parser.parse_args()

    dem_path = DATASETS[args.dataset]
    work_dir = FEATURES_BASE / args.dataset / "_intermediate"
    channel_paths = {}
    channels = set(args.channels)

    need_hydro_terrain = bool(channels & {"resid15", "resid45", "openness", "omega", "log_flowacc", "depression_depth"})
    hydro = compute_hydro_terrain(dem_path, work_dir) if need_hydro_terrain else {}

    if "resid15" in channels:
        channel_paths["resid15"] = hydro["resid15"]
    if "resid45" in channels:
        channel_paths["resid45"] = hydro["resid45"]
    if "log_flowacc" in channels:
        channel_paths["log_flowacc"] = hydro["log_flowacc"]
    if "depression_depth" in channels:
        channel_paths["depression_depth"] = compute_depression_depth(dem_path, hydro["filled"], work_dir)
    if "profile_curvature" in channels:
        channel_paths["profile_curvature"] = compute_profile_curvature(dem_path, work_dir)
    if "multidirectional_hillshade" in channels:
        channel_paths["multidirectional_hillshade"] = compute_multidirectional_hillshade(dem_path, work_dir)
    if "openness" in channels:
        channel_paths["openness"] = compute_openness(dem_path, work_dir / "openness.tif", device=args.device)
    if "omega" in channels:
        channel_paths["omega"] = compute_omega(hydro["resid15"], hydro["flowdir"], work_dir / "omega.tif")

    naip_needed = channels & NAIP_CHANNELS
    if naip_needed and (args.naip is None or args.mrsiddecode_bin is None):
        parser.error(f"--naip and --mrsiddecode-bin are required for channels: {naip_needed}")
    for band_name in ("red", "green", "blue"):
        if band_name in channels:
            channel_paths[band_name] = compute_raw_naip_band(
                band_name, dem_path, args.naip, args.mrsiddecode_bin, work_dir, work_dir / f"{band_name}.tif"
            )
    if "savi_anomaly" in channels:
        channel_paths["savi_anomaly"] = compute_savi_anomaly(
            dem_path, args.naip, args.mrsiddecode_bin, work_dir, work_dir / "savi_anomaly.tif"
        )
    if "dist_to_road" in channels:
        channel_paths["dist_to_road"] = compute_dist_to_road(
            args.dataset, dem_path, work_dir, work_dir / "dist_to_road.tif"
        )

    out_dir = FEATURES_BASE / args.dataset
    build_stack(channel_paths, args.channels, out_dir / "stack.tif")
    compute_norm_stats(channel_paths, args.channels, out_dir / "norm_stats.json")


if __name__ == "__main__":
    main()

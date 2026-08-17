"""
Combine full-raster predictions from multiple checkpoints into one
composite raster (max probability across models), for manual QA/QC
review in QGIS -- e.g. spotting candidate unlabeled berms among flagged
detections.

Each input must already exist -- run scripts/evaluate.py --split full
for each checkpoint first.

Usage:
    python scripts/combine_predictions.py \
        --predictions outputs/checkpoints/altarvalley_combined/eval_full/predictions.tif \
                       outputs/checkpoints/altarvalley_13ch_bcedice/eval_full/predictions.tif \
        --out outputs/qgis_review/predicted_berms_combined.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio

NODATA = -1.0  # matches scripts/evaluate.py's predictions.tif nodata convention


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True, type=Path,
                         help="Two or more predictions.tif files (from scripts/evaluate.py --split full)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if len(args.predictions) < 2:
        parser.error("Need at least 2 prediction rasters to combine")

    with rasterio.open(args.predictions[0]) as src:
        profile = src.profile
        combined = src.read(1)
        valid = combined != NODATA
        combined = np.where(valid, combined, 0.0)

    for p in args.predictions[1:]:
        with rasterio.open(p) as src:
            band = src.read(1)
        band_valid = band != NODATA
        valid |= band_valid
        combined = np.maximum(combined, np.where(band_valid, band, 0.0))

    combined = combined.astype(np.float32)
    combined[~valid] = NODATA

    args.out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(bigtiff="YES")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(combined, 1)

    print(f"Combined {len(args.predictions)} rasters -> {args.out}")
    print(f"Valid pixels: {int(valid.sum()):,}")
    print(f"Pixels with combined prob > 0.5: {int((combined[valid] > 0.5).sum()):,}")


if __name__ == "__main__":
    main()

"""
Centerline-based evaluation: how well does the model find the true berm
LINE, independent of the arbitrary buffer width used to build training
masks?

The buffered polygon mask (data/processed/masks/...) is a derived training
target, not verified ground truth -- the only real label is the digitized
(and DEM-snapped) centerline. Area-overlap IoU against a buffered polygon
conflates two different things: "did the model find the right feature" and
"does its predicted blob happen to be close to the width we buffered to."
Buffer width shows up on both sides of that comparison (building the
training target AND the evaluation target), which is circular.

This decouples them: skeletonize the model's prediction down to a
1-pixel-wide line (so prediction WIDTH doesn't matter at all), then measure
distance between that skeleton and the true centerline directly, at a few
small, principled tolerances (not a swept "which width scores best"
parameter) -- the same family of metric as clDice / the Mnih & Hinton
relaxed road-completeness metric.

    correctness(T) = fraction of predicted-skeleton pixels within T
                      metres of the true centerline
    completeness(T) = fraction of true-centerline pixels within T
                      metres of the predicted skeleton
    centerline_f1(T) = harmonic mean of the two

Uses the snapped label shapefiles directly (data/raw/labels_snapped/),
not the buffered masks -- this is the "yardstick" evaluation, run
alongside (not instead of) scripts/evaluate.py's IoU-based metrics.

Usage:
    uv run python scripts/evaluate_centerline.py --checkpoint-dir outputs/checkpoints/altarvalley_combined
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate import load_manifest  # noqa: E402

LABELS_SNAPPED = Path(__file__).parent.parent / "data" / "raw" / "labels_snapped"
TOLERANCES_M = [1, 2, 3, 5, 8, 10]

# Which snapped shapefiles make up the true reference line, per dataset --
# mirrors rasterize_labels.py's "altarvalley_combined" entry.
REFERENCE_LINES = {
    "altarvalley": [
        LABELS_SNAPPED / "altarvalley_longberms_snapped.shp",
        LABELS_SNAPPED / "altarvalley_structures_snapped.shp",
    ],
}


def rasterize_true_centerline(dataset: str, transform, width: int, height: int, crs) -> np.ndarray:
    gdfs = [gpd.read_file(p) for p in REFERENCE_LINES[dataset]]
    gdf = gpd.GeoDataFrame(pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True), crs=crs)
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    return rio_rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "train", "full"], default="val")
    parser.add_argument("--dataset", default="altarvalley")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    manifest = load_manifest(args.checkpoint_dir)
    eval_dir = args.checkpoint_dir / f"eval_{args.split}"
    pred_path = eval_dir / "predictions.tif"
    if not pred_path.exists():
        print(f"{pred_path} not found -- run scripts/evaluate.py first.")
        sys.exit(1)

    row_range = tuple(manifest[f"{args.split}_range"]) if args.split != "full" else None
    with rasterio.open(pred_path) as src:
        probs = src.read(1)
        pred_transform = src.transform
        width, height = src.width, src.height
        crs = src.crs

    print("Rasterizing true centerline (snapped labels, not the buffered mask)...")
    true_line = rasterize_true_centerline(args.dataset, pred_transform, width, height, crs)
    print(f"  true centerline pixels: {int(true_line.sum()):,}")

    pred_binary = probs > args.threshold
    print(f"  predicted-positive pixels: {int(pred_binary.sum()):,}")
    print("Skeletonizing prediction (width no longer matters after this)...")
    skeleton = skeletonize(pred_binary)
    print(f"  predicted-skeleton pixels: {int(skeleton.sum()):,}")

    print("Computing distance transforms...")
    dist_to_true_line = distance_transform_edt(true_line == 0)
    dist_to_skeleton = distance_transform_edt(skeleton == 0)

    skel_dists = dist_to_true_line[skeleton]
    true_dists = dist_to_skeleton[true_line == 1]

    results = {}
    for T in TOLERANCES_M:
        correctness = float((skel_dists <= T).mean()) if len(skel_dists) else 0.0
        completeness = float((true_dists <= T).mean()) if len(true_dists) else 0.0
        f1 = 2 * correctness * completeness / (correctness + completeness + 1e-9)
        results[str(T)] = {"correctness": correctness, "completeness": completeness, "centerline_f1": f1}
        print(f"  T={T:>2}m: correctness={correctness:.4f}  completeness={completeness:.4f}  "
              f"centerline_f1={f1:.4f}")

    out = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "threshold": args.threshold,
        "true_centerline_pixels": int(true_line.sum()),
        "predicted_positive_pixels": int(pred_binary.sum()),
        "predicted_skeleton_pixels": int(skeleton.sum()),
        "by_tolerance_m": results,
    }
    out_path = eval_dir / "metrics_centerline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

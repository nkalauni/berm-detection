# Experiment log — Altar Valley berm detection

Living document. Append new runs to the table below; move detailed
findings/discussion into the sections underneath. Metrics are always from
`scripts/evaluate.py` (full non-overlapping tiled inference over the val
row-range) unless noted otherwise -- the training loop's own reported
`val_iou` uses `pos_fraction`-oversampled crops and reads much higher than
true full-coverage performance, so don't use it to compare runs.

## Setup common to all runs below

- Study area: Altar Valley, merged DEM (`data/processed/dem/AltarValleyMerged.tif`)
- Feature stack: 12 channels (`data/processed/features/altarvalley/stack.tif`) --
  resid15, resid45, openness, profile_curvature, multidirectional_hillshade,
  depression_depth, omega, log_flowacc, red, green, blue, savi_anomaly
- Model: U-Net, ResNet-34 encoder, ImageNet-pretrained first conv inflated
  onto the red/green/blue channel slots
- Train/val split: spatial, rows [0:63689] train / [63945:79931] val (80/20
  by row, on the merged raster -- not a tile list)
- Hyperparameters: lr=1e-3, batch_size=8, dice loss, AdamW, 50 epochs
  (picked via `scripts/sweep_hyperparams.py`, see below)
- Labels: snapped centrelines, buffered N metres each side (see per-run
  buffer width), rasterized onto the merged DEM's grid

## Results

| Run | Buffer | Labels used | checkpoint_dir | val_iou (train-time, biased) | **Full-eval IoU** | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 5m | longberms only | `altarvalley` | 0.3493 (epoch 21) | **0.1168** | 0.137 | 0.443 | 0.209 |
| 2 | 5m | longberms + structures (combined) | `altarvalley_combined` | 0.3478 | **0.1772** | 0.222 | 0.469 | 0.301 |
| 3 | 2m | longberms + structures | `altarvalley_combined_buf2m` | 0.3013 | **0.1427** | 0.179 | 0.415 | 0.250 |
| 4 | 10m | longberms + structures | `altarvalley_combined_buf10m` | 0.4270 | **0.1596** | 0.182 | 0.562 | 0.275 |
| 5 | 3m | longberms + structures | `altarvalley_combined_buf3m` | 0.3208 | **0.1534** | 0.196 | 0.413 | 0.266 |
| 6 | 4m | longberms + structures | `altarvalley_combined_buf4m` | 0.3359 | **0.1695** | 0.217 | 0.438 | 0.290 |
| 7 | 6m | longberms + structures | `altarvalley_combined_buf6m` | 0.4016 | **0.1704** | 0.208 | 0.482 | 0.291 |
| 8 | 7m | longberms + structures | `altarvalley_combined_buf7m` | 0.3755 | **0.1498** | 0.176 | 0.500 | 0.261 |
| 9 | 8m | longberms + structures | `altarvalley_combined_buf8m` | 0.3906 | **0.1792** | 0.218 | 0.499 | 0.304 |

All metrics at decision threshold 0.5. See each run's
`outputs/checkpoints/<checkpoint_dir>/eval_val/metrics.json` for the full
threshold sweep and `.../diagnostics/` for figures.

## Key findings so far

### 1. Training's own val metric is misleading -- always use scripts/evaluate.py
Run 1's training loop reported val_iou=0.3493, but full-coverage evaluation
over the whole val region gave 0.1168. Training's validation set uses
`pos_fraction=0.5` (oversampled toward berm-containing crops) so it never
really tested how often the model false-positives across the much larger
berm-free majority of the landscape. Real precision was only 0.137 at
threshold 0.5.

### 2. The longberms-only mask (run 1) was missing 808 real labeled features
`rasterize_labels.py`'s `"altarvalley"` dataset key only ever pointed at
`altarvalley_longberms_snapped_buf{buf}m.shp`; the structures shapefile
was buffered but never rasterized into the mask actually used for training.
Combining them (run 2, `"altarvalley_combined"` key) raised full-eval IoU
0.1168 -> 0.1772 and precision 0.137 -> 0.222, recall held ~flat (~0.47).
Real improvement, but not the dominant effect (see #4).

### 3. Channel sensitivity is reproducible across runs, and doesn't match the original design
Zeroing each channel at inference and measuring the IoU drop (200
berm-centered val tiles), for both run 1 and run 2:

| Channel | Run 1 drop | Run 2 drop |
|---|---|---|
| resid45 | +0.1193 | **+0.1741** |
| profile_curvature | +0.0906 | +0.1559 |
| log_flowacc | +0.0526 | +0.0462 |
| depression_depth | -0.0012 | +0.0391 |
| multidirectional_hillshade | +0.0337 | +0.0111 |
| openness | +0.0320 | +0.0306 |
| resid15 | +0.0017 | +0.0123 |
| blue | -0.0042 | +0.0120 |
| green | +0.0090 | +0.0117 |
| red | +0.0086 | +0.0076 |
| omega | -0.0003 | +0.0062 |
| savi_anomaly | -0.0054 | +0.0048 |

`resid45` and `profile_curvature` dominate in both runs by a wide margin.
The originally-planned "4-channel core" (resid15, openness, omega,
savi_anomaly -- picked a priori as "where most of the performance lives")
shows near-zero sensitivity in the actual trained model. Worth an explicit
ablation run: does dropping omega/savi_anomaly/depression_depth (near-zero
in both runs) hurt at all, or can the channel count be cut without losing
performance?

### 4. The dominant remaining problem is false positives spread across ~the whole landscape, not concentrated failures near real berms
The per-1024px-block IoU heatmap is red (IoU~0) across nearly the entire
study area in both runs, including blocks with zero ground-truth berm
density -- meaning the model predicts *some* positive pixels almost
everywhere, not just near mislabeled/missed real berms.

Tested the "labels are just incomplete" hypothesis directly: computed
distance from every pixel to the nearest real labeled berm, compared for
false-positive vs. true-negative (background) pixels (run 2, val region):

| | FP pixels | TN (background) pixels |
|---|---|---|
| Median distance to nearest real berm | 949m | 3266m |
| % within 20m of a real berm | 8.0% | 0.2% (40x less) |

Partial support for under-labeling (40x enrichment within a tight 20m
radius -- some false positives probably are real missed berms right at
the edge of known networks), but not the whole story: the bulk of the FP
distribution sits far (median ~950m, 75th %ile ~3.7km) from any real
label, more consistent with the model over-firing on a generic
convex-ridge-at-45m-scale shape (exactly the channel it relies on most)
wherever similar terrain occurs, not specifically near berms.

The false-positive gallery (`diagnostics/fp_gallery.png`) shows concrete
examples: several flagged false positives trace clean, distinct linear
ridge features with nothing labeled there at all (plausible unlabeled
berms, worth a QC pass), and at least one clear road (the berm-vs-road
confusion the flow-orthogonality channel was originally meant to prevent --
worth checking whether Omega is actually being computed/used effectively
given its near-zero sensitivity score above).

### 5. Buffer width: 4m-8m form a plateau clearly ahead of 2-3m; recall keeps climbing out to 10m

Full sweep, same hyperparameters and combined labels throughout, one run
per width (single seed, no repeats -- see the noise caveat below):

| Buffer | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| 2m | 0.1427 | 0.179 | 0.415 | 0.250 |
| 3m | 0.1534 | 0.196 | 0.413 | 0.266 |
| 4m | 0.1695 | 0.217 | 0.438 | 0.290 |
| 5m | 0.1772 | 0.222 | 0.469 | 0.301 |
| 6m | 0.1704 | 0.208 | 0.482 | 0.291 |
| 7m | 0.1498 | 0.176 | 0.500 | 0.261 |
| 8m | **0.1792** | 0.218 | 0.499 | **0.304** |
| 10m | 0.1596 | 0.182 | 0.562 | 0.275 |

8m edges out 5m on IoU/F1 by about 1% -- but that's within the run-to-run
noise this setup already shows elsewhere (single-seed 50-epoch runs, no
repeats; recall the training-time val_iou curve bounces 0.20-0.35 across
epochs of the *same* run). Don't read 8m as "the" optimum. The real
pattern: **2-3m are clearly worse on every metric** (too thin a target for
a 1m-resolution model to hit reliably), **4m-8m form a plateau** (IoU
0.15-0.18, F1 0.26-0.30, no clear winner within it), and **recall keeps
climbing as the buffer widens** (0.415 at 2m -> 0.562 at 10m) while
precision stays roughly flat past 4m -- a wider target is easier to
overlap with, it doesn't make the model's predictions any more accurate.
5m stays the default; 10m is the one to reach for if a use case values
recall over precision (e.g. a first-pass screen meant for human review).

## Open questions / next steps

- [x] Buffer width comparison, full 2-10m sweep -- 4m-8m plateau, 2-3m
      clearly worse, recall keeps climbing with width while precision
      stays flat past 4m (see finding #5). No repeated-seed runs yet, so
      treat the exact ranking within the 4-8m plateau as noisy.
- [ ] Spot-check the near-real-berm false positives (<20m, the most
      actionable slice from finding #4) against source imagery/QGIS --
      candidates for adding to the label set.
- [ ] Try dropping omega/savi_anomaly/depression_depth (consistently
      near-zero sensitivity) -- does a leaner model do as well?
- [ ] The training-time val_iou curve is noisy throughout all 50 epochs
      (bounces 0.20-0.35), not a clean rise-then-plateau -- likely metric
      variance from `val_samples_per_epoch=400` under severe class
      imbalance, worth a larger val sample count to check.
- [ ] `evaluate.py`'s tiled inference uses a non-overlapping grid
      (stride = patch_size, see scripts/evaluate.py:tiled_inference) --
      try overlapping tiles with blending, both as an inference-quality
      fix (a berm crossing exactly on a tile seam gets no cross-tile
      context right now) and as another hyperparameter to sweep.

### 6. Road-mask post-filter: tested, doesn't help (negative result, logged so it isn't retried blind)

Hypothesis from the FP gallery (one flagged false positive was clearly a
road): suppress predictions that overlap a known road, using OpenStreetMap
`highway=*` ways buffered 5m (`scripts/build_road_mask.py`), applied as a
post-hoc filter to `altarvalley_combined` (5m buffer) predictions -- no
retraining, cheapest version of the idea to test first.

| | IoU | Precision | Recall |
|---|---|---|---|
| Baseline | 0.1772 | 0.2217 | 0.4687 |
| Filter: all `highway=*` roads | 0.1627 | 0.2136 | 0.4057 |
| Filter: maintained roads only (excl. track/path/footway) | 0.1759 | 0.2213 | 0.4615 |

Both variants are flat-to-worse, not better. Why: of the predictions that
overlap a mapped road, **25-29% are real true positives**, not confusion
errors -- real berms in this landscape frequently run along or cross
actual ranch roads/tracks (access roads plausibly follow the same terrain
features berms are built on), so a hard spatial cutoff throws out about
as many correct detections as it catches road-confusion false positives.

Doesn't necessarily kill the underlying idea -- the one FP-gallery example
really was a road, it's just not representative of the broader
road-adjacent pixel population. Possible reasons a hard filter fails where
a learned signal might not: OSM road coverage/positional accuracy in this
remote ranch country may be imprecise; and a soft, learned input (a road-
proximity channel the model can weigh against other evidence) might
succeed where a blanket cutoff can't, since it wouldn't need to be right
100% of the time to help. Data for that (`data/processed/roads/altarvalley/`)
is already built if this gets revisited as a 13th channel + retrain.

Follow-up: trying it as a soft `dist_to_road` channel (distance to nearest
OSM centreline, not a hard cutoff) instead -- see build_feature_stack.py.

### 7. Methodological gap caught: the hyperparameter sweep predates the label fix

Checked commit order after being asked directly whether there could be a
bug explaining the sweep's much higher val_iou (0.6722) vs. the full
training runs' (0.35-0.43 same-metric, 0.15-0.18 full-coverage). Two
things, not one:

1. **Not a bug -- the sweep's region was deliberately the easiest slice
   available.** It evaluated on rows 58000:70000, the single
   densest-berm 12000-row window in the whole 79931-row raster (picked on
   purpose so a fast 8-epoch scan would have enough positives to be
   informative). The full runs' val set is the raster's actual
   southernmost 20% -- much larger, representative, far more berm-sparse.
   This was flagged as a ranking caveat at the time but not stated
   clearly enough as also explaining the *absolute* gap.
2. **A real gap, found by checking `git log` order:** `scripts/sweep_hyperparams.py`
   (and the sweep run that picked lr=1e-3/batch_size=8/dice) predates
   the missing-structures-labels fix (finding #2). Every full run since
   has trained on the combined mask; the hyperparameter choice itself was
   never re-validated against it. Not a code bug (sweep and training
   share the same Trainer/BermDataset code, no forked logic), but a real
   staleness -- re-running the sweep against the current combined mask
   to confirm the choice still holds (see next results entry).

### 8. Centerline-based evaluation (the new yardstick) -- validates the IoU-based findings, doesn't overturn them

Raised concern: the only real ground truth is the digitized (DEM-snapped)
centerline *line*; the buffered polygon mask is our own derived artifact
with an arbitrary width baked in, used on *both* sides of the buffer-width
comparison (building the training target and the evaluation target) --
circular. `scripts/evaluate_centerline.py` decouples this: skeletonizes
the model's prediction to a 1px-wide line (so prediction width stops
mattering entirely), then measures distance to the true centerline
directly, at several small, principled tolerances (not swept for best
score) -- the same family as clDice / Mnih & Hinton's relaxed
road-completeness metric. Reports `correctness` (precision-like, on
predicted-skeleton pixels), `completeness` (recall-like, on true-line
pixels), and their harmonic mean `centerline_f1`.

Run against `altarvalley_combined` (5m buffer, the run with buffer-mask
IoU=0.1772, precision=0.222, recall=0.469, F1=0.301):

| Tolerance | Correctness | Completeness | Centerline F1 |
|---|---|---|---|
| 1m | 0.172 | 0.356 | 0.232 |
| 2m | 0.194 | 0.402 | 0.262 |
| 3m | 0.205 | 0.428 | 0.277 |
| 5m | 0.212 | 0.457 | 0.290 |
| 8m | 0.223 | 0.495 | 0.308 |
| 10m | 0.228 | 0.513 | 0.316 |

At T=5m these are nearly identical to the buffer-mask precision/recall/F1
above (0.212/0.457/0.290 vs. 0.222/0.469/0.301). **The circularity concern
was legitimate to check, but empirically the width-independent metric
tells the same story as the buffer-based one** -- the buffer-width
sensitivity findings (#5) and the "model relies on generic terrain shape"
diagnosis (#3/#4) aren't artifacts of the evaluation methodology.

Going forward: **centerline correctness/completeness/F1 at T=3m is the
primary yardstick** (3m chosen as a principled default matching plausible
digitization+snapping positional uncertainty, not swept for best score),
reported alongside the buffer-mask IoU numbers for continuity with
earlier results, not as a replacement for them.

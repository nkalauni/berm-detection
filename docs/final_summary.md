# Final Summary — Berm Detection (Summer Internship Wrap-Up)

This is the narrative handoff document. `docs/experiments.md` is the
detailed, numbered experiment log (every run, every finding, with exact
numbers) -- this document synthesizes it into the questions asked for the
end-of-internship wrap-up. Read this first for the story; go to
`experiments.md` for the receipts.

---

## 1. What worked and what didn't

**Worked:**
- **A fully reproducible, scripted pipeline** (one script per step, every
  step idempotent, everything checked into git except large rasters) meant
  every finding below could actually be traced, verified, and re-run. The
  8-point buffer-width sweep, the 2x2 channel/loss comparison, and the
  centerline-vs-IoU cross-check were only possible because re-running a
  step costs a command, not a manual redo.
- **Separating the training-time validation metric from a real,
  full-coverage evaluation script** was the single most important
  methodological fix. The training loop's own reported val_iou (oversampled
  toward positive examples) consistently overstated true performance by
  roughly 2x. Without `scripts/evaluate.py` doing honest, unbiased,
  full-landscape inference, this project would have reported misleadingly
  good numbers throughout.
- **Testing ideas cheaply before committing to them.** The road-mask idea
  was tested as a 5-minute post-hoc filter before being built into a
  retrained channel; the hyperparameter choice was sanity-checked with an
  8-epoch sweep before every 50-epoch full run. Both caught real problems
  early (the filter didn't work at all; the sweep's own recommendation
  turned out not to transfer to the full landscape either).
- **Following up on "wait, are you sure?"** repeatedly surfaced real
  issues that would otherwise have gone unnoticed: a hyperparameter choice
  validated on stale labels, an evaluation metric that was implicitly
  circular with the training-target buffer width, a whole label class
  silently dropped by a script bug. Every one of these was found by
  someone asking a skeptical question, not by a scheduled check.

**Didn't work / recurring friction:**
- **Small-region hyperparameter sweeps are a weak predictor for this
  task**, more than expected. Not just in magnitude (a cherry-picked dense
  region will always look better than the full heterogeneous landscape --
  expected) but in *which choice wins at all*. The sweep recommended
  `bce_dice` twice; full-scale evaluation preferred `dice` both times. Any
  future hyperparameter decision here needs a full-scale check before being
  trusted, not just a quick sweep.
- **Data products built outside the tracked pipeline are risky.** The
  original merged DEM mosaic (made ad hoc in QGIS, not by a script) was
  silently 100% corrupted and went undetected until it started producing
  visibly wrong terrain-channel outputs. Anything not produced by a
  versioned script in `scripts/` should be treated with suspicion and
  regenerated/verified before being trusted.
- **The a priori channel design didn't match what the model actually
  learned to use.** The "4-channel core" (resid15, openness, omega,
  savi_anomaly) was chosen upfront on domain reasoning about which scale
  and type of feature should discriminate berms from lookalikes. In
  practice the trained model barely uses `omega` or `savi_anomaly` at all,
  and relies most on `resid45` and `profile_curvature` -- channels that
  weren't expected to dominate. Good reminder that channel importance
  needs to be measured, not assumed, even when the domain reasoning behind
  the design seems sound.
- **A cheap fix isn't always a working fix.** The road-mask-as-hard-filter
  idea was the most cost-effective thing to try, and it made things worse,
  not better -- because real berms in this landscape frequently run along
  or cross actual access roads, a fact that wasn't obvious before testing
  it directly.

---

## 2. Which berms are easier/harder to detect

From the visual diagnostics (`plot_diagnostics.py`'s crop gallery):

- **Easier:** clean, continuous, well-defined linear berms with a strong
  topographic crest visible in hillshade, minimal vegetation/clutter
  obscuring the feature, not crossing or running alongside a road. Best
  observed cases reach IoU up to 0.92, with the predicted probability
  tracing the true centerline almost exactly.
- **Harder:** short or subtle segments whose height signal is faint
  relative to surrounding micro-topography; berms near or crossing roads
  (the model doesn't reliably tell the two apart -- see false-positive
  sources below); and, by inference from the channel-sensitivity results,
  any berm whose defining signal is mainly at a fine (~15m) scale rather
  than the ~45m scale the model has learned to rely on most.
- **Untested but worth flagging:** the two label sets (long berm
  centrelines vs. the shorter "structures" features) were always combined
  into one class for training and evaluation. A per-class breakdown
  (does the model do noticeably better/worse on one label type than the
  other?) was never run -- see "what to try next."

---

## 3. Challenges during QA/QC, label prep, and model development

- **A whole label class was silently missing for the first several
  training runs.** `rasterize_labels.py`'s `"altarvalley"` dataset key only
  ever pointed at the long-berms shapefile; the 808-feature structures
  shapefile was buffered but never actually rasterized into the training
  mask, and this went unnoticed until directly asked "did you use both
  shapefiles?" Fixing it raised precision 62%. Lesson: verify which
  *inputs* actually reached the mask, not just that the mask exists.
- **The buffer width used to turn line labels into an area training
  target is inherently a modeling choice, not ground truth** -- and using
  it on both sides of an evaluation (building the training target *and*
  the evaluation target with the same width) is quietly circular. Fixing
  this required building a second, width-independent evaluation metric
  (skeletonize the prediction, measure distance to the true centerline
  directly) rather than just picking a "better" buffer width.
- **Label snapping (correcting manual digitization to the true DEM ridge)
  produces its own QC flags** (`snap_shift`, `snap_flags`, `needs_qc` per
  feature, from `scripts/snap_labels_to_dem.py`) that, as far as I know,
  were generated but never reviewed. The centerline evaluation metric now
  depends on these snapped lines being correct -- worth a review pass
  before more labeling work, since any residual snapping error propagates
  into the "ground truth" the whole project now measures against.
- **Extreme class imbalance (berms cover well under 0.05% of pixels) made
  evaluation itself fragile**, independent of the model. Several real bugs
  this summer were specifically about rare-class evaluation being easy to
  get technically-correct-but-meaningless: a validation split that
  happened to contain zero labeled berms (IoU stuck at 0 regardless of
  model quality), and pure-random validation sampling that almost never
  saw a positive pixel for the same reason.
- **An ad hoc (non-scripted) DEM merge silently produced an all-zero
  raster.** Caught only because downstream terrain channels started
  looking implausible. Any one-off GIS step done outside this repo's
  scripts is a real risk for the same reason.

---

## 4. Channel importance (measured, not assumed)

Measured by zeroing each channel at inference and recording the IoU drop,
on 200 berm-centered validation tiles, consistent across two independent
training runs (see `docs/experiments.md` findings #3 and #10):

| Tier | Channels |
|---|---|
| **Most relied on** | `resid45` (45m-window residual relief), `profile_curvature` |
| Moderately useful | `log_flowacc`, `multidirectional_hillshade`, `openness` |
| **Barely/not relied on** | `resid15` (15m-window residual relief), `omega` (flow-orthogonality), `savi_anomaly`, `depression_depth` |
| Inconclusive | `dist_to_road` (13th channel, added late) -- hurts under `dice` loss, helps under `bce_dice`; real interaction not yet resolved |

The important nuance: this is *not* simply "fine-scale channels don't
help" -- `profile_curvature` is one of the finest-scale channels in the
whole stack and is the #2 most relied-on. The more precise reading is
that the model didn't end up using the specific channels the original
design expected to carry the discriminating signal (`omega` especially --
it was built specifically to separate berms from roads, and its near-zero
importance lines up directly with the model's demonstrated failure to
make that distinction; see below).

---

## 5. Potential sources of false positives

In order of how much evidence supports each, from most to least:

1. **Generic terrain-shape over-generalization (dominant).** The
   region-by-region IoU heatmap shows the model firing on *something*
   across nearly the entire study area, not concentrated near real berms.
   Given the channel-sensitivity results, this is consistent with the
   model keying off "convex/ridge-like at ~45m scale" broadly, rather than
   anything specific to constructed earthworks.
2. **Roads / access tracks.** Directly confirmed visually (the
   false-positive gallery shows a road flagged as a berm). Quantitatively,
   predictions overlapping a mapped road are *not* less likely to be real
   detections than background (25-29% were genuine true positives in the
   road-mask-filter test) -- the model doesn't cleanly separate the two,
   which lines up with `omega`'s near-zero measured importance.
3. **Incomplete labels.** ~8% of false positives sit within 20m of a real
   labeled berm -- a 40x enrichment over random background -- consistent
   with some of these being genuine missed berms rather than model
   errors. This is the single most actionable slice: it's a short list of
   specific locations, not a diffuse problem.
4. **Possible (untested) snapping/digitization noise** in the reference
   centerline itself, which the centerline metric now depends on directly.
   Not directly measured this summer, but worth ruling in or out before
   trusting small differences between models on the centerline yardstick.

---

## 6. What to try next

Roughly in order of effort-to-payoff:

1. **Manual QA/QC pass using `outputs/qgis_review/predicted_berms_combined.tif`**,
   focused specifically on: (a) flagged detections within ~20m of an
   existing label (the most likely genuine misses, per finding #4 above),
   and (b) the `needs_qc`-flagged features from the original snapping step,
   which as far as I know were never reviewed.
2. **Resolve the `dist_to_road` x loss-function interaction properly** --
   repeat-seed runs, and a real hyperparameter sweep that includes the
   channel, rather than one training run per combination.
3. **Multi-temporal NAIP** (discussed but never implemented) -- likely the
   highest-leverage *untried* idea specifically for the road/berm
   confusion, since berms are static legacy earthworks and roads get
   graded/maintained. A stability/difference feature between two NAIP
   dates years apart could distinguish the two in a way no single-date
   channel can.
4. **Per-label-class breakdown** (long berms vs. structures) -- cheap to
   add to `evaluate.py`, never done, could reveal one class is dragging
   down the combined metric.
5. **Investigate *why* `omega` and `savi_anomaly` are underused** rather
   than just accepting it -- worth a targeted check of whether they're
   computed correctly (both were validated against synthetic test cases
   during development, but not re-checked against real labeled berms
   specifically) versus genuinely uninformative for this landscape.
6. **Extend to the other study areas** (Safford, Cochise, Big Chino, Upper
   Gila) -- every script in this pipeline is already parameterized by
   `--dataset`; each area just needs its own DEM/NAIP/labels run through
   the same steps.
7. **Spatial cross-validation** instead of a single train/val split --
   given how heterogeneous the landscape is (the whole reason the
   hyperparameter sweep didn't transfer), a single 80/20 row split may
   itself have more variance than assumed. Worth quantifying.

---

## 7. Recommended defaults (best-validated data/config as of hand-off)

| Item | Location | Notes |
|---|---|---|
| Best label source | `data/raw/labels_snapped/altarvalley_longberms_snapped.shp` **and** `altarvalley_structures_snapped.shp` (both, combined) | DEM-snapped versions, not the raw un-snapped originals in `data/raw/labels/`. Using only one of the two shapefiles was the finding #2 bug -- always use both. |
| Best buffer width | **5m** (4-8m form a plateau with no statistically clear winner; 2-3m are measurably worse; 10m trades precision for recall) | See `docs/experiments.md` finding #5/#10. `configs/train_altarvalley_buf{2,3,4,5,6,7,8,10}m.yaml` exist for re-running the comparison. |
| Best training mask | `data/processed/masks/altarvalley_combined_buf5m/AltarValleyMerged_mask.tif` | Built via `rasterize_labels.py --dataset altarvalley_combined --buffer 5`. |
| Best feature stack | `data/processed/features/altarvalley/stack.tif` (+ `norm_stats.json`) | 13 channels (12-channel plan + `dist_to_road`). Built via `build_feature_stack.py`. |
| Train/val split | Rows 0:63689 (train) / 63945:79931 (val) of the merged raster | Spatial (row-range) split, not a tile list -- see `src/data/dataset.py:split_row_range`. |
| **Best-validated model** | `configs/train_altarvalley.yaml` -> `outputs/checkpoints/altarvalley_combined/best.pth` | 12ch, `dice` loss, lr=1e-3, batch_size=8. The safest, most-validated single choice on both the IoU and centerline yardsticks at the standard 0.5 threshold. |
| Promising-but-unconfirmed alternative | `outputs/checkpoints/altarvalley_13ch_bcedice/best.pth` | 13ch (+dist_to_road), `bce_dice` loss. Reaches the best IoU of anything tested (0.197) at its own optimal threshold, but not yet validated with repeat runs -- see "what to try next" #2. |

---

## 8. Where everything else lives

- `docs/experiments.md` -- the detailed, numbered experiment log with exact
  metrics for every run. Keep appending to it the same way if you continue
  this work.
- `README.md` -- full pipeline walkthrough, download to training, as
  numbered commands.
- `scripts/train_uarizona_hpc.slurm` -- SLURM batch script for UA HPC.
  Several fields are marked `TODO` (account/partition names, module names)
  since they depend on your advisor's allocation and weren't available to
  fill in here -- confirm those against UA HPC's docs or your advisor before
  first use.
- `outputs/qgis_review/predicted_berms_combined.tif` -- combined model
  predictions across the whole study area, for the manual QA/QC pass in
  item 1 above.

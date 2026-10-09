# Frozen scene holdout

The design was committed as `1f9d726` before any registration on these new
fitting scenes. `/workspace/SpatialRust-python-delivery/docs/3DMATCH_HOLDOUT_PLAN.json`
records the original archive, input/reference and algorithm hashes, seeds,
sampler budget, stage settings and primary interventions. The original discovery
scenes are excluded by the plan validator. This is a scene holdout within the
same public 3DMatch benchmark, not a second dataset or a blind external trial.

## Fixed design

Three positive nonconsecutive pairs per scene are integer quantiles of sorted
official GT headers, without using registration outcomes:

| Scene | Source to target pairs |
| --- | --- |
| sun3d-home_at-home_at_scan1_2013_jan_1 | 2 to 0; 28 to 24; 59 to 56 |
| sun3d-hotel_umd-maryland_hotel1 | 14 to 0; 24 to 14; 55 to 53 |
| sun3d-mit_76_studyroom-76-1studyroom2 | 2 to 0; 54 to 13; 65 to 63 |

Official fragment/evaluation ZIPs were downloaded over verified HTTPS from
`https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/`.
CRC and archive hashes are recorded; selected PLY bytes are preserved. GT and
information values match the official toolbox at commit
`4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36`. Clouds contain 137,697–595,761
points. Original metadata fixes source-to-target direction and metre units.

Seeds 7, 8 and 9 with both methods produce 54 planned baseline registrations.
Native generated priors are then held fixed for 54 planned refinement replays
at retained fractions 1.0 and 0.8. The primary alternatives are the frozen
0.8 trim and a 0.2 m centroid-motion guard versus ordinary final native poses.
The guard keeps the initializer when final source-centroid displacement exceeds
0.2 m; otherwise it keeps the final pose. Reference labels never enter either
choice. Other motion limits and forward/balanced candidate selection are
secondary exploratory diagnostics, not additional primary claims.

Publisher normalized information-score squared at most 0.04 defines correctness.
All planned failures stay in denominators. Exact untrimmed controls, unchanged
gates/iteration caps/stopping settings and saved generated priors are checked.
Repeated seeds are conditional replicates, not independent dataset samples.
Three concurrent workers each use one numerical-library thread; elapsed times
do not establish a speed ranking. No default settings are changed by this study.

## Reproduction and verification

Use the pinned local native/helper builds recorded in the plan. The manifest
hash also binds absolute local input paths; a new host needs a new frozen plan
before its own fitting rather than weakening these receipt checks.

```bash
python scripts/validate_3dmatch_holdout_plan.py \
  --plan docs/3DMATCH_HOLDOUT_PLAN.json --cases CASES/cases.json
python scripts/compare_3dmatch_cases.py --cases CASES/cases.json \
  --seeds 7 8 9 --iterations 10000 --workers 3 --case-timeout 1800 \
  --output-dir BASELINE
python scripts/study_3dmatch_refinement_cases.py --cases CASES/cases.json \
  --comparisons BASELINE --trim-fractions 1 .8 --workers 3 \
  --case-timeout 1800 --output-dir REFINEMENT
python scripts/study_stage_selection.py --cases CASES/cases.json \
  --comparisons BASELINE --output-dir STAGES
python scripts/summarize_3dmatch_holdout.py \
  --plan docs/3DMATCH_HOLDOUT_PLAN.json --cases CASES/cases.json \
  --baseline BASELINE/study.json --refinement REFINEMENT/study.json \
  --stages STAGES/study.json --output-dir RESULTS
```

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1` and the
documented activated Python/native environment. The summary validates frozen
inputs and controls, recomputes publisher scores from saved baseline/replay/stage
poses, rejects changed endpoints or selections, and exports JSON, exact HTML
counts and standalone SVG. It executes no registration.

## Verified results

All 54 baseline registrations and 54 refinement replays completed. All 27
untrimmed replay matrices equal their original final matrices exactly. Complete
checkpoint resume succeeds without fitting again. Publisher scores are recomputed
from the bound saved poses by the final summary rather than trusting aggregates.

| Primary native rule | Correct/planned | Gains versus final | Losses versus final |
| --- | --- | --- | --- |
| Ordinary final pose | 17/27 | 0 | 0 |
| Fixed retained fraction 0.8 | 22/27 | 5 | 0 |
| Fixed centroid-motion limit 0.2 m | 17/27 | 0 | 0 |

| Scene | Native ordinary | Native trim 0.8 | Open3D ordinary |
| --- | --- | --- | --- |
| home_at | 6/9 | 9/9 | 6/9 |
| hotel1 | 8/9 | 8/9 | 9/9 |
| studyroom | 3/9 | 5/9 | 0/9 |
| Total | 17/27 | 22/27 | 15/27 |

The five trim gains occur on only two pairs: home_at source 28 to target 24,
all three seeds, and studyroom source 2 to target 0, seeds 7 and 8. Their baseline
scores 0.04115–0.04553 are near the 0.04 threshold; trim scores become
0.00498–0.02781. This supports fixed trimming under the tested conditions,
not recovery from arbitrary globally wrong initializations. Five outputs still
fail. Some successful outputs have worse scores despite remaining correct.
No default trim is changed; this is selected-pair evidence within one benchmark.
Native exceeds Open3D on this cohort, but neither single-run counts nor repeated
seeds establish general library superiority.

### Stage failure generalization

Initial and final native correctness both total 17/27: 16 retained, one recovered,
one lost and nine unrecovered. For studyroom source 2 to target 0, all three
coarse poses are correct but all three ordinary fine poses become incorrect.
Seed 8 starts correct (score 0.03129), improves at coarse (0.01467), then fails at
fine (0.04477). Final centroid displacement is only 0.08595 m, so the frozen
0.2 m guard misses the loss. It selects the initializer on six other outputs,
all still incorrect; equal counts do not mean every selected pose is unchanged.

The same seed's common forward fraction rises 0.30804 to 0.48988, reverse
fraction rises 0.35462 to 0.54650, and forward RMSE falls 0.03261 to 0.02520 m.
Neither forward nor balanced proximity protects the good stage. The fine stage
changes both voxel sampling and correspondence gate, so this does not isolate
point density, resolution or physical outliers as the cause. It does show that
a small-motion rule can miss drift and that optimizing proximity need not
improve reference pose accuracy. Smaller motion limits are secondary diagnostics:
0.05 m scores 17/27 with one gain and one loss, whereas 0.1 m scores 16/27.
They are not retrospectively promoted to primary rules.

### Candidate-pool failure generalization

Secondary three-seed candidate selection scores 6/9 for native and 5/9 for
Open3D. Pooling all six poses scores 5/9 under both frozen proximity rules,
although a correct candidate exists on six pairs. On studyroom source 65 to
target 63, native seed 7 is correct with forward/reverse fractions 0.26484/0.32401
and publisher score 0.00010194. The chosen Open3D seed 8 has greater fractions
0.56582/0.63509 but incorrect score 1.29132. Extra candidates introduce a misleading
alternative here. The previous discovery cohort's pooled 11/12 success therefore
does not establish general selection quality.

A manually chosen post-fit diagnostic plots these two candidates with the same
source/target display samples and equal axis scales in XY/XZ/YZ. It illustrates
the support mismatch, not the physical cause or a new accuracy test.
`scripts/render_saved_pose_geometry.py` binds original cloud bytes and saved
candidate reports, rejects ambiguous choices, and exports optional Matplotlib
PNG/SVG plus exact display coordinates and transforms. It executes no fitting.

## Current-machine artifacts

- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-results/result.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-results/report.html`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-comparison/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-refinement/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-stages/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-candidates/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-refinement-plot/paired_scores.png`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-pooling-geometry-delivered/geometry.png`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-pooling-geometry-delivered/geometry.svg`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-holdout-pooling-geometry-delivered/geometry.json`

Local verification: all 561 ONNX-enabled Python tests pass; the identical default
abi3 wheel passes 558 tests with exactly three expected ONNX skips on both
CPython 3.8 and 3.14. New tests reject discovery-scene leakage, changed gates,
seeds or intervention settings, input/native byte changes, duplicated/missing
paired rows, changed stage endpoints or decisions, and ambiguous display poses.
These Linux checks do not close remote Windows/macOS or additional-dataset gates.

# Live official-fragment comparison

Official fragment and evaluation archives were obtained over verified HTTPS from
`https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/`.
GT poses and information matrices agree numerically with the official toolbox
`andyzeng/3dmatch-toolbox` at commit
`4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36`.
Archive, original PLY, GT, information and reference hashes are preserved.

Four nonconsecutive positive pairs per scene were fixed by integer quantiles of
sorted GT headers before registration. Selection never uses pose error or solver
outcome. Each pair uses seeds 7, 8 and 9 with both SpatialRust and Open3D 0.19:
12 cases, 72 registrations, approximately 90,000–710,000 points per cloud.
Both use the same voxel coordinates, distance gates and RANSAC iteration cap;
descriptor neighborhoods, samplers, stopping and precision differ. Concurrent
timings are diagnostic and do not support a speed ranking.

| Scene | SpatialRust correct/planned | Open3D correct/planned |
| --- | --- | --- |
| 7-scenes-redkitchen | 6/12 | 11/12 |
| sun3d-home_md-home_md_scan9_2012_sep_30 | 10/12 | 10/12 |
| sun3d-hotel_umd-maryland_hotel3 | 7/12 | 6/12 |
| Total | 23/36 | 27/36 |

All planned solver rows produced poses. Correct means the publisher's normalized
information score squared is at most 0.04, evaluated after fitting. These are
selected positive-pair counts, not full benchmark recall, precision or
independent samples: seeds repeat each fixed pair. One native near-half-turn
formula is undefined and explicitly counts as incorrect.

## What the failures show

- Redkitchen source 53 → target 17: native fails all seeds (74–178 degree
  rotation errors), while Open3D succeeds all three. The current initializer
  requires investigation; this comparison does not isolate descriptor matching,
  sampling or refinement as the cause.
- Home source 56 → target 16: each method succeeds on one seed, but on different
  seeds. Initialization instability is visible even when aggregate counts match.
- Hotel source 27 → target 25: native succeeds on seed 7; Open3D succeeds on no
  seed. The successful native pose has only 0.209 forward proximity support.
  Low proximity alone is therefore insufficient grounds for rejecting a pose.
- Hotel source 12 → target 0: neither method satisfies the publisher criterion.
  Open3D seed 8 has support 0.582 and score 0.6631, while seed 7 has lower support
  0.514 and lower score 0.0603. Selecting the most proximity support can worsen
  reference accuracy. Repeated geometry and partial coverage are hypotheses
  requiring controlled perturbations, not established causes.

At the same case/seed, 21 predictions are correct in both methods, 2 only in
SpatialRust and 6 only in Open3D. Their oracle union is 29/36. This demonstrates
complementary outputs; reference labels cannot be used by a deployed selector.
No default algorithm or acceptance threshold was changed from these observations.

## Post-fit stage attribution

`scripts/diagnose_3dmatch_stages.py` verifies the saved study/manifest/child/reference
bindings and evaluates the native initial pose and each saved refinement stage.
Among 36 native rows, 18 retain an initially correct pose, 5 recover an initially
incorrect pose, 3 lose an initially correct pose, and 10 remain incorrect.
All three losses occur on hotel source 12 → target 0: initial scores
0.00698/0.00170/0.01284 become 0.06137/0.04224/0.05004 after the coarse ICP
stage and 0.07003/0.05192/0.06059 after the full-resolution stage. This locates
the loss in refinement rather than initial global sampling for these outputs.
It does not establish the physical cause or justify skipping ICP universally:
five other outputs need refinement to satisfy the criterion.

The bound post-fit diagnosis and standalone HTML are saved at
`/workspace/SpatialRust-python-delivery/target/3dmatch-stage-diagnosis-final/`.

## Controlled replay of the hotel drift

The hotel 12 → 0 case was selected after observing its refinement failures, so
this is an exploratory intervention, not an independent validation set. Its
three saved FPFH initializations were held fixed and each replayed with retained
fractions 1.0, 0.8, 0.6 and 0.4. No reference enters the fitting or trim choice.
All three 1.0 controls reproduce the original final matrices exactly.

| Retained fraction | Publisher correct/planned | Median translation error (m) | Median rotation error (degrees) |
| --- | --- | --- | --- |
| 1.0 | 0/3 | 0.28843 | 2.39744 |
| 0.8 | 3/3 | 0.07987 | 3.11321 |
| 0.6 | 3/3 | 0.08217 | 3.17598 |
| 0.4 | 3/3 | 0.07381 | 2.68489 |

The controlled change suppresses translation drift and satisfies the publisher
criterion in this case, while median rotation error increases. Forward proximity
support also decreases from a median 0.514 to approximately 0.467–0.473. Thus
neither larger proximity support nor a single rotation threshold describes this
result adequately. The intervention supports sensitivity to retained residuals;
it does not isolate physical outliers, prove a universal best fraction, or justify
changing default trimming. Other cases still require prospective paired replay.

`scripts/evaluate_3dmatch_refinement.py` checks the replay/comparison/reference
hash bindings, identical native build and initial matrices, complete unique
seed/fraction slots, unchanged stage gates/iteration caps/stopping criteria,
actual trim settings and exact untrimmed controls before scoring original GT.
The twelve refinements and their post-fit evaluation are under
`/workspace/SpatialRust-python-delivery/target/3dmatch-hotel-drift-replay/` and
`/workspace/SpatialRust-python-delivery/target/3dmatch-hotel-drift-evaluation-delivered/`.

## Reproduction and interruption recovery

Preparation and bounded isolated execution:

```bash
python scripts/prepare_3dmatch_cases.py --archive-dir ARCHIVES \
  --scenes 7-scenes-redkitchen sun3d-home_md-home_md_scan9_2012_sep_30 \
  sun3d-hotel_umd-maryland_hotel3 --cases-per-scene 4 --output-dir CASES
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python scripts/compare_3dmatch_cases.py --cases CASES/cases.json \
  --workers 2 --output-dir COMPARISONS
```

The initial parent process ended without its aggregate report after an environment
reconnection; its precise termination cause is not established. All 12 child
receipts survived. Explicit `--reuse-comparisons COMPARISONS --output-dir NEW_REPORT`
validates current input/reference hashes, seed order, iteration budget, all planned
row slots, row bindings and consistent native/helper/control fingerprints, then
scores existing poses without executing registration. Missing or incompatible
receipts fail validation; this mode does not silently rerun missing cases.

Current-machine artifacts:

- `/workspace/SpatialRust-python-delivery/target/3dmatch-live-cases/cases.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-live-comparison/`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-live-comparison-final/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-live-comparison-final/report.html`

The data and large generated results remain outside Git. The aggregate records
that registration was not re-executed and retains hashes of original child receipts.

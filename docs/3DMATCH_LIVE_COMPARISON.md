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
changing default trimming. Other cases require paired replay; the fixed other-pair study below supplies
conditional validation rather than independent blind evidence.

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

## Fixed other-pair validation

After selecting 0.8 in the exploratory hotel replay, all other 11 pairs of the
original manifest are compared against 1.0. The pilot pair
`sun3d-hotel_umd-maryland_hotel3-12-to-0` is excluded. Original baseline outcomes
were already examined: this is conditional validation, not an unseen benchmark.
Three saved generated priors per pair yield 66 refinements without new RANSAC.

| Retained fraction | Evaluated/planned | Publisher correct | Gains | Losses |
| --- | --- | --- | --- | --- |
| 1.0 | 33/33 | 23 | 0 | 0 |
| 0.8 | 33/33 | 24 | 1 | 0 |

All 33 controls reproduce original final matrices exactly. The single recovery
is hotel source 27 to target 25, seed 8: squared publisher score drops from
0.09821798 to 0.02204845 (threshold 0.04). Eight outputs remain incorrect.
Trimming does not repair the large-error redkitchen initializations. No success
becomes incorrect, but error can increase: hotel source 36 to target 34, seed 8,
rises from 0.00060383 to 0.01152939 while remaining correct. Counts alone hide
regressions; default trimming remains unchanged.

`scripts/study_3dmatch_refinement_cases.py` freezes input, native, helper,
reference and parameter hashes before fitting. Per-case checkpoints retain
errors in planned denominators; `--resume` rejects incompatible plans or changed
artifacts. A complete real-study resume succeeds without fitting again. Each
child uses one numerical-library thread; concurrent times are not speed rankings.

```bash
python scripts/study_3dmatch_refinement_cases.py --cases CASES/cases.json \
  --comparisons COMPARISONS \
  --exclude-case-id sun3d-hotel_umd-maryland_hotel3-12-to-0 \
  --trim-fractions 1 .8 --workers 3 --output-dir REFINEMENT
# Add --resume to the identical command to recover checkpoints.
python scripts/render_refinement_batch.py --study REFINEMENT/study.json \
  --fraction .8 --output-dir PLOT
```

The optional Matplotlib reporting CLI produces PNG/SVG, a log-scale paired
scatter plot, exact HTML table and input/renderer hashes without registration.
Undefined scores and failed cases remain recorded. Ground truth is used only
after fitting; repeated seeds are not independent dataset samples.

Current-machine artifacts:

- `/workspace/SpatialRust-python-delivery/target/3dmatch-other-pairs-refinement/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-other-pairs-refinement/report.html`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-other-pairs-refinement-plot/paired_scores.png`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-other-pairs-refinement-plot/paired_scores.svg`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-other-pairs-refinement-plot/report.html`

## Reference-free multi-candidate selection

`scripts/study_global_candidate_selection.py` rescored all 72 saved poses on the
same 12 pairs without executing registration. Source and target are independently
voxelized at 0.05 m once per pair, with 0.05 m forward/reverse distance gates.
The rules are fixed: maximum forward supported fraction then minimum forward
RMSE, or maximum harmonic mean of both fractions then minimum worst-direction
RMSE. Exact ties retain saved candidate order. Neither rule takes reference
poses, publisher labels or errors as inputs. All selections finish before
publisher evaluation; reference bytes are hashed beforehand for integrity only.

| Candidate pool | Candidates per pair | Forward selected correct | Balanced selected correct | Post-fit available correct pose |
| --- | --- | --- | --- | --- |
| SpatialRust | 3 seeds | 9/12 | 9/12 | 9/12 |
| Open3D | 3 seeds | 10/12 | 10/12 | 10/12 |
| Both methods | 6 poses | 11/12 | 11/12 | 11/12 |

These denominators count pairs, unlike the original per-seed 23/36 and 27/36.
The method pool and three-start compute budget differ, so this is not a claim
of improved single-run recall or speed. This is exploratory analysis of previously
examined selected positive pairs, not an independent benchmark or a validated
production confidence threshold. Downsampling changes the original full-cloud
support objective. Matching the post-fit available-pose count on these 12 pairs
does not establish universal optimal selection; balanced scoring adds no
observed correctness gain here.

The complementarity is concrete: Open3D supplies correct poses for redkitchen
source 58 to target 7 and source 53 to target 17 where all native poses fail.
SpatialRust supplies the correct pose for hotel source 27 to target 25 where
all Open3D poses fail. Both ranking rules recover those available alternatives.
The remaining pooled failure is hotel source 12 to target 0, where all six final
poses fail: selection cannot recover an absent correct candidate. Earlier stage
attribution and fixed-prior trimming identify refinement drift on that pair.
This distinguishes expanding global candidate coverage from protecting a good
initializer during refinement. No ground-truth-informed trim/method switch is
installed in the ordinary API, and Open3D remains outside the native core.

The CLI binds manifest, input clouds, comparisons, references, metadata, native
extension and helpers by SHA-256 and rejects changes. Error candidates stay in
the planned pool and an unavailable selection stays in the pair denominator.
It writes JSON, an exact HTML table, and a standalone SVG for both fixed rules.

```bash
python scripts/study_global_candidate_selection.py --cases CASES/cases.json \
  --comparisons COMPARISONS --output-dir SELECTION
```

Current-machine artifacts:

- `/workspace/SpatialRust-python-delivery/target/3dmatch-global-candidate-selection-delivered/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-global-candidate-selection-delivered/report.html`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-global-candidate-selection-delivered/selection.svg`

## Can reference-free metrics protect an initializer?

`scripts/study_stage_selection.py` evaluates initial, coarse and fine poses for
all 36 saved native outputs: 108 poses, with no registration. It uses the same
0.05 m voxel geometry and 0.05 m bidirectional gates as the candidate study.
Forward or harmonic balanced support, with gated-RMSE tie breaking, selects a
stage before publisher evaluation. Exact ties preserve the earlier stage.

Both proximity rules score 23/36 correct, with zero gains and zero losses versus
final outputs. They do not prevent any of the three hotel source 12 to target 0
losses. In those losses, forward and reverse support rise and gated RMSE falls
as publisher pose accuracy worsens. For seed 7, forward fraction rises
0.40358 to 0.49340 and reverse fraction 0.43278 to 0.54770; forward RMSE falls
0.02950 to 0.02745 m while publisher squared score rises 0.00698 to 0.07003.
Thus bidirectional proximity alone does not repair this failure. This establishes
proxy failure on the selected case, not the physical cause of the mismatch.

The tool also measures source-centroid displacement and rotation from the
initializer. Centroid displacement uses the full source centroid transformed
by each pose; translation-vector differences alone depend on the coordinate
origin and can misdescribe rotation about an off-origin object.

Motion guards keep the initializer if final centroid displacement exceeds a
fixed limit, otherwise keep the final pose. The limits 0.05, 0.1, 0.2 and 0.4 m
are fine/coarse correspondence-gate multiples, not fitted optimizer parameters.
They are exploratory diagnostic alternatives, evaluated together; no best
threshold is independently validated.

| Rule | Correct/planned | Gains versus final | Losses versus final |
| --- | --- | --- | --- |
| Always initializer | 21/36 | 3 | 5 |
| Always final | 23/36 | 0 | 0 |
| Forward proximity | 23/36 | 0 | 0 |
| Balanced proximity | 23/36 | 0 | 0 |
| Motion limit 0.05 m | 22/36 | 3 | 4 |
| Motion limit 0.1 m | 22/36 | 3 | 4 |
| Motion limit 0.2 m | 24/36 | 3 | 2 |
| Motion limit 0.4 m | 23/36 | 0 | 0 |

The 0.2 m guard saves the hotel losses but suppresses two legitimate recoveries
on home source 50 to target 27 (seeds 7 and 9). The three hotel losses move the
source centroid approximately 0.255–0.295 m with only 2.01–4.60 degrees of
rotation. Legitimate recoveries can require greater rotation and substantial
motion. These descriptive differences motivate independent validation, not a
new truth-informed threshold or a production confidence guarantee. All prior
baseline outcomes were known; no default refinement or selection changes.

Input/reference/comparison/native/helper hashes are checked; saved final-stage
poses must match outputs and stage names must be unique. Selections finish before
reference metadata is parsed. Solver failures remain in the planned denominator;
undefined publisher scores remain explicit. The CLI exports exact per-stage
metrics, motion, selections and post-fit scores, HTML, and standalone SVG.

```bash
python scripts/study_stage_selection.py --cases CASES/cases.json \
  --comparisons COMPARISONS --output-dir STAGE_SELECTION
```

Current-machine artifacts:

- `/workspace/SpatialRust-python-delivery/target/3dmatch-stage-selection-final/study.json`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-stage-selection-final/report.html`
- `/workspace/SpatialRust-python-delivery/target/3dmatch-stage-selection-final/selection.svg`

# Fixed keyframe trajectory validation

The previous native point-to-plane study lowers error but both xyz and rpy
fail the fixed 0.10 m first-pose ATE limit. This separate study tests a retained
keyframe to reduce repeated frame-to-frame chaining error. It does not alter
the original estimators, frozen trajectories, calibration or score limits.

## Conditions fixed before execution

`run_tum_keyframe.py` uses the previous estimated pose in the active keyframe
as a zero-displacement initialization. The native point-to-plane solver and
20-neighbor normals, original 20/20 CPU voxel stages and stopping defaults
remain fixed. Every 30 successful poses, after registration and geometric
support validation, the current frame becomes the next keyframe. This is a
fixed roughly one-second interval at the nominal capture rate, not a
reference-derived displacement threshold or a tuned pose-graph optimizer.

The adapter returns current-to-previous deltas to the unchanged world chain.
Promotion never reanchors the world, fits scale, uses future frames or resets
failed tracking. Keyframe/current ordinals, initial and final keyframe poses,
relative deltas, native stages and support are retained in each successful row.
Ordinals count valid poses from the sole identity anchor rather than image
indices, so initial decoding failures cannot mislabel the first keyframe.
Only one keyframe is retained; there is no unbounded dense map or hidden GPU
transfer. Original depth counts, pixel indices and projected xyz retain the
existing exact typed IO audit.

The acceptance conditions are unchanged: every planned pose generated,
at least 95% reference-matched, first-pose-only SE(3) ATE RMSE at most 0.10 m,
and one-second translation RPE RMSE at most 0.05 m. Numerical audit tolerances
remain 1e-8 m / 1e-6 degrees and are not physical sensor accuracy limits.

The three keyframe plans and the separate desk point-to-plane baseline plan
are committed before fitting or desk acquisition. XYZ/rpy are already observed
and are exploratory. The official Freiburg 1 desk sequence is the untouched
validation sequence for this study, fixed before acquisition, generation or
reference interpretation. Its baseline uses the previously fixed point-to-plane
settings, with identity pair initialization and the same geometry/IO budgets.
Reference and accelerometer contents remain withheld during preparation;
reference poses are read only after estimates are frozen. Settings and limits
will not be retuned after inspecting the outcomes. This is one held-out
sequence for this implementation, not a universal blind benchmark claim.

## Input-only revision fixed before desk preparation or fitting

The first protocol/implementation commit is
`aff896602f490eb996436417fbef029d3eec535e`, before desk acquisition. The static
30-frame policy loses rpy at pose 42: support is 0.496512, below the unchanged
0.5 minimum (previous support 0.538556). XYZ also loses tracking. Their full
frozen success/error timelines are retained, and no reference scores have been
read when choosing the next policy.

`run_tum_adaptive_keyframe.py` adds a separately fixed policy: promote after an
already accepted pose has less than 0.75 support, or after 30 accepted poses,
whichever happens first. The 0.5 acceptance limit remains fixed. No failed pose
is promoted, retried, substituted or reset. The overlap margin reserves geometry
for the next image using only current support; it does not read that next image.
The original static plans and outputs are retained. This revision is committed
after desk acquisition but before any desk preparation, fitting or reference
interpretation. Therefore pre-acquisition fixation applies to the original
static study; the adaptive desk study has a pre-fitting fixation. Neither is
claimed to be a universally blind benchmark.

## Outcomes and a separately fixed hybrid revision

All seven static/overlap trajectories were frozen before scoring. Independent
SciPy audits reproduced their scores with the unchanged numerical tolerances.
None satisfies all four acceptance conditions. Static xyz/rpy stop at images
569/42; static desk stops at 26. Overlap xyz/rpy retain all planned poses but
worsen ATE and RPE compared with the preceding point-to-plane baseline. Overlap
desk stops at 266 when native plane ICP finds only three correspondences,
below its six-correspondence requirement. The desk baseline also stops at 526.
The low static-desk ATE describes only its 26 successful poses and cannot
establish full-sequence accuracy. Every failed pose remains in the timeline.

`run_tum_hybrid_keyframe.py` changes the coarse objective to native closed-form
point-to-point SVD ICP, followed by the same native point-to-plane refinement.
The voxel/gate schedule, 20/20 iteration budgets, accepted-pose keyframe policy,
50% final support, calibration and four score limits remain fixed. Diagnostics
identify each objective and whether its stopping cause is observable. It does
not switch objectives after failure, retry, reset, fit scale or use references.

XYZ/rpy/desk are now evaluated exploratory sequences. The hybrid plans and
the separate unchanged desk2 plane baseline are fixed before acquiring,
preparing, fitting or interpreting reference poses for Freiburg 1 desk2.
This new sequence is the untouched validation for this revision. The earlier
desk result remains a failed validation of the earlier methods; it does not
become held out again. The hybrid protocol commit is
`2248b315674c36dfacf38cca7c70b6b2698071b8`, before desk2 acquisition at
2026-10-11T01:10:22Z. All five new estimates and helper/native bindings were
frozen and verified before reference scoring. No limits were retuned.

## Complete observed outcomes (2026-10-11)

Every row fails at least one fixed acceptance condition. Errors on partial
trajectories describe only the evaluated part, not the ungenerated remainder.
First-failure image indices are zero based; the stopped chain never resets.

| Sequence | Method | Generated / planned | Evaluated | ATE RMSE (m) | 1 s RPE RMSE (m) | First failure |
| --- | --- | --- | --- | --- | --- | --- |
| xyz | static keyframe | 569 / 798 | 566 | 0.798630 | 0.420820 | 569 |
| rpy | static keyframe | 42 / 722 | 42 | 0.442162 | 0.654069 | 42 |
| desk | static keyframe | 26 / 595 | 26 | 0.024602 | unevaluated | 26 |
| xyz | overlap keyframe | 798 / 798 | 795 | 0.459847 | 0.137479 | none |
| rpy | overlap keyframe | 722 / 722 | 716 | 1.473665 | 0.257359 | none |
| desk | overlap keyframe | 266 / 595 | 266 | 0.836153 | 0.452297 | 266 |
| desk | frame-to-frame plane | 526 / 595 | 525 | 0.455415 | 0.105820 | 526 |
| xyz | SVD + plane keyframe | 798 / 798 | 795 | 0.854618 | 0.155599 | none |
| rpy | SVD + plane keyframe | 722 / 722 | 716 | 0.618903 | 0.137033 | none |
| desk | SVD + plane keyframe | 265 / 595 | 265 | 0.426457 | 0.206332 | 265 |
| desk2 | SVD + plane keyframe | 639 / 639 | 624 | 1.045931 | 0.179715 | none |
| desk2 | frame-to-frame plane | 639 / 639 | 624 | 0.523587 | 0.102466 | none |

The preceding xyz/rpy plane baselines have ATE 0.125197/0.529330 m and RPE
0.043704/0.138976 m. None of the complete keyframe variants improves ATE over
those baselines. Hybrid rpy has slightly lower RPE, but still fails both limits;
this does not establish a general trajectory improvement. Hybrid desk fails
at support 0.486208, below 0.5. The hybrid cannot be promoted as a better default
based on this evidence; the original default and every prior artifact remain.

Fresh desk2 covers 21.288169 s, with 639 depth / 640 RGB images prepared.
Both estimators generate all 639 poses and 146,714,333 projected points, and
each retains exact typed depth counts/pixel indices in 12 native IO audits.
Independent f64 projection differs by at most 3.729e-7 m. Fifteen reference
matches are unavailable at indices 19–29, 429, 431–432 and 638, leaving 624
evaluations and 591/609 one-second RPE pairs. Missing references are kept
distinct from tracking failures. Hybrid/plane rotation RPE is 11.217597 /
6.826300 degrees, also showing substantial motion error.

All 12 independent SciPy score audits pass the unchanged 1e-8 m / 1e-6 degree
numerical tolerances. That verifies calculation, not physical accuracy. Both
Python 3.8 and 3.12 suites pass 731 tests, with three expected ONNX skips each.
The [metadata receipt](receipts/2026-10-11_tum_keyframe_validation.json) binds
all input, protocol, generation, scoring, audit, coverage and missing-pair
records. Inclusive missing-pair runs encode both indices increasing by one;
the full unchanged pair lists are retained in the local scored artifacts.

Standalone hash-bound comparisons were generated and visually inspected for
all four sequences under `/workspace/SpatialRust/target/tum-keyframe-reference/`:
`xyz-comparison-report-v2`, `rpy-comparison-report-v2`,
`desk-comparison-report-v2`, `desk2-comparison-report-v2` (PNG, SVG and HTML).
The original v1 exports remain; v2 includes outer artists in its saved bounds
so the rpy legend is not clipped. This changes export framing, not scores.

## Evidence status

Analytic native motion recovery across two keyframe promotions, singular
geometry/lost-chain retention, exact helper binding, fixed settings and wrong
archive root regressions pass, including actual native hybrid motion recovery
through two promotions and singular-geometry rejection. All real outcomes
remain failures against the fixed engineering gates, and the new desk2 hybrid
accuracy is worse than its baseline. Broader processing, test counts and correct
score recomputation cannot replace accurate motion evidence. The maturity
remains 87%, provisional; 90% is unsupported.

The source calibration remains the publisher-recommended approximate ROS
intrinsics for already registered depth, without extra distortion, rescaling
or sensor transforms. This cannot certify measured clock synchronization or
front/rear mounting for the original Autoware bag. Data attribution remains
the TUM RGB-D Benchmark (Sturm et al., IROS 2012), CC BY 4.0. Archives, images,
point-cloud dumps and native binaries are local artifacts and are not committed.

## Replaying without using references for generation

Use the exact archive SHA, committed plan and installed native-library identity
in the evidence receipt. Prepare into a fresh directory with
`scripts/prepare_tum_sequence.py --archive PATH --archive-sha256 SHA --plan PLAN
--output-dir PREPARED`. Preparation retains integer nanoseconds, all indexed
frames, original depth/pixel attributes and per-image hashes; it withholds
reference and accelerometer contents. It is not a subsampled short-sequence run.

Choose the declared adapter CLI (`run_tum_keyframe.py`,
`run_tum_adaptive_keyframe.py` or `run_tum_hybrid_keyframe.py`) with
`--prepared-dir PREPARED --manifest-sha256 SHA --output-dir GENERATED`.
The separate plane baseline uses `run_tum_point_to_plane.py`. Freeze every
estimate, plan and generation code/native binding before scoring. Verify
historical helper hashes against the recorded generation commit rather than
assuming they equal a later checkout.

Only then use `evaluate_tum_sequence.py --archive PATH --estimates ESTIMATES
--estimates-sha256 SHA --plan PLAN --output-dir EVALUATION`, and independently
recompute with `audit_tum_trajectory.py` using its required estimate, evaluation
and reference hashes. Keep fresh output directories and every failed or
unmatched row. A full timeline with missing poses does not constitute successful
full-sequence motion estimation. Audit tolerances cannot be substituted for
the physical accuracy acceptance conditions.

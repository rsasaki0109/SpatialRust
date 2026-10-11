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
become held out again. Hybrid generation, scoring and audits are pending at
this preregistration commit. No result will justify tuning these fixed limits.

## Evidence status

Analytic native motion recovery across two keyframe promotions, singular
geometry/lost-chain retention, exact helper binding, fixed settings and wrong
archive root regressions pass, including actual native hybrid motion recovery
through two promotions and singular-geometry rejection. The preceding real
results are failures, preserved for comparison. Unit tests alone do not
establish accurate real motion. The maturity remains 87%, provisional.

The source calibration remains the publisher-recommended approximate ROS
intrinsics for already registered depth, without extra distortion, rescaling
or sensor transforms. This cannot certify measured clock synchronization or
front/rear mounting for the original Autoware bag. Data attribution remains
the TUM RGB-D Benchmark (Sturm et al., IROS 2012), CC BY 4.0. Archives, images,
point-cloud dumps and native binaries are local artifacts and are not committed.

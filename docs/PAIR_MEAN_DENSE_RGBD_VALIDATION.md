# Initial-correspondence pair-mean fitting

Operational slice 146N. Commit this contract and both complete plans before
implementation or new fitting. The previous gray-only policy completes
798/798 xyz and 722/722 rpy poses, but rpy ATE 0.120216m fails the fixed 0.10m
gate. Preserve the raw, joint, gray-only and external controls and all failures.

## Fixed model

Add an explicit `DenseRgbdPreprocessing::Gaussian3GrayOnlyPairMean` and Python
`preprocessing="gaussian3-gray-only-pair-mean"`. Use exactly the previous
Gaussian3 gray arithmetic and original valid full-resolution fitting depth.
At the supplied initial source-to-target pose, visit full-resolution source
pixels in row-major order. Admit the same original-depth and bilinear
target-depth/filtered-gray correspondences as the native projection rule,
including its original depth-difference threshold and missing-corner behavior.
Accumulate source gray and sampled target gray separately over those rows.

Before changing any fitting pixels, require the existing minimum count and
finite positive means and gains. Scale each entire fitting gray plane by
0.5 divided by its own mean, without clipping. Do this exactly once before
pyramid construction. Streaming accumulation owns no pixel-sized scratch;
independent gain application may use the existing private pool, at most four
workers and serial on WASM. Charge the same four retained raw float64 planes
under the unchanged 128MiB logical preflight. Keep owned Python input copies,
safe APIs, all older selectors/defaults and explicit CPU/GPU boundaries.

The external implementation also uses initial-correspondence means and a 0.5
target, but its depth preprocessing, projection, float arithmetic, derivatives
and stopping differ. See the pinned [odometry source](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/odometry/Odometry.cpp)
and [image scaling source](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/Image.cpp).
This is an independent native component test and claims no equivalence.

Keep final statistics on retained original raw images and the >=30 / >=0.5
final correspondence gate. Keep the 20/10/5 schedule, depth weight 0.968,
0.03m depth difference, 0.1..5m depths, camera settings, derivatives, stopping,
full-cloud support >=0.5 at 0.1m, world composition and typed IO. Identity then
the exact preceding returned increment is the sole seed; any loss preserves
all later rows and ends estimator calls. No time rescaling, retry or reset.

## Evidence and resource gates

Run every original 798 xyz / 722 rpy frame and freeze both full timelines before
either new reference score. Require 100% generation, >=95% reference coverage,
first-pose SE3 ATE <=0.10m and one-second translation RPE <=0.05m. Retain the
three/six publisher gaps and every unmatched RPE pair in their denominators.
Independently audit original raw measurements, seed/world lineage and scores
at the existing tolerances. Do not change the model after seeing its scores.

Test independent mean arithmetic, sampling/depth rejection, empty/zero mean,
finite gains, gain-only brightness correction with original raw diagnostics,
ownership/strides and exact serial/parallel scaling. Prove all three previous
native selectors retain their numerical outputs with the old/new wheels.

Build/package/test root: `E:/SpatialRustBuild/native-dense-pair-mean-v1/`;
reuse `E:/SpatialRustBuild/maturity-80-target/` offline. Bound new package/test
bytes plus incremental cache growth to 2GiB, new scientific results to 1GiB,
owned commands to one hour and CPU workers to four; keep >=20GiB free.
Scientific root: `E:/SpatialRustData/maturity-80/pair-mean-dense-study-v1/`;
prepared/result folders use `pair-mean-dense-v1`. Reuse original causal
PNG/index payloads through checked hard links, with no download, copy or
fallback. Preserve all archives, variants, failures, logs and frozen binaries.
Only exact passing pytest scratch may be removed after passing XML,
absolute-parent and full no-reparse checks.

Native rpy/held-out accuracy, canonical measured clock uncertainty and physical
frame accuracy, both raw Linux hardware classes and full mapping/semantic/
release acceptance remain in the 80% audit. Implementation tests alone do not
establish trajectory or operational acceptance. Deliver tests, complete
evidence, PR and merge for this single family.

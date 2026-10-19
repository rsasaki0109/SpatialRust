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

## Pre-fit implementation validation

Protocol `7606269` precedes implementation. Source `c6fdb50` adds the typed
selector, safe owned Python bridge and independent mean/gain tests. The first
native test compile rejects an ambiguous `Mat3::identity()` in an added test;
`3366ea5` specifies `Mat3<f64>` without changing the estimator. Preserve that
failed compile and its log. The corrected native suite passes 20/20, including
independent initial-pose/depth admission, missing corners, zero/nonfinite mean
rejection before mutation, gain-only brightness correction with original raw
diagnostics and exact parallel/serial multiplication. Native/bridge Clippy and
the WASM compile check pass.

The dedicated installed wheel passes all 106 focused Python cases and the full
suite's 1,100 tests, with three existing optional ONNX skips. The 31 new cases
cover the explicit policy, original raw measurements, ownership/strides,
concurrent calls, fixed normalization/seed/camera/gates and lost-chain audits.
Installed bytes, ICP/support and exact PCD round-trip checks pass.

Fifteen synthetic old/new-wheel cases (five each for raw, joint Gaussian3 and
gray-only Gaussian3) have identical serialized numerical poses and original
diagnostics; their policy tags are checked separately. This is a synthetic
regression check, not a full real-sequence equivalence claim. The new module
SHA-256 is `ee9063970899e05c9b6d8987e2c0d1e19238cf83827356f037b7cf9ad221295c`.

All previous frozen native files retain their hashes. Only two explicitly
owned passing pytest roots are removed after the XML/path/reparse checks,
reclaiming 560,093,288 logical bytes. No real fitting has started at this
checkpoint; trajectory acceptance remains unproven.

## Complete-scene outcome

Workflow source `69482cd` binds the installed module above. Both full timelines
were frozen at 2026-10-19T22:55:40.582373Z before either reference score. Both
finish without failures or reset: 798/798 xyz and 722/722 rpy. See the
[source-bound receipt](receipts/2026-10-20_pair_mean_dense_rgbd.json).

| Fitting policy | xyz ATE / RPE (m) | rpy poses | rpy ATE / RPE (m) | Acceptance |
| --- | --- | --- | --- | --- |
| Native raw causal control | 0.089993 / 0.026591 | 722/722 | 0.114793 / 0.048145 | xyz only |
| Native joint Gaussian3 | 0.082755 / 0.025186 | 349/722 | 0.119374 / 0.049937, prefix only | xyz only; rpy incomplete |
| Native gray-only Gaussian3 | 0.091755 / 0.027265 | 722/722 | 0.120216 / 0.048803 | xyz only; rpy ATE fails |
| Native gray-only pair mean | 0.086962 / 0.025649 | 722/722 | 0.130595 / 0.048722 | xyz only; rpy ATE fails |
| External CPU hybrid control | 0.051174 / 0.023110 | 722/722 | 0.099684 / 0.048494 | both development scenes |

Every policy completes all 798 xyz poses. Pair-mean fitting passes xyz but does
not improve rpy ATE over either the raw or gray-only control, and still fails
the unchanged 0.10m gate. The synthetic gain correction is verified; it does
not establish a trajectory accuracy benefit. Keep the raw default and all
experimental selectors explicit. Do not change target means, weights or gates
after these scores. The joint rpy prefix is not full-sequence accuracy, and
the external control still differs in multiple preprocessing/solver details.

Reference coverage remains 795/798 and 716/722, with all three/six publisher
gaps in the denominator. One-second RPE evaluates 762/768 and 680/692 pairs.
Independent NumPy audits pass all 1,518 admitted causal/raw pairs. Counts and
fractions agree exactly; maximum photometric/depth differences are
2.221e-16 / 8.674e-18, seed differences meet 1e-12 and world differences stay
below 3.853e-14. Independent SciPy score differences stay below 1.118e-15m and
3.980e-13 degrees. All eight owned generation/audit/score commands exit zero;
rpy trajectory acceptance remains false.

Every admitted pair records four workers, 22,743,840 logical working bytes
and 3,072,000 input-copy bytes; these charges are not process RSS or a speed
comparison. Scientific output is 316,877,511 logical bytes, below 1GiB.
New package bytes plus incremental cache growth finish at 25,292,248 bytes;
the maximum successful-command sample is 585,373,429 bytes, below 2GiB.
The 918,149,680 logical PNG/index bytes are shared through checked hard links,
with zero image payload copies or downloads. All 315 prior JSON files and
32 frozen native/wheel files, including eight wheels, retain their hashes.
Archives, old variants and the initial test compile failure remain preserved.

This closes the fixed normalization experiment. Native rpy/held-out accuracy,
canonical measured clock uncertainty and physical frame accuracy, both raw
Linux hardware classes and full mapping/semantic/release acceptance remain
unproven. Operational 80% is not established.

## CI portability amendment

After the study is frozen, source `a2637c8` addresses CI environment failures.
Ubuntu 26.04 cannot set up the requested CPython 3.8; the official build
manifest provides Ubuntu 24.04 x64, so only the legacy Python Linux lanes
are pinned there. Newest-Python Linux coverage retains `ubuntu-latest`.
Keep all four native wheel lanes and every named check.

Four tiny synthetic audit tests explicitly receive a scoped adequate-space
fixture. It is not autouse and is never injected into the real study CLI.
Three new boundary tests prove that the production auditor rejects 0 and
20GiB-minus-one before input reads/output, and admits exactly 20GiB to the
frozen-input check. The production auditor and its 20GiB floor are unchanged.
One added Rust test line is wrapped by rustfmt; estimator code is unchanged.

The portability selection passes 11/11; the updated full installed-wheel suite
passes 1,103 with the same three optional ONNX skips. Its two passing scratch
roots reclaim 600,718,061 logical bytes; combined verified pytest cleanup is
1,160,811,349 bytes. The original scientific source/module bindings and all
measurements remain preserved, with no estimation rerun or gate change.
Failure logs, official-manifest proof, source-equivalence proof, XML and
cleanup records are bound in the receipt's separate `ci_portability` section.

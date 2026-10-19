# Gray-only Gaussian3 fitting with original depth

Operational slice 146M. Commit both full-scene plans and this contract before
new fitting. Preserve the complete [joint Gaussian3 experiment](GAUSSIAN_DENSE_RGBD_VALIDATION.md):
xyz passes at 798/798 and ATE/RPE 0.082755/0.025186m, while rpy loses tracking at
index 349 because original raw support 0.497388 misses the fixed 0.5 gate.
Raw causal and external CPU controls remain fixed. These observations do not
identify a single cause of loss or prove generalization.

## Prospective model

Add an explicit `DenseRgbdPreprocessing::Gaussian3GrayOnly` and Python selector
`preprocessing="gaussian3-gray-only"`. Keep raw and joint selectors, defaults,
their arithmetic and source-bound results. Use the same float64 Gaussian3 gray
outer product [1,2,1]/16 and replicated borders, then the same pyramid and
bilinear derivatives. Copy original full-resolution metric depth into the fit
base: invalid/out-of-range/nonfinite samples stay missing at their own pixels,
with no depth smoothing, neighborhood invalidation or hole fill. This changes
only the depth component of the previous preprocessing block.

Retain original raw levels for final statistics and the >=30 / >=0.5 final
correspondence gate. Charge the same four extra float64 planes (32*width*height)
before allocation, under the unchanged 128MiB logical bound. Preserve the owned
four-array Python boundary, safe API, explicit CPU ownership, deterministic
row tiles (<=4 private workers; WASM serial) and all core/GPU/dependency contracts.

Use the original 20/10/5 schedule, 0.968 depth weight, 0.03m depth difference,
0.1..5m depths, stopping and camera settings. Apply no mean normalization,
rescale or extra extrinsic/undistortion. Identity then the exact preceding
source-to-target increment remains the sole initializer; any loss preserves
remaining rows and ends estimator calls. Keep raw full-cloud support >=0.5 at
0.1m, world composition, timestamps and typed IO unchanged.

## Full-scene acceptance and resources

Run all 798 xyz / 722 rpy frames. Commit the new model before estimation and
freeze both full timelines before either new reference score. Keep 100% pose
generation, >=95% reference coverage, first-pose SE3 ATE <=0.10m and one-second
translation RPE <=0.05m. Preserve all three/six publisher gaps and unmatched RPE
denominators. Independently audit original raw statistics, seed/world lineage
and scores at existing tolerances. Failed prefixes remain incomplete; there is
no threshold change, reset, retry, selection or post-score model adjustment.

Build/package/test root: `E:/SpatialRustBuild/native-dense-gray-gaussian3-v1/`;
reuse `E:/SpatialRustBuild/maturity-80-target/` without overwriting any frozen
module, wheel or executable. Cap new package/test bytes plus incremental cache
growth at 2GiB, scientific results at 1GiB, owned commands at one hour and build/
compute workers at four; keep >=20GiB free. Use existing offline dependencies.
Scientific root: `E:/SpatialRustData/maturity-80/gray-gaussian3-dense-study-v1/`;
prepared/result folders use `gray-gaussian3-dense-v1`. Reuse original causal
PNGs/indexes via checked hard links, with no download, payload copy or fallback.
Preserve all archives, prior variants, failures, logs and source/binary hashes.

Only exact owned passing pytest scratch may be removed after passing XML,
absolute-parent and full no-reparse checks. Verify old/new raw and joint
synthetic numerical results as well as native/Python math, IO and ownership
before real fitting. Keep rpy/held-out, canonical measured clock/physical frame
accuracy, both raw Linux hardware classes and full mapping/semantic/release
acceptance in the 80% audit. Deliver implementation, tests, evidence, PR and merge.

## Pre-fit implementation validation

Protocol `6dc3de1` precedes implementation and fitting; numerical source
`28d1f93` adds the explicit variant while retaining prior selectors and bounds.
Native tests pass 15/15, checking copied depth bits, missing-depth locations,
identical joint/gray-only gray arithmetic, final raw statistics and exact
serial/parallel results including partial tiles. Native and Python-bridge
Clippy and the WASM compile check pass.

Focused Python tests pass 75/75; the full installed-wheel suite passes 1,069 with
three existing optional ONNX skips. The 26 new cases cover independent original
raw statistics, ownership/strides/concurrent calls, identical constant-depth
joint/gray-only outputs, fixed controls, causal state and loss/audit behavior.
Installed wheel bytes, ICP/support and exact PCD round-trip checks pass.

Ten synthetic cases run through the previous and new wheels: five raw and five
joint Gaussian3 cases (identity, integer/subpixel motion, a hole, explicit seed).
Serialized numerical poses and original diagnostics agree exactly, excluding
only policy identifier handling. New module SHA-256 is
`f3a58c85172c8596630e18eaa010e7905a49e65ad5e849db0f7a0d63c1467abc`.
All frozen previous modules, wheels and executables retain their hashes.
Only the two passing pytest scratch roots were removed, reclaiming 478,843,534
logical bytes. These implementation checks precede complete-scene estimation;
they do not establish trajectory acceptance.

## Complete-scene outcome

Workflow source `d1388c6` binds the installed module above. Both complete
timelines were frozen at 2026-10-19T22:02:21.322379Z before either reference
score. All 798 xyz and 722 rpy poses were generated, with no failures or reset.
See the [source-bound receipt](receipts/2026-10-20_gray_gaussian3_dense_rgbd.json).

| Fitting policy | xyz poses | xyz ATE / RPE (m) | rpy poses | rpy ATE / RPE (m) | Acceptance |
| --- | --- | --- | --- | --- | --- |
| Native raw causal control | 798/798 | 0.089993 / 0.026591 | 722/722 | 0.114793 / 0.048145 | xyz only |
| Native joint Gaussian3 | 798/798 | 0.082755 / 0.025186 | 349/722 | 0.119374 / 0.049937, prefix only | xyz only; rpy incomplete |
| Native gray-only Gaussian3 | 798/798 | 0.091755 / 0.027265 | 722/722 | 0.120216 / 0.048803 | xyz only; rpy ATE fails |
| External CPU hybrid control | 798/798 | 0.051174 / 0.023110 | 722/722 | 0.099684 / 0.048494 | both development scenes |

The gray-only policy completes rpy, but its ATE exceeds the fixed 0.10m gate
and the raw control's error. It does not establish an accuracy improvement.
The joint policy's surviving-prefix metrics do not describe its full rpy
trajectory. The external estimator differs in normalization, derivatives,
pyramid and stopping behavior; this comparison proves neither equivalence nor
the individual cause of the joint failure. Keep all three native selectors
explicit and retain the raw default.

Reference coverage is 795/798 xyz and 716/722 rpy; all three/six publisher gaps
remain in the denominator. One-second RPE evaluates 762/768 and 680/692 pairs.
Independent NumPy audits pass all 1,518 admitted causal/raw pairs: counts and
fractions agree exactly, maximum photometric/depth differences are
2.152e-16 / 8.674e-18, seed difference is zero and world difference is below
3.442e-14. Independent SciPy score differences stay below 1.114e-15m and
4.690e-13 degrees. All eight owned generation/audit/score commands exit zero;
that records the completed experiment, while rpy acceptance remains false.

Every admitted pair records four workers, 22,743,840 logical working bytes
and 3,072,000 input-copy bytes. These are explicit algorithm/input charges,
not process RSS or a speed comparison. New scientific output is 315,595,702
logical bytes, below 1GiB. New package bytes plus incremental reused-cache
growth finish at 127,501,415 bytes; the maximum successful-command sample is
606,333,578 bytes, below 2GiB. The 918,149,680 logical PNG/index bytes are shared
through checked hard links, with zero image payload copies or downloads.
All 293 prior JSON files and 30 frozen native/wheel files, including seven
wheels, retain their hashes. Archives and failed variants remain preserved.

This closes the component experiment, with native rpy accuracy still open.
Held-out motion, canonical measured clock uncertainty and physical frame
accuracy, both raw Linux hardware classes and full mapping/semantic/release
acceptance remain unproven. Operational 80% is not established.

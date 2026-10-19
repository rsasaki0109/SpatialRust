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

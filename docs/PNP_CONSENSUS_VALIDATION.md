# Robust-PnP consensus preservation

## Failure and scope

Epic 146D fixes the final stage of deterministic six-point robust PnP. The
selected proper pose, original-order masks/residuals, inlier count and inlier
residual sum are retained through optional inlier refitting. The native API,
sampling sequence, confidence, iteration budget, camera model, initializer,
thresholds and strict proper-rotation validation retain their contracts.
`spatialrust-core`, heavy dependencies and CPU/GPU transfers are unchanged.

The preceding [keyframe study](TUM_KEYFRAME_RGBD_VALIDATION.md) preserves five
failed real-motion cases. A post-freeze prefix with an explicitly deeper LK
pyramid has a numerically valid 54/93-inlier hypothesis, but its final cold DLT
refit raises a proper-rotation error. It is a numerical reproduction, not an
accepted or physically certified trajectory.

A separate 90-point procedural fixture reproduces the same public native/Python
error without sensor data: xorshift64 seed 5, a near plane of thickness .002m,
small fixed rotation/translation, bounded image noise and one-third outliers.
The new Rust regression fails on the parent with that exact rotation error;
both contiguous and strided Python cases fail on the preserved parent wheel.
The original sensor arrays remain external and are not committed.

## Selection and resource contract

Optional refitting may replace the selected model only when it has more inliers,
or equal inliers with a strictly smaller sum of **Euclidean pixel residuals over
inliers**. This is exactly the selection objective already used by RANSAC.
Errors, lower counts, worse equal-count error and exact ties preserve the chosen
hypothesis. No valid hypothesis or too few inliers still returns an error.

The original owned mask/residual vectors are reused. A valid rejected candidate
requires one restoration projection pass; errors leave the original rows intact.
There is no additional full-row output allocation. Selected buffers remain live
through refitting; this is not a peak-memory or speed claim. Result rows always
correspond to the returned pose in original input order. No invalid rotation is
normalized into acceptance, and no identity pose or lost-track reset is added.

Local regression covers the actual procedural failure, refit error, reduced
consensus, behind-camera projections, equal-count error improvement/regression,
more-inlier priority, exact ties, missing hypotheses and output-buffer/row
consistency. Parent planar/near-planar and noisy pose properties remain required.

## Fixed public-data protocol

Implementation and these four plans are committed before any candidate motion:

- `benchmarks/tum-fr1-xyz-rgbd-pnp-consensus-plan.json`
- `benchmarks/tum-fr1-rpy-rgbd-pnp-consensus-plan.json`
- `benchmarks/tum-fr1-xyz-adaptive-pnp-consensus-plan.json`
- `benchmarks/tum-fr1-rpy-adaptive-pnp-consensus-plan.json`

Each binds its prior complete frontend plan. Calibration, geometry, input
associations, tracking/PnP controls and original acceptance limits are identical.
Only the native final-refit policy changes. The stateless and adaptive frontends
are tested on **both** already observed development scenes; neither is held out.
All previous prefixes, errors and complete parent xyz scores remain unchanged.
No wider LK setting is a new full-motion candidate in this slice.

1. Preserve the original installed native extension at
   `E:\SpatialRustBuild\maturity-80-venv\Lib\site-packages\spatialrust\spatialrust.pyd`
   with SHA-256 `7a9d983db9fcd473caf559fdd8785ba52c080eb4944b87a1c54693fd967c92e1`.
   Build/install the candidate into a separate E-drive wheel/venv directory;
   record protocol, source, lockfiles, compiler, wheel and loaded-extension hashes.
2. Preflight at least 20 GiB on the E-drive output filesystem. Reuse source
   images via the checked hard-link helper, with fresh metadata and run paths
   below `E:\SpatialRustData\maturity-80\`. No new dataset download or native
   input payload copy is needed. Keep source and all old result hashes intact.
3. Run each frontend/scene once, retaining all 798 xyz / 722 rpy planned rows,
   support tests and typed IO checks. Freeze all four estimates before new
   reference scoring. No reference initializer, scale fitting, frame selection,
   variable settings or disconnected tracking recovery.
4. Evaluate with the unchanged scorer and independent SciPy audit. Retain all
   unmatched reference rows and uncovered one-second pairs. Use unchanged
   1e-8m/1e-6degree numerical audit tolerances. A case needs 100% generated poses,
   >=95% evaluated poses, ATE RMSE <=0.10m and 1s translation RPE RMSE <=0.05m.
   Keep every individual result; partial prefixes cannot establish improved
   complete-motion accuracy.
5. Separately replay the pinned 93-row diagnostic input through the parent and
   candidate public PnP APIs. The parent proper-rotation error must be observed;
   the candidate must return a proper finite model with >=54/93 inliers under
   the same 2px threshold and consistent owned masks/residuals. This numerical
   check does not satisfy any full-motion or physical calibration gate.
6. Run installed-wheel Python regressions, full native Vision tests/properties,
   Clippy, formatting, relevant portability compilation and all required hosted
   checks before merge. Record storage/work overlap without a throughput ranking.

The candidate venv is separate so none of the old frozen extension paths or
benchmarks is invalidated by installation. Every success/failure is source-bound;
no previous estimate or evaluation is rewritten.

## Results

Before public fitting: the new native regression fails on the parent and passes
after the fix. Full native Vision has 166 passes, 13 property passes, two ignored
isolated performance probes, and Clippy/format checks pass. Parent installed
Python has two expected failures on the new public-API regression. Candidate
wheel and real-motion outcomes are pending.

Canonical Autoware measured clock/physical calibration, accurate complete real
motion, Linux hardware-GPU execution and the public-release operational boundary
remain separate requirements. These unit results do not establish the 80% goal.

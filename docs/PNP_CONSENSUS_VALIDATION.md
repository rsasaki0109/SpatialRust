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

Implementation and all four plans were fixed at source/protocol commit
`cf457fd84d241714f005502a920cad5c4ef2c1d8` before candidate fitting. The
[compact complete record](receipts/2026-10-19_pnp_consensus.json) binds source,
compiler, both lockfiles, wheel/loaded-extension hashes, every case and original
baseline. The original extension remains byte-identical in its separate venv.

| Method / scene | Generated | Evaluated | ATE RMSE (m) | 1s RPE RMSE (m) | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen parent xyz-rgbd | 798/798 | 795 | 0.120235 | 0.033505 | No |
| Frozen parent xyz-adaptive | 240/798 | 237 | 0.056469 | 0.030103 | No |
| Frozen parent rpy-rgbd | 2/722 | 2 | 0.010364 | unavailable | No |
| Frozen parent rpy-adaptive | 2/722 | 2 | 0.007455 | unavailable | No |
| Candidate xyz-rgbd | 798/798 | 795 | 0.122263 | 0.035838 | No: ate |
| Candidate xyz-adaptive | 798/798 | 795 | 0.118821 | 0.028585 | No: ate |
| Candidate rpy-rgbd | 2/722 | 2 | 0.011220 | unavailable | No: complete, coverage, rpe |
| Candidate rpy-adaptive | 2/722 | 2 | 0.005037 | unavailable | No: complete, coverage, rpe |

ATE/RPE on incomplete cases describe only the accepted prefix. They cannot be
compared as complete-motion accuracy or count toward the unchanged full-motion
gate. Every one of the 3040 planned rows is retained, including failure,
reference-gap and uncovered RPE rows. 0 of four candidate cases meet all
original limits. No threshold, reference initializer, scale fit or reset changed.

Adaptive xyz extends the preserved parent 240-frame prefix to all 798 frames,
but still misses ATE. Stateless xyz ATE changes from 0.120235m to 0.122263m;
this fix does not improve its full-motion accuracy. Rpy loses the connected
chain at zero-based row 2: stateless has 25/60 PnP inliers below the unchanged
0.5 fraction, while adaptive fails its fixed inlier/depth-support guard. Later
failure rows are preserved and no disconnected segment is restarted.

All four independent SciPy scoring audits pass the original 1e-8m/1e-6degree
tolerances. Independent checks verify original pixel IDs, f32 metric source
depths (including adaptive 3x3 medians), returned-pose PnP masks and world/keyframe
composition across 288,068 source landmark rows. All 32 typed IO roundtrips are
exact, and all old estimate/score hashes remain unchanged.

The separately pinned 93-row input reproduces the parent's public proper-rotation
error. The candidate returns 54/93 inliers, determinant 1, orthogonality error
2.23e-16 or less and independent residual disagreement below 5.51e-14px. Owned
outputs survive input mutation. This checks numerical validity, not physical
pose accuracy; this wider-LK diagnostic is not a new full-motion candidate.

Installed-wheel Python passes all 837 cases with three optional ONNX skips,
including both contiguous/strided procedural regressions. Parent Python has two
expected failures on these inputs, and the native parent regression also fails.
Candidate native Vision passes 166 unit cases and 13 properties; two isolated
performance probes remain ignored. Clippy and formatting pass. WASM compilation
passes; no WASM execution is claimed. Hosted checks remain required before merge.

The first wheel launch cannot find the installed maturin script. A retry's
cp932 log printer fails before compilation; the stopped child and empty output
are checked before a UTF-8 retry uses the existing absolute launcher. The final
build/install succeeds without a package download or old-environment overwrite.
Python tests overlap real generation on a shared host; no throughput ranking is
made.

Source images share hard links: 1,836,299,360 logical input bytes avoid copies,
with zero new archive/image payload bytes. New run/evaluation/audit files total
403,485,954 logical bytes; the record separately identifies the new wheel/venv
and reused native build cache. This is not NTFS allocated space or a free-space
delta claim. Only the successful dedicated pytest basetemp is removed
(167,190,822 logical bytes); XML/logs, old/new wheels, source data and frozen
results remain. The output free-space floor stays 20 GiB.

Canonical Autoware measured clock/physical calibration, accurate complete real
motion, Linux hardware-GPU execution and the public-release operational boundary
remain separate requirements. These unit results do not establish the 80% goal.

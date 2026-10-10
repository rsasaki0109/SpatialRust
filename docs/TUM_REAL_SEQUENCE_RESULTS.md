# TUM real-sequence results

The fixed official `freiburg1_xyz` sequence now completes **798/798** native
depth-frame generations over **26.594239 seconds**, with **185,553,749** projected
points. The independent publisher trajectory evaluates 795 poses; three remain
unmatched at a measured reference gap. The first-pose-aligned translation ATE
RMSE is **0.658033 m**, with **0.175945 m** one-second translation RPE. These errors
expose substantial drift despite successful processing and geometric support.
They do not demonstrate accurate odometry or an accuracy improvement.

The estimate moves from 86% to **87%, provisional**, on verified complete real
sensor ingestion, typed IO and independently audited full-sequence scoring. The
score measures engineering progress, not library parity. The observed drift,
unmeasured clock synchronization and original Autoware mounting-calibration gate
prevent a 90% claim.

## Source, freezing and calibration

The verified official HTTPS URL is
`https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz`,
redirecting to `webshare.cvg.cit.tum.de`. Runtime revision 9 includes that exact
hostname and the transfer succeeds after publication. The archive is 448,204,271
bytes, SHA-256
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`.
This is a locally recorded hash of official verified-HTTPS bytes; no publisher
archive checksum was supplied. The acquisition receipt and response headers are
retained under `/workspace/SpatialRust/target/tum-reference/official`.

The native protocol/helper commit `b2506d4f1ae92f3bd89ac88f5b4b5dbbd0dff6af`
precedes acquisition and reference interpretation. Its plan SHA-256 is
`8a3515a19466695568ce4a06512e20feea54dbe78c80a23c3f4589ff034ae29b`.
Preparation retains all 798 RGB and 798 depth images, exact integer nanosecond
times and hashes. Every depth frame has a nearest RGB association within 20 ms;
RGB is not used by fitting. Groundtruth is excluded from preparation/generation.
Native estimates are frozen as
`ba322b06c684a6631226875b6f2ea4b7e56745bd9fd014b2bb8b911821e524f0`
before reference extraction and evaluation. The original frozen helper hashes
are preserved rather than rewritten after adding a baseline adapter.

The publisher-recommended ROS default intrinsics and depth counts / 5000 use the
RGB optical frame, with no extra depth correction, undistortion or IR-to-RGB
extrinsic. This is a documented approximation, not newly measured calibration.
Every native coordinate agrees with an independent f64 pinhole formula within
**2.710e-7 m**; this numerical check does not establish physical camera accuracy.
All **15** selected xyz/raw-depth/pixel input and restored PCD pairs retain their
exact types and values through native IO. There are zero generation failures,
no disconnected trajectory reset and no reference-derived initialization.

## Complete post-fit coverage and accuracy

Only the first jointly valid pose defines the SE(3) world gauge. No scale or
best-fit trajectory alignment is optimized. The raw ATE is also retained;
the raw/world-gauge difference is not a before/after algorithm improvement.

| Frozen result | SpatialRust | Open3D diagnostic baseline |
| --- | ---: | ---: |
| Planned / generated / evaluated poses | 798 / 798 / 795 | 798 / 798 / 795 |
| Raw translation ATE RMSE | 2.453987 m | 2.453915 m |
| First-pose translation ATE RMSE | 0.658033 m | 0.657967 m |
| Median / p95 / max translation ATE | 0.610299 / 0.967970 / 1.014826 m | 0.610175 / 0.967947 / 1.014809 m |
| Evaluated / planned one-second RPE pairs | 762 / 768 | 762 / 768 |
| Translation RPE RMSE | 0.175945 m | 0.175942 m |
| Rotation RPE RMSE | 3.977720° | 3.977490° |
| Whole-process wall time on this machine | 127.734 s | 128.983 s |
| Peak process RSS | 150,272 KiB | 348,004 KiB |

The timing includes source/code verification and selected IO. Open3D also has
different import/library and stopping overhead; these two single-run process
observations are not a speed or memory ranking of the registration kernels.

Indices 201, 202 and 203 have timestamps 1305031108867534000,
1305031108903540000 and 1305031108935116000 ns. Their enclosing reference poses
are 1305031108835700000 and 1305031108945800000 ns: **110,100,000 ns** apart.
They cannot meet the fixed 20 ms limit to each bracket end, and are not
extrapolated or removed from coverage. The unevaluated RPE pairs are
171→201, 172→202, 173→203, 201→231, 202→232 and 203→233. The 67,582,000 ns span
between unmatched depth frames is distinct from the 110,100,000 ns reference
gap. Both measurements are retained.

## Baseline and independent calculation audit

The separate Open3D plan/code is committed as
`82a3e950cace9f1cbcea472d999a5a4870e3e115` before baseline fitting. The native
result had already been seen, so this is a same-sequence diagnostic, not another
blind holdout. Its calibration, frame/depth limits, common native voxel points,
distance gates, identity pair initialization and iteration budgets are checked
against the pre-acquisition native plan. Open3D requests zero relative stopping
thresholds; native uses its fixed absolute stopping criteria. Open3D's actual
update count is not exposed by the legacy API and is not invented. Native stage
traces retain 480/797 coarse and 515/797 fine iteration-limit outcomes.

Both methods preserve all planned frames and use the same point support
acceptance rule. Open3D estimates are frozen as
`ca6c8ebca3196320c2aa0180edbebcb34f083728befaf5c616364f1dde9d02fe`
before their post-fit evaluation. Their nearly overlapping errors corroborate
the shared point-to-point tracking limitation under this protocol; they do not
prove a physical cause, universal library parity or successful trajectory recovery.

`scripts/audit_tum_trajectory.py` imports neither the generator nor the scoring
helper. It independently parses Decimal times, uses SciPy 1.18.1 quaternion
SLERP/rotation magnitude, reconstructs the first-pose gauge, ATE/RPE and every
coverage denominator, and checks all rows against frozen scores. Both actual
audits pass fixed 1e-8 m and 1e-6° comparison tolerances. These are agreement
tolerances for numerical calculations, not physical sensor uncertainty.

After adding the explicit adapter, the native default is replayed on the same
real inputs to check that the adapter does not change native pose/IO outcomes.
That is a regression check on already scored data, not a new accuracy study.
Both Python 3.8 / PyArrow 17 and Python 3.12 / PyArrow 26 pass **682** applicable
tests, with three expected ONNX skips on each; 35 TUM regressions include method
identity checks so a baseline cannot silently replace the native default.

## Review artifacts and remaining gates

Frozen inputs/results and native/Open3D audits are under
`/workspace/SpatialRust/target/tum-reference`. The standalone PNG, SVG and HTML
comparison is `report-v2/report.html`. Writable Matplotlib/font caches are
configured for the managed environment; the initial report with fallback-cache
warnings is preserved. Reproduce the figure with
`scripts/plot_tum_trajectory.py --manifest target/tum-reference/plot-manifest-v1.json --output-dir NEW_DIRECTORY`.
Set `MPLCONFIGDIR=/workspace/.spatialrust-env/matplotlib` and
`XDG_CACHE_HOME=/workspace/.spatialrust-env/font-cache` before rendering.

The compact review record is
[`receipts/2026-10-10_tum_real_sequence.json`](receipts/2026-10-10_tum_real_sequence.json).
Reference trajectory/data attribution: TUM RGB-D Benchmark, Jürgen Sturm,
Nikolas Engelhard, Felix Endres, Wolfram Burgard and Daniel Cremers,
*A Benchmark for the Evaluation of RGB-D SLAM Systems* (IROS 2012).
The [official overview](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license)
states CC BY 4.0 for data and BSD-2-Clause for accompanying source unless stated
otherwise. Derived plots retain that data attribution. Official images,
archives, point-cloud dumps and native binaries are not committed.

The original Autoware bag still has `registration_ready:false`: TUM/L515
calibration cannot supply its measured clock or front/rear-to-body extrinsics.
The visible drift also leaves accurate real motion estimation unresolved.
Those gaps remain material for 90%; successful processing and matching another
implementation's failing result do not close them.

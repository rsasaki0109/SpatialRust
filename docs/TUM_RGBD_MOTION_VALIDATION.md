# Native image-tracking RGB-D motion protocol

Date: 2026-10-19. Operational v1.3 slice 146A. Both trajectories complete,
but both fail the unchanged 0.10 m ATE acceptance limit.

The existing depth-only study processes the public TUM RGB-D Freiburg 1 XYZ
sequence but its point-to-plane trajectory exceeds the fixed 0.10 m ATE limit.
This separate adapter connects existing native grayscale, Shi-Tomasi detection,
forward/backward pyramidal Lucas-Kanade tracking and RGB-D PnP RANSAC. It keeps
the previous depth-only default and every prior failed outcome.

## Fixed inputs and interpretation

The [plan](../benchmarks/tum-fr1-xyz-rgbd-pnp-plan.json) binds the unchanged
original protocol by raw SHA-256. Registered depth uses the publisher's
recommended ROS-default pinhole approximation, 640x480 images and 0.0002 m per
stored depth unit. Do not apply depth rescaling or IR-to-RGB extrinsics again.
Nearest RGB associations retain their signed time deltas and the original
20 ms ceiling. This is published registered RGB-D data, not newly measured
physical calibration or measured clock synchronization.

Official source:
`https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz`.
The TLS-verified official redirect supplies 448,204,271 bytes with SHA-256
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`.
This matches the already observed development input. It is not a new scene or
an untouched holdout. Images, archives, trajectories and derived sensor data
remain outside the repository.

Publisher interpretation and license:
[TUM file formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats),
[TUM overview and CC BY 4.0](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license).
Attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers,
*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.

## Frozen generation and acceptance

Use every indexed depth frame. Decode and verify both depth and associated RGB
bytes, then select at most 300 native Shi-Tomasi features in the previous
image. Retain valid source depths and native forward/backward-consistent LK
tracks. Fixed controls include three pyramid levels, radius five, 0.75 px
round-trip error, at least 20 tracks, 2 px PnP residual, at least 15 inliers and
50% inlier support. Native PnP retains its existing deterministic defaults:
seed zero, confidence 0.99, at most 2,000 RANSAC iterations.

PnP maps previous-camera points into the current camera. Invert that rigid
transform exactly once, then use the unchanged right-composed world chain.
Retain the original full-cloud 50% distance-gated support test at 0.1 m. A lost
chain never resets, retries a different objective or uses reference poses.
All planned failures remain in the frozen timeline. Typed original depth and
pixel-index IO audits and independent f64 projection checks remain in place.

Only after estimates and their source/native/plan hashes are frozen may the
reference trajectory be read. Keep the existing first-pose-only SE(3) gauge,
no scale fit, all coverage denominators, the 20 ms reference bracket limit and
one-second RPE. Acceptance remains 100% generated poses, at least 95% matched,
ATE RMSE at most 0.10 m and translation RPE RMSE at most 0.05 m. Independent
SciPy scoring uses the unchanged numerical audit tolerances. No threshold is
relaxed after seeing a result.

## API and boundary validation

`track_points_forward_backward` accepts uint8 grayscale images and finite
float64 Nx2 coordinates, owns inputs before releasing the GIL, and returns
owned next coordinates, validity and round-trip errors in original row order.
Invalid tracks can have infinite errors; their coordinates are not usable
motion observations. The native LK options reject more than 32 levels before
an unrepresentable shift or oversized pyramid allocation, and reject windows
that cannot fit isize. Patches larger than the image fail without iterating
the enormous window.

Analytic translation, strided arrays, ownership, empty shapes, degenerate and
outside-image tracks, nonfinite coordinates, configuration bounds, protocol
binding, RGB checksum/path/time association, noncommuting inverse direction,
lost-chain retention and the unchanged point-support guard are tested. This
does not add a GPU path or make a cross-library speed claim.

## Reproduction

Use an installed wheel built from the current source and the repository's
frozen dependency graph. Keep all data/results outside the repository; the
following are local absolute paths, not portable defaults:

- source: `E:/SpatialRustData/maturity-80/tum-fr1-xyz/rgbd_dataset_freiburg1_xyz.tgz`;
- prepared input: `E:/SpatialRustData/maturity-80/tum-fr1-xyz/prepared-rgbd-v1`;
- isolated native runtime: `E:/SpatialRustBuild/maturity-80-venv`.

Run `scripts/prepare_tum_sequence.py` with the exact archive hash and fixed RGB-D
plan. Run `scripts/run_tum_rgbd_odometry.py` with the prepared manifest hash and
a fresh output directory. Freeze estimates before `scripts/evaluate_tum_sequence.py`,
then run `scripts/audit_tum_trajectory.py` with each exact result/reference hash.
The same unchanged plane baseline can be generated separately using its own
plan and `scripts/run_tum_point_to_plane.py`.

The canonical Autoware clock/front/rear calibration gates remain separate and
unresolved. This public-data study cannot certify those artifacts, broad GPU
coverage or the complete operational maturity target.


## Observed results and independent audit

The protocol was committed as `0cbd2a60c7945d700ea33bb2c5dfa7b7bed90f2a` before first
image-tracking/PnP fitting. Both full trajectories were frozen before reading
reference pose contents for this study. No failed/unmatched row was removed,
and neither limits nor fitting settings were revised after the result.

| Measurement | Native RGB-D image/PnP | Native point-to-plane |
| --- | ---: | ---: |
| Planned poses | 798 | 798 |
| Generated poses | 798 | 798 |
| Reference matched poses | 795 | 795 |
| Unmatched reference poses | 3 | 3 |
| Planned 1 s RPE pairs | 768 | 768 |
| Evaluated RPE pairs | 762 | 762 |
| Projected points | 185553749 | 185553749 |
| Exact typed IO audit frames | 15 | 15 |
| First-pose ATE RMSE (m), limit 0.10 | 0.120234832 | 0.124991768 |
| 1 s translation RPE RMSE (m), limit 0.05 | 0.033505267 | 0.043714717 |
| All acceptance conditions | **fail: ATE** | **fail: ATE** |
| Unisolated generation wall (s), not a speed ranking | 346.91 | 194.27 |

Both methods preserve all 798 planned poses and 795 reference matches. Three
unmatched poses at indices 201, 202 and 203 cross the existing 110.1 ms
reference gap; six planned RPE pairs remain unevaluated. Generation success and
lower error in this one study do not establish accurate-motion acceptance.
The image adapter remains explicit; the ordinary depth-only default is unchanged.

Separate SciPy 1.17.1 recomputation agrees with both complete score/coverage
records under the unchanged 1e-8 m / 1e-6 degree tolerances. Those are numerical
audit tolerances, not physical sensor uncertainty. The single observed wall
times include file/source checks and different work; own regression/probe work
overlapped candidate generation. They cannot establish a throughput ranking.

Local validation passes 160 native vision library tests, 753 full Python tests
and installed-wheel stubtest. Three optional ONNX tests skip; the two ignored
native cases are isolated performance surveys. The separate 100-case focused
run is a subset and is not added to the full Python count. Both native/binding
Clippy runs deny warnings; all-feature WASM vision compiles with warnings denied.
WASM execution is not claimed. Native runtime is Windows x86_64, Rust 1.98.1,
CPython 3.12.10, NumPy 2.3.4, release/thin-LTO/one-codegen-unit.

The [compact receipt](receipts/2026-10-19_tum_rgbd_image_motion.json) records
every aggregate, condition, estimate/evaluation/audit hash and source/native
binding. Local source hashes retain actual checkout bytes; protocol-commit
blob hashes also verify the same source after LF/CRLF normalization. Dataset
protocol JSON retains exact committed LF bytes under existing `.gitattributes`.
Four restored local CRLF protocol files initially failed original hash checks;
restoring their unchanged Git bytes fixes the checks without relaxing identity.

Full trajectories, source images, raw logs and score audits remain under
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/`, in separate `run-rgbd-v1`,
`run-plane-v1`, `evaluation-*-v1` and `audit-*-v1` directories. The input-only
preparation withheld ground-truth and accelerometer contents. The installed
native binary is hash-checked against its exact wheel before and after fitting.
No local adapter model name is recorded in published evidence.

This closes the native image/Python workflow and auditability slice. The
operational 80% target remains active: accurate motion, actual multi-GPU
execution and the complete v1.3 public-release boundary still require evidence.

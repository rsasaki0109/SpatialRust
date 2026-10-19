# Native image-tracking RGB-D motion protocol

Date: 2026-10-19. Operational v1.3 slice 146A. The first image-tracking fit is
pending; this protocol does not establish successful real-motion accuracy.

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

# Epic 105: camera calibration contracts

Epic 105 places calibration ownership in `spatialrust-camera`. The common
`CalibrationOptions` and `CalibrationReport` contracts make robust thresholds,
iterations, convergence, RMS, max residual, and observation counts explicit.

Delivered solvers cover:

- robust `fx/fy/cx/cy` fitting from known camera-space point observations;
- Kannala–Brandt4 four-coefficient fisheye angle fitting and round-trip mapping;
- stereo translation fitting for a supplied relative rotation;
- hand-eye translation fitting for a supplied hand-eye rotation under `AX = XB`;
- sparse world-point bundle adjustment with fixed calibrated camera poses.

Synthetic tests include a mono outlier, exact fisheye coefficients, non-trivial
stereo rotation, hand-eye motions about three axes, and two-view BA convergence.
All supplied rotations are checked for finite coefficients, orthonormality, and
positive unit determinant. Singular normal equations and invalid observation
indices return typed errors rather than panicking.

The fixed-camera BA slice establishes data ownership and residual behavior
without pulling Ceres or another native optimizer into default builds. Joint
camera-pose/intrinsics refinement can extend the provisional contracts later.

The OpenCV 4.10 comparison uses `projectPoints` and
`fisheye.projectPoints` to generate convention-compatible observations. On the
reference run, SpatialRust recovered pinhole parameters within `3.41e-13` and
fisheye coefficients within `4.62e-14`, below the `1e-8` gates.

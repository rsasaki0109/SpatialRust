# Epic 107: robust visual and RGB-D odometry

Epic 107 composes the calibrated geometry delivered by Epic 88 with mapping
contracts from Epic 93. It adds deterministic grid selection, forward/backward
LK diagnostics, scale-explicit monocular odometry, metric RGB-D odometry, and
optional mapping bridges. No runtime, codec, or device dependency enters the
portable algorithms.

On a deterministic 48-track RGB-D scene with one invalid-depth row,
SpatialRust recovered translation with maximum error
`1.0408919004500916e-09` metres. Against OpenCV 4.10 `solvePnPRansac`, maximum
translation difference was `2.1377243378251087e-08` metres and maximum rotation
matrix difference was `3.2804083622441615e-10`; both accepted 48 inliers and
SpatialRust reported the one rejected depth explicitly.

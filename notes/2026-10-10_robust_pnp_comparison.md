# Python robust PnP and OpenCV failure study

Exposed the existing Rust six-point RANSAC PnP through a new Python API. Float64
Nx3/Nx2 correspondence scalars are validated and owned before releasing the GIL.
Return values are proper object-to-camera R/t plus owned full-row bool inliers
and pixel residuals. Settings require positive threshold/budget, confidence in
(0,1), positive image dimensions and finite coordinates. Invalid projection
uses the native maximum finite f64 sentinel; acceptance is not pose confidence.
No OpenCV production dependency or behavior change to ordinary solve_pnp.

Verified 397 Python tests, 11 Rust geometry tests, strict extension Clippy and
stubtest, plus 2 optional direct-comparison tests. New regressions cover 20%
incorrect correspondences, independent reprojection residuals, full-row masks,
proper recovered pose, strided inputs, reproducible seeds, input immutability,
owned result lifetime and GIL release. The release extension was rebuilt using
the repository target directory to isolate previous source-bundle build caches.

Artifacts: `/workspace/SpatialRust-python-delivery/target/opencv-pnp-failure-study-final/`.
240 runs, 4 geometries × 3 conditions × 5 seeds × 4 methods. Both consume identical
object/image arrays, pinhole intrinsics, 3 px threshold, .99 confidence, cap 300.
OpenCV 4.12.0 ITERATIVE uses different initialization and RANSAC sampling than
native DLT/six-point RANSAC; numerical work is not identical. All-row independent
generating-pixel RMSE, positive-depth fraction, generating-pose errors, returned
inlier count, errors and input/code/native hashes are recorded. Thread settings
and versions are recorded, and native-file changes during a run are rejected.

Volumetric 30% wrong correspondences: both plain methods recover 0/5, both RANSAC
methods 5/5. Clean/noisy volume: all methods 5/5. Thin geometry with 1 px noise:
native plain/RANSAC 0/5, OpenCV plain 4/5 and RANSAC 3/5. Exact planar clean cases:
native methods error in all 5, OpenCV methods recover all 5. Planar noisy cases
remain difficult even for OpenCV (4/5 plain, 2/5 robust). Line poses are ambiguous;
their generating-pose error cannot establish physical accuracy. Outlier sampling
cannot repair a rank-deficient DLT initializer: planar-aware initialization is
the next concrete improvement. No real-camera or universal ranking claim.
Provisional maturity 68%; long-term target 90%.

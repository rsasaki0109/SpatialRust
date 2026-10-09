# Scale-independent proper rigid estimation

A new regression reproduced an incorrect 180-degree rotation at coordinate scale
1e-4: the previous cross-covariance solver squared the matrix and discarded
eigenvalues below absolute 1e-12. Small valid covariances disappeared, and planar
covariances also required treatment of their missing singular direction.

The public estimate_rigid_transform API now solves the same paired-point proper
least-squares problem with Horn's unit-quaternion formulation. It normalizes the
cross covariance before a symmetric 4x4 Jacobi eigensolve, avoids squaring the
condition number, and uses a relative convergence tolerance. Centroids and
centered differences are computed in f64. Translation is corrected with the
actual returned f32 rotation. No dependency or public signature was added.
Nonfinite inputs, unrepresentable translation and failed eigensolve return None.
Collinear/coincident correspondences remain nonunique; a returned pose does not
certify observability or correct matching.

Rust tests cover volume/plane at scales 1e-4, 1 and 1e4, a 20000-point cloud at
origin (10000,-20000,30000), half turns, reflections (proper rotation required),
invalid inputs, and unrepresentable translations. All 23 registration tests with
all CPU registration features pass. Strict registration/extension Clippy passes.

Python independently fits each first-update correspondence set with NumPy SVD,
checking returned rotation, determinant, orthogonality, correspondence counts and
centroid consistency. The 24 parameter combinations use five seeds each (120
fits), covering scale, plane/volume, large origin and noise. Exact f32 nearest
distance ties are excluded from these solver comparisons because native traversal
and NumPy input-order tie breaking select different equally near points; separate
search tests cover the native tie contract. All 278 Python tests pass with the
rebuilt release ONNX-enabled extension.

Public cloud_bin_0/1 candidate validation was rerun with cap30, coarse .1 m,
fine/evaluation .02 m and translation/rotation thresholds 1e-4. Independent runs
agree and aligned XYZ roundtrips exactly. Numerical changes intentionally change
poses/support slightly; this is not an output-preserving optimization and there
is no verified ground-truth accuracy claim. It affects algorithms using the shared
paired rigid estimator, including ICP and feature-based registration.

Artifacts:
/workspace/SpatialRust-python-delivery/target/public-icp-stable-estimation/report.html
/workspace/SpatialRust-python-delivery/target/public-icp-stable-estimation/receipt.json

Subjective maturity: 54%, provisional. A multi-condition convergence/failure
study with explicit assessment of false convergence remains toward 55%.
Remote CI is unavailable because GitHub API access is denied by policy.

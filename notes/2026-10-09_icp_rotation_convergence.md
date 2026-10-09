# Rotation-aware ICP convergence

The previous transform stopping condition checked translation alone. A pure
rotation update could therefore report convergence regardless of its angle.
The condition now requires both translation length (coordinate units) and
shortest rotation angle (radians) below the existing transformation_epsilon.
No public fields or defaults change. Some inputs can consequently require more
iterations or reach the iteration limit rather than report convergence.

The angle uses f64 hypot and atan2 of quaternion components, retaining small
rotations even when the f32 scalar component rounds to one. Equivalent q/-q
representations give the same angle. Translation length also uses f64 hypot.
The fitness_epsilon documentation now describes its existing absolute mean
squared distance threshold; its behavior was not changed to improvement.

Validation: 7 registration tests and 18 search tests passed, including a centered
pure-rotation ICP fixture that must not converge after its first nonzero update,
and small-angle/sign-equivalence helper checks. Registration strict Clippy passed.
The release Python extension was rebuilt with ONNX Runtime; all 219 Python tests
passed.

Public-data rerun: Open3D DemoICPPointClouds cloud_bin_0/1, 198835/137833 points,
three supplied poses, coarse gate 0.1 m, fine and evaluation gates 0.02 m,
30 iterations per stage. The full alignment report equals the prior run after
excluding source_file and target_file paths. All three candidates remain
nonconverged. Optimized and independent runs agree, and aligned XYZ survives
PCD roundtrip exactly. This fix does not explain this pair's iteration-limit exit.
There is no independently verified ground truth and no accuracy claim.

Local artifacts:
/workspace/SpatialRust-python-delivery/target/public-icp-rotation-convergence/receipt.json
/workspace/SpatialRust-python-delivery/target/public-icp-rotation-convergence/report.html

Remote CI status is unavailable because GitHub API access is denied by policy.
Subjective project maturity remains 50%; this is not a measured compatibility score.

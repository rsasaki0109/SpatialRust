# Conditional geometry information

Added NumPy CPU geometry diagnostics for point-to-point fixed correspondences
and point-to-plane unit-normal constraints. Points are centered before averaging
to retain small geometry at large origins and normalized by RMS radius. Parameters
are rotation about that centroid (radians) then translation divided by radius;
coincident points use a documented 1 m normalization fallback. Mean JᵀJ spectra,
relative effective rank at 1e-6, conditional condition number and weak eigenvectors
are saved. JSON never emits infinity. HTML plots six relative eigenvalues and
explains the assumed objective and scope. Optional global CLI integration uses
full original source/target geometry, preserving registration behavior.

Verified 380 Python tests, including exact analytical ranks: line point-to-point
5, coincident 3, planar point-to-point 6, planar surface normals 3. Both models
match independent central finite-difference rigid-motion Jacobians. Scale,
large-origin, axis rotation and normal-sign invariance are tested. A symmetric
ring has local rank six despite multiple global poses. This distinguishes local
fixed-pair conditioning from global uniqueness; support or rank cannot certify
an estimated pose. Input overflow, nonfinite/unit normals, report mismatch and
exclusive/partial-output rollback are checked.

Public full workflow:
`/workspace/SpatialRust-python-delivery/target/public-global-geometry-validation/`.
It still chooses seed 8 and exactly the prior global run's 123,511 forward and
129,116 reverse supported points, preserving 198,835 output points. Conditional
point-to-point ranks are both 6; source/target condition numbers 3.76725/3.02220.
Separate target-normal artifacts:
`/workspace/SpatialRust-python-delivery/target/public-target-surface-information.json`
and `.html`, rank 6 and conditional condition number 5.03703.

These spectra use whole clouds, not actual retained ICP pairs. They do not model
correspondence uncertainty, capture basins, physical overlap or statistical noise.
Surface-normal orientation sign cancels in JᵀJ, but wrong normals still change
the objective. Existing reports without diagnostics remain readable with the
stdlib-only renderer; geometry sections additionally need NumPy. No native API
or runtime wheel dependency changed. Provisional maturity 66%, target 90%.

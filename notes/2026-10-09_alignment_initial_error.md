# Initial pose error sweep

Run `python scripts/study_alignment_initial_error.py` using the installed
SpatialRust wheel and NumPy. It overwrites synthetic inputs/results under
`/workspace/SpatialRust-python-delivery/target/prior-basin/`.

Five fixed random seeds each generate 400 uniform points. Source points have
60-degree rotation and translation (5, -2, 0.5) metres. Perturbations are applied
as target-frame perturbation multiplied by the true source-to-target inverse.
Translation perturbation stays (0.03, -0.02, 0.01) metres, and angular error is
0, 3, 10, 20 or 40 degrees. Voxel leaf is 0.025 m, correspondence gate 0.15 m,
and iteration budget is 100 per stage. Rotation errors use SO(3) projection.
Success means rotation error below 1 degree and transform translation error
below 0.01 m; these thresholds are study choices, not library guarantees.

| Angular perturbation | Successful trials |
| --- | --- |
| 0 degrees | 5/5 |
| 3 degrees | 5/5 |
| 10 degrees | 5/5 |
| 20 degrees | 5/5 |
| 40 degrees | 0/5 |

One failed 40-degree trial reported convergence. Failed trials had support
40.5–48.25%, rotation error 34.84–39.93 degrees and transform translation error
3.11–3.65 m. A wrong rotation changes the inverse transform's translation for
the offset source, so the translation error is not an independent input sweep.
These samples demonstrate limited initialization tolerance; they do not locate
a universal convergence boundary. More geometry, noise and sampling conditions
are needed. This is synthetic evidence, not an external-library comparison.

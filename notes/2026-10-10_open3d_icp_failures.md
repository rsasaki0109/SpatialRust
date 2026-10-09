# Controlled Open3D ICP comparison

Artifacts: `/workspace/SpatialRust-python-delivery/target/open3d-icp-failure-study-final/`.
468 registrations, 234 paired conditions; both libraries recover 138 poses and
all recovery classifications agree. No runtime errors. All SpatialRust runs
use their complete 100-update budget. Open3D's public result does not expose its
executed update count; zero relative stopping thresholds disable normal stopping.

Both libraries consume identical seeded f32 points; their updates use f32 and
f64 respectively. Gates .3/.6/1.2 m; prior errors 0/20/40 degrees; five seeds
for clean, noisy, nearby-outlier, random-replacement and partial-source clouds,
one symmetric ring. Priors are constructed using truth plus perturbation: these
are controlled local-ICP tests, not an independent initialization pipeline.
Assessment uses SciPy nearest distances at .05 m on original point counts, not
the libraries' differing fitness definitions. Known synthetic poses assess
recovery (<1 degree, <.01 m). Ring symmetry makes generating-pose error
unobservable physically. The script records every pose, status and source/native
hash, library versions and single-thread environment; optional tests passed 3/3.

Evidence of shared assumptions: gate .6 allows all five clean 40-degree cases
to recover, but all five near-outlier 0-degree cases fail. Shrinking the gate to
.3 recovers all five near-outlier cases with 0/20-degree priors, while only 1/5
clean 40-degree cases recover. Broad gates help the capture basin but admit
misleading correspondences. This is compatible with the earlier trim study:
discarding large residuals can help outliers yet remove useful pairs under poor
priors. Neither library escapes this tradeoff merely by using ordinary ICP.

No PCL/OpenCV direct result, real-sensor ranking, global registration result or
universal timing advantage is established here. Maturity advances provisionally
from 57% to 58% for this reproducible external failure assessment, not parity.

# Candidate search and preprocessing are conditionally complementary

Run `python scripts/study_alignment_candidates.py` with an installed SpatialRust
wheel and NumPy. It overwrites synthetic artifacts under
`/workspace/SpatialRust-python-delivery/target/candidate-study/`.

Five seeds reuse the 320 matching plus 80 nearby replacement source points,
5 mm Gaussian noise per axis and 40-degree initial pose error from the filter
study. Each trial tests additional target-frame rotations -40, -20, 0, 20 and
40 degrees applied to the same initial prior. At each search gate (.3 or .6 m),
selection maximizes distance-gated support on the original 400-point source at
fixed evaluation distance .05 m; ties use smaller gated RMSE then stable order.
Known generating poses are used only after selection to assess correctness.

| Preprocessing | ICP gate (m) | Single original prior | Selected from 5 |
| --- | --- | --- | --- |
| None | .3 | 1/5 | 5/5 |
| SOR k20, std multiplier .5 | .3 | 2/5 | 4/5 |
| None | .6 | 0/5 | 0/5 |
| SOR k20, std multiplier .5 | .6 | 0/5 | 0/5 |

All 100 candidate runs completed without errors. Twenty selected indices were
verified against support/RMSE ordering, independent of their correctness labels.
The candidate grid deliberately contains a near-correct pose in this experiment;
these results do not establish general global-registration capability. Original
source evaluation prevents preprocessing from hiding removed hard points.

Candidate search improves narrow-gate recovery here, while SOR can discard
useful geometry and neither combination rescues the broad gate. Thus combining
algorithms does not imply an improvement across conditions. It also increases
compute: summed five-candidate time was about .32–.51 seconds per five-seed
filter/gate group, versus .06–.13 seconds for the original-prior candidates.
These single-run timings include registration and original-source evaluation,
exclude preprocessing and are illustrative, not a performance benchmark.
Real geometry, candidate coverage and additional outlier distributions remain
unverified; proximity scoring itself can prefer incorrect poses.

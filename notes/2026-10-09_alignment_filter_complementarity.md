# Density filters before ICP: limited complementarity

Reproduce with `python scripts/study_alignment_filters.py` using an installed
SpatialRust wheel and NumPy. Synthetic results and HTML are overwritten under
`/workspace/SpatialRust-python-delivery/target/filter-study/`.

The same five seeds from the gate study contain 320 genuine source points and
80 nearby nonmatching replacements, with 5 mm Gaussian noise per axis. Initial
pose error is target-frame 40 degrees plus (0.03, -0.02, 0.01) m. Each filtered
source is registered at gates .3, .6 and 1.2 m, leaf .025 m, 100 iterations per
stage, and fixed evaluation gate .05 m. Exact float32 coordinate membership
tracks which genuine and replacement points survive. Pose correctness uses
SO(3)-projected rotation error below 1 degree and translation error below .01 m.

| Filter | Correct at .3 m / 5 | .6 m / 5 | 1.2 m / 5 |
| --- | --- | --- | --- |
| None | 1 | 0 | 0 |
| SOR k20, std multiplier 1 | 0 | 0 | 0 |
| SOR k20, std multiplier .5 | 2 | 0 | 0 |
| Radius .3 m, minimum 2 neighbors | 0 | 0 | 0 |
| Radius .4 m, minimum 5 neighbors | 2 | 0 | 0 |

All 75 trials completed without execution errors. SOR multiplier .5 retains
258–288 of 320 genuine points, but also 14–36 of 80 nonmatching points. Density
filtering sometimes complements the narrower ICP gate, but does not rescue
the broad gates here. Sparse genuine regions can be removed while clustered
nonmatching regions survive; this challenges the assumption that nonmatching
points are necessarily sparse. It is not proof of a universal best filter.

JSON reports both filtered-source diagnostics and support after applying the
estimated pose to the original 400-point source. Removed points remain in the
original evaluation denominator. This prevents removal alone from inflating
reported source support. Results are limited to five synthetic uniform seeds;
no real-sensor accuracy or external-library advantage has been established.

The generated HTML now shows paired retention bars: genuine points retained
out of 320 and nonmatching points retained out of 80. Each displays the mean,
seed range and number of completed filtering runs, alongside raw counts and
registration results. These bars use known synthetic membership, not predicted
inlier labels. Filter retention is repeated across search-gate rows because
the same filtered cloud is reused; those rows are not independent filter trials.

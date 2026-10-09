# Distance gate affects optimization and diagnostics

Reproduce with `python scripts/study_alignment_distance_gate.py`, an installed
SpatialRust wheel and NumPy. Synthetic inputs/results are overwritten under
`/workspace/SpatialRust-python-delivery/target/gate-study/`.

Five fixed seeds generate 400 fully overlapping uniform points, with source
rotation 60 degrees and translation (5, -2, 0.5) metres. Initial inverse pose is
left-multiplied by a target-frame 40-degree rotation and translation
(0.03, -0.02, 0.01) metres. Leaf is 0.025 m and budget 100 iterations per stage.
Success requires SO(3)-projected rotation error below 1 degree and transform
translation error below 0.01 m. These are study thresholds, not API guarantees.

| Gate (m) | Correct / 5 | Converged / 5 |
| --- | --- | --- |
| 0.05 | 0 | 4 |
| 0.15 | 0 | 1 |
| 0.3 | 1 | 3 |
| 0.6 | 5 | 5 |
| 1.2 | 5 | 5 |

At 0.3 m, four wrong poses still had forward support 90.5–92.75% and reverse
support 90.5–92.25%. Rotation error was 32.60–39.96 degrees and transform
translation error 3.011–3.684 m; gated RMSE was 0.162–0.174 m. Translation
error refers to the offset source origin, not centroid residual.

At 0.05 m, wrong poses had smaller RMSE (0.0322–0.0363 m) but only 2.75–3.75%
support. Thus gated RMSE cannot be compared across gates without checking the
accepted populations. The broad gates recovered all seeds here, but this clean,
fully overlapping, outlier-free example does not establish a generally best gate.
Noise, partial overlap and outliers can change this tradeoff.

## Paired fixed-evaluation experiment

Run `python scripts/study_alignment_distance_gate.py --evaluation-distance 0.05
--output-dir target/fixed-gate-study` on one line. The script also writes a
standalone `report.html` with success/convergence counts and support ranges.
An omitted evaluation distance retains the original per-search-gate evaluation.

All 25 estimated transforms exactly matched their paired runs with variable
evaluation distance. Correct-pose counts therefore remain 0, 0, 1, 5, 5.
However, the four wrong poses at search gate 0.3 m have only 2–3% forward
support at fixed evaluation gate 0.05 m, versus 90.5–92.75% when evaluated at
0.3 m. The successful broad-gate runs still have 100% fixed-distance support.
This separates optimizer recovery from relaxed proximity evaluation in this
specific study; it does not make proximity support a correctness certificate.

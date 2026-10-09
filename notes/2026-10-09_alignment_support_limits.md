# Synthetic alignment diagnostic limits

Run `python scripts/study_alignment_support.py` with an installed SpatialRust
wheel and NumPy. It regenerates synthetic inputs/results under
`/workspace/SpatialRust-python-delivery/target/alignment-study/`.

Five fixed random seeds use 400 uniform target points, 240-point partial source
clouds, or 240 matching source points plus 160 distant outliers. Translation is
(0.025, -0.018, 0.012) metres; voxel leaf is 0.04 m, distance gate 0.12 m,
and each registration stage has a 60-iteration budget. A 180-vertex ring rotated
20 degrees supplies an intrinsically ambiguous case. Its five runs reuse the
same geometry and are not independent random cases.

| Case | Forward support | Reverse support | Maximum gated RMSE |
| --- | --- | --- | --- |
| Baseline | 100% | 100% | 3.28e-8 m |
| 60% shared points | 100% | 64.75–68.5% | 5.74e-8 m |
| 40% distant source outliers | 60% | 64.75–68.5% | 5.74e-8 m |
| Symmetric ring rotated 20 degrees | 100% | 100% | 4.16e-17 m |

All runs reported convergence. Random cases had maximum translation error
4.43e-8 m and rotation error 1.21e-6 degrees, measured after projecting the
float32 estimated rotation onto SO(3). The ring had 20-degree rotation error
relative to the generating pose despite perfect support and tiny RMSE:
geometrically equivalent poses cannot be distinguished from these points alone.

Support counts proximity within a gate, not physical correspondence. Reverse
support exceeds the known 60% shared-point proportion through incidental nearby
points. Gated RMSE omits rejected points, so distant outliers leave its value
nearly unchanged. Inspect both directions and their denominators; ambiguous
geometry needs additional pose priors or discriminative features. These are
synthetic observations, not real-sensor validation or a PCL/Open3D comparison.

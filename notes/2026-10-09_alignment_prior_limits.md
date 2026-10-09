# Initial pose complements local ICP, with limits

Reproduce with `python scripts/study_alignment_priors.py` using an installed
SpatialRust wheel and NumPy. Synthetic inputs and results are overwritten under
`/workspace/SpatialRust-python-delivery/target/pose-prior-study/`.

Random clouds have 180 points, five fixed seeds, a 60-degree rotation and
translation (5, -2, 0.5) metres. A separate 180-vertex ring is rotated 20 degrees;
it is one symmetric geometry, not five independent trials. Both use a 0.025 m
leaf, 0.15 m distance gate and 100 iterations per stage. Priors are identity,
the generating inverse pose, or that pose perturbed by 3 degrees and translation
(0.03, -0.02, 0.01) metres. Rotation errors use SO(3) projection of the float32
estimated matrix.

Random cases all failed without a prior because no correspondences were within
the gate. Exact and perturbed priors recovered all five, with rotation error at
most 2.10e-6 degrees and translation error at most 3.71e-7 m. A nearby prior
therefore restores the local correspondence assumption in these cases.

The ring without a prior had 100% support in both directions and RMSE 3.77e-17 m,
but 20-degree error relative to its generating pose. The exact prior recovered
that pose. The perturbed prior retained 3-degree rotation error and 14.95 mm
translation error, with 100% support, 16.77 mm gated RMSE and nonconvergence.
Thus a prior helps select a pose but does not remove geometric ambiguity or
guarantee refinement. These are synthetic observations, not claims of real-data
accuracy or superiority to another library.

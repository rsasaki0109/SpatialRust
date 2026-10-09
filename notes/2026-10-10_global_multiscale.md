# No-caller-pose global file workflow

Added FPFH/RANSAC candidate generation followed by explicit multiscale ICP.
Default seeds 7/8/9 are fixed, not selected from ground truth. Coarse source
and target use .1 m voxels, .5 m feature radius, .2 m feature-match gate and
10,000 RANSAC samples. Each coarse cloud is capped at 5,000 points to bound
quadratic descriptor matching. Optional minimum final forward support rejects
weak proximity; accepted candidates select by full-source forward count,
then RMSE, then seed order. Generated initialization has explicit provenance
in JSON/HTML, including settings and normal-orientation limitations.

Verified: 359 Python tests. Synthetic 600-point cloud with a 60-degree rotation
and translation (3,-2,.5) recovers to 5e-5 m without a supplied prior. Tests cover
failure isolation, no-hypothesis rejection before refinement, parameter checks
before IO, coarse-point budgets, selection rule, normals/scalar attributes,
immutable inputs, PCD output and malformed visualization metadata.

Public artifacts:
`/workspace/SpatialRust-python-delivery/target/public-global-multiscale-validation/`.
Same cloud_bin_0/1 inputs as earlier runs, now without caller pose. Explicit
schedule .1/.2/30, .05/.1/30, full/.02/30, final translation/rotation thresholds
1e-4 and zero fitness threshold. Seed 8 selected. Forward support 123,511/198,835
(62.1173%) and gated RMSE .006565124 m; reverse 129,116/137,833 (93.6757%) and
RMSE .006369311 m. Saved full-source XYZ exactly matches readback; field schema
is preserved. Receipt hashes inputs, pipeline, native extension and renderer.
One complete local three-seed run took 11.48 s; no controlled speed claim.

No verified real-pair pose ground truth. Partial overlap, symmetry, local
basins and default-viewpoint normal orientation can still mislead estimation.
Normals/descriptors are recomputed for each seed; this is a remaining efficiency
opportunity. Global feature fitness is not nearest-point full-source fitness.
Provisional maturity moves to 61% for the integrated, verified workflow; target
90%, with broader platforms and external real-data evidence still outstanding.

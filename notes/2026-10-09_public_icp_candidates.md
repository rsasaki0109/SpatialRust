# Public ICP sample pair: candidate workflow validation

This validates the current optimized Python candidate workflow on the first two
PCD clouds in Open3D's public, versioned `DemoICPPointClouds.zip`. It does not
execute Open3D and is not a library comparison. Dataset bytes and output clouds
remain ignored under `target/`; only the harness and this note are committed.

Fetch and run:

```bash
mkdir -p /workspace/SpatialRust-python-delivery/target/public-icp-data
curl --fail --location --max-time 45 \
 https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip \
 --output /workspace/SpatialRust-python-delivery/target/public-icp-data/DemoICPPointClouds.zip

LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/validate_public_icp_candidates.py
```

The downloaded archive SHA-256 is
`b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5`.
The harness extracts only the two named files without general archive extraction,
records their hashes and creates JSON, aligned PCD and standalone HTML under
`/workspace/SpatialRust-python-delivery/target/public-icp-candidate-validation/`.
The archive and extracted-file hashes identify the data observed, rather than
constituting an independently authenticated upstream checksum.

Source has 198,835 points and target 137,833 points. Three supplied candidates
are identity, the rounded Open3D tutorial initial guess projected onto SO(3),
and that projected pose perturbed by a target-frame five-degree Z rotation plus
(30, -20, 10) mm translation. Projection is necessary because the tutorial's
three-decimal rotation is not orthogonal enough for the rigid-input contract.
The harness retains the original rounded guess and all actual candidates.

Settings are .05 m voxel leaf, .1 m ICP correspondence distance, .02 m support
evaluation distance and five iterations per stage. Selection uses full original
source forward supported count and RMSE; it does not use a known true pose.

The receipt asserts complete selected-report equality and per-candidate
pose/support/convergence equality against independent file alignment. It also
asserts that all original source points remain present and saved/reloaded aligned
XYZ are bit-for-bit identical. Attribute roundtrip is not asserted by this harness;
separate regression tests cover attribute preservation. All 216 Python tests
passed during this validation slice.

Completed observations (CPython 3.12, NumPy 2.2.6, local SpatialRust 1.2.0 release
extension with ONNX enabled):

| Candidate | Forward supported points | Forward fraction | Forward gated RMSE | Converged |
| --- | --- | --- | --- | --- |
| Identity | 19,169 / 198,835 | 9.64% | 11.72 mm | false |
| Projected tutorial guess (selected) | 122,446 / 198,835 | 61.58% | 7.83 mm | false |
| Perturbed tutorial guess | 73,200 / 198,835 | 36.81% | 9.43 mm | false |

The selected reverse support is 128,511 / 137,833 (93.24%), RMSE 7.71 mm.
Direction changes the query denominator and sample distribution; these fractions
are not physical overlap measurements. All candidate results match independent
alignment exactly, and all 198,835 aligned XYZ points round-trip exactly through
PCD. No candidate converged within the five-iteration cap.

The optimized candidate execution took 105.40 seconds and the following three
independent file alignments took 115.33 seconds. This is a completed observation,
not a randomized repeated speed comparison. Absolute time shows a practical
remaining cost on this pair. Code inspection finds ungated nearest queries before
the correspondence-distance cutoff, and separate searches for updated fitness;
component profiling is still required to attribute this pair's elapsed time.

Single sequential optimized and independent-run wall times are recorded for
diagnosis, not as an acceleration benchmark. The real pair is larger and has
different geometry and initial-pose difficulty than the synthetic timing study.
No true source-to-target pose was supplied with this validation: high support,
small residual or convergence cannot establish pose correctness. Multiple scenes,
ground-truth error, randomized repeated timings and cross-library execution remain
necessary for broader claims.

An initial 30-iteration-per-stage run was interrupted after several minutes
without a validation receipt. It supplies no completed timing or correctness
evidence. The five-iteration cap bounds this workflow check; it is not evidence
that five iterations are sufficient for accurate registration.

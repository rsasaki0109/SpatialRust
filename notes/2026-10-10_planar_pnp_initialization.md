# Plane-aware PnP initialization and the 70% checkpoint

The controlled OpenCV study exposed a broken assumption: calibrated 3D DLT
needs enough object depth variation to constrain its projective matrix. Exact
planes add null directions; thin noisy configurations also gave poor initial
poses. RANSAC could reject incorrect image pairs but could not repair this
initializer. This slice fixes the shared initializer in both PnP APIs.

Object offsets are normalized by their extent before covariance eigendecomposition.
Smallest/largest eigenvalue at most 1e-6 chooses a homography on a fitted plane;
middle/largest at most 1e-10 rejects collinear geometry. A right-handed plane basis,
both homography signs, positive camera depth and measured reprojection residuals
choose an initial rigid pose. Gauss–Newton then uses the original 3D points,
including thin geometry's actual thickness. No generating pose enters selection.
Other geometry retains the existing DLT initialization. These fixed cutoffs are
heuristics for initializer choice, not uncertainty estimates.

The paired study uses four geometries, three noise/correspondence conditions,
five seeds and four methods (240 runs per build). Native plain recovery rises
15/60 → 28/60; native RANSAC rises 20/60 → 39/60. There are 13 and 19 gains,
respectively, and no recovery regressions. OpenCV plain remains 28/60 and robust
38/60; all OpenCV row dictionaries match exactly. This small synthetic study
does not establish superiority: contaminated thin cases still recover only 3/5
with native RANSAC versus OpenCV's 5/5. Collinear generating poses remain ambiguous.

Baseline runner and native binary snapshots were archived before rebuilding.
Their SHA-256 hashes match the original receipt. The baseline calculation
fingerprint was added from that verified archived runner, with explicit provenance;
the AST fingerprint covers experiment constants, fixture generation and solver
evaluation, excluding presentation changes. The paired validator also requires
matching input hashes, camera, robust settings, versions, threading and recovery
criteria, and rejects changed OpenCV results or contradictory recovery flags.

Artifacts (local, not committed):

- Before: `/workspace/SpatialRust-python-delivery/target/opencv-pnp-failure-study-final/`
- After: `/workspace/SpatialRust-python-delivery/target/opencv-pnp-planar-study-final/`
- Paired JSON/HTML: `/workspace/SpatialRust-python-delivery/target/pnp-planar-repair-comparison/`
- Final default wheel, JUnit and receipt: `/workspace/SpatialRust-python-delivery/target/final-maturity-wheel/`

Native fingerprints: before `318fff8ad68d0f5a3074e3a4669e17eed92d876dc14d171716856e2ed1eb2d57`,
after `7dd9ccb51c4f593c79d2d485daa9a15e843a305c499935be1bd1dc07f8a57df9`.
Calculation fingerprint: `086b652f5bdbb17ec2994f0b6c5592db475586a2109819fdf5dd41125f11b3f3`.

Validation: 421 ONNX-enabled Python tests, 142 full-feature Rust vision tests,
18 OpenCV comparison/report tests, strict extension Clippy and Python stubtest
pass. The current default release wheel is installed in a separate environment:
418 tests pass with exactly 3 expected ONNX skips. Wheel archive bytes match
installed imports, including stubs and PEP 561 marker; ICP, support and exact
PCD XYZ roundtrip also pass. Rust regressions cover arbitrary planes, four-point
planar fits, scales 1e-4/1/1e4, thin depth, outliers and collinear rejection.

The provisional engineering maturity assessment reaches 70%, building on the
merged global/multiscale workflow, external ICP comparisons, geometry diagnostics,
robust PnP and repaired wheel/source packaging. This is a subjective project
checkpoint, not a percentage of PCL/OpenCV/Open3D parity. The longer-term target
remains 90%. Real sensor ground truth, MSRV, other platform runtimes and GPU
conformance still need independent validation. Remote CI cannot be inspected
through the currently denied GitHub API; local passes are not remote CI evidence.

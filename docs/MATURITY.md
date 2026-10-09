# Maturity assessment

Current assessment (2026-10-10): **70%, provisional**. This is a subjective
engineering progress estimate, not measured feature parity with PCL, OpenCV or
Open3D, not the fraction of their functionality implemented, and not a claim
of equivalent production readiness. The long-term target remains 90%.

The 70% estimate closes a failure discovered by the external comparison:
planar/near-planar PnP now uses scale-normalized geometry detection and homography
initialization, then refines against the actual 3D points. On unchanged inputs
and controls, plain recovery improves from 15/60 to 28/60 and robust recovery
from 20/60 to 39/60, with zero recovery regressions and exactly unchanged OpenCV
rows. Archived baseline hashes and calculation fingerprints validate the pairing.
Verification passes 421 ONNX-enabled Python tests, 142 full-feature Rust vision
tests, 18 optional comparison/report tests, strict extension Clippy and stubtest.
A fresh default wheel passes 418 tests with exactly 3 expected ONNX skips;
installed native/type bytes and ICP/support/PCD checks match the wheel. Noisy
planar failures, collinear ambiguity, real ground-truth datasets and platform
coverage remain gaps. Reaching this checkpoint does not establish library parity.

The 68% estimate adds owned, validated, GIL-releasing Python robust PnP and a
240-run OpenCV 4.12.0 failure comparison. All 397 Python tests, 11 Rust geometry
tests, 2 optional comparison tests, strict extension Clippy and stubtest pass.
With 30% wrong volumetric correspondences, native plain fitting recovers 0/5
and native RANSAC 5/5. Planar/near-planar noisy cases still expose a DLT
initialization gap versus OpenCV, and collinear poses are ambiguous. This is
verified robust functionality and external failure evidence, not parity.

The 66% estimate adds geometry-conditioned failure diagnostics and visualization:
point-to-point and point-to-plane information spectra, weak directions, and
origin/scale normalization. Analytical line/plane/coincident ranks and independent
finite-difference Jacobians verify the math; 380 Python tests pass. Optional
global reports preserve poses/support while adding full-cloud spectra. Public
source/target and target-normal analysis run successfully. Local fixed-pair
information does not establish global identifiability, actual retained-pair
observability, noise covariance or confidence; those limits are explicit.

The 64% estimate adds PCL 1.15.0 to the controlled external ICP study: 702
registrations, 234 paired conditions, all three libraries recover 138 poses
with no classification disagreement. Native comparator and runner hashes,
actual PCL update counts and common full-source evaluation are recorded; 5
optional comparison tests pass. Inclusive zero-delta stopping and numerical
precision differ, and this remains synthetic local ICP rather than real-sensor
or global-registration parity.

The 63% estimate adds isolated wheel and source-distribution runtime validation
and fixes a confirmed release defect: source-rebuilt wheels omitted the type
stub and PEP 561 marker. Both normal and repaired source-rebuilt wheels pass
356 tests with exactly 3 expected ONNX skips; native/type archive bytes match
installed imports, and conversion/ICP/support/PCD checks pass. CI now repeats
the two release paths before publishing. Local evidence is Linux x86_64,
CPython 3.12, NumPy 2.2.6; remote CI and other platform runtimes remain unverified.

The 61% estimate adds a complete no-caller-pose FPFH/RANSAC → multiscale ICP
file workflow: bounded coarse descriptor matching, deterministic multi-seed
generation, failed-hypothesis isolation, optional minimum proximity support,
full-source candidate selection and visualization of initialization provenance.
All 359 Python tests pass. It recovers a large known synthetic pose without
truth entering initialization, and runs on the public cloud_bin_0/1 pair while
preserving 198,835 points, field schema and exact saved XYZ. Real-pair ground
truth remains unavailable; successful output and support do not certify accuracy.

The 59% estimate adds validated native FPFH/RANSAC configuration, finite/unit
input checks, rejection of overflowing geometry, a corrected zero RNG state,
Python reproducible seeds, bounded normal-estimation scratch, and GIL release.
Validation includes 29 CPU-registration tests, 334 Python tests, strict Clippy
and stubtest. Global hypothesis quality and the separate keypoint Python path
remain limitations; boundary correctness is not evidence of better accuracy.

The 58% assessment adds a direct controlled Open3D 0.19.0 comparison: 468 runs,
234 paired conditions, shared original-source evaluation and source/native
fingerprints, with 3 optional comparison tests passing. All paired recovery
decisions matched (138 recovered per library), exposing shared gate/outlier/
initialization tradeoffs. This establishes an external comparison method and
failure evidence, not superiority or broad real-sensor parity.

The explicit multiscale file workflow adds independently configurable stage
resolution, gates, trimming and stopping, with target-frame composition,
original-source attribute preservation and per-stage visualization. All 314
Python tests pass. A three-stage run on the public cloud_bin_0/1 pair preserves
198,835 source points and exact XYZ after PCD roundtrip. This supports the 57%
assessment; one public pair without verified ground truth does not establish
accuracy superiority, external-library parity or broad performance gains.

Earlier 2026-10-10 update at 55%: optional ICP trimming and a 702-run
paired synthetic study are now available, with both improvements and regressions
documented. Local verification is 290 Python tests and 25 CPU-registration tests,
plus strict Clippy and stubtest. This native addition alone did not establish
greater overall deployment readiness.

The subsequent file-workflow integration and controlled public-pair validation
support the current 56% estimate: both CLIs propagate trimming, record/display
settings, validate before IO, and preserve full-source output/support. Public
candidate runs match independent runs and exact XYZ roundtrips. Comparison checks
input, prior, parameter, native-extension and pipeline-source fingerprints. All
298 Python tests pass. On this one public pair, retaining 80% slightly worsens
proximity metrics, so it remains opt-in; no true-pose improvement is claimed.

The recent assessment moved from 50% to 55% after these completed and merged
capabilities, rather than changing the score merely to meet a requested target:

| Capability | Verified evidence |
| --- | --- |
| Inspectable ICP behavior | Owned optional per-update history and explicit stop reasons, sharing ordinary updates; NumPy distance checks and trace/plain equality |
| Usable convergence visualization | File and candidate CLI integration, standalone SVG and exact measurement tables, old-report compatibility and malformed-report rejection |
| Controllable stopping and safer boundaries | Separate translation/angular/absolute-MSE thresholds, explicit recorded settings, Rust/Python invalid inputs and transform overflow rejected |
| Scale-independent rigid estimation | Proper quaternion fit with f64 centroids and normalized eigensystem; scale/plane/large-origin regressions; 120 independent NumPy SVD comparisons |
| Reproducible failure assessment | 567 controlled runs over scale, noise, replacements, partial clouds and symmetry; saved update histories, source/native hashes, CI smoke tests and full-study artifacts |

Local verification for this milestone: 280 Python tests with the rebuilt
ONNX-enabled release extension, 23 registration tests with all CPU registration
features, strict registration and extension Clippy, and Python stubtest. Public
cloud_bin_0/1 optimized candidate runs match independent runs and full aligned
XYZ roundtrips exactly. There is no independently verified real-pair ground truth.
Remote CI cannot currently be inspected because GitHub API access is denied;
configured CI steps and successful local checks do not establish remote success.

Important remaining gaps toward 90%:

- Direct, controlled PCL/Open3D/OpenCV comparisons on multiple public datasets
  with verified reference poses, failure regimes, memory and throughput evidence.
- Stronger treatment of outliers, correspondence ambiguity, degenerate geometry
  and global initialization. Stopping thresholds do not solve these problems.
- Broader confirmed platform, release, MSRV and GPU/backend conformance. Existing
  workflows and features are not evidence that every deployment works.
- Continued API compatibility, packaging, documentation and operational validation
  as features evolve. f32 XYZ precision still limits large-origin sensor geometry.

See the dated notes for limits of individual experiments. Reported geometric
support is distance-gated proximity, not measured physical overlap. A met
convergence criterion does not establish the correct pose.

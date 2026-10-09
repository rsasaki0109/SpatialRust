# Maturity assessment

Current assessment (2026-10-09): **55%, provisional**. This is a subjective
engineering progress estimate, not measured feature parity with PCL, OpenCV or
Open3D, not the fraction of their functionality implemented, and not a claim
of equivalent production readiness. The long-term target remains 90%.

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

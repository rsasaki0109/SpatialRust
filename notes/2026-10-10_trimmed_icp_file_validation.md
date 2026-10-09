# Trimming in file workflows and controlled public-pair comparison

Both alignment examples and the public-pair validation harness accept
--trim-fraction. The keyword trim_fraction is propagated through file/cloud
alignment and optimized candidate search into both coarse and fine ICP stages.
Settings are validated before reading files. Non-default values appear per stage
in JSON and standalone HTML, including reports without history. Default 1.0
preserves existing ordinary report fields and alignment output. Convergence
criteria serialization remains separate from correspondence selection options.

Tests cover validation-before-IO, default equivalence, both-stage propagation,
CLI execution and HTML, selected-candidate versus independent reports, escaped
stage names and invalid serialized fractions. All 298 Python tests pass with the
previously rebuilt release ONNX-enabled native extension; this slice changes
examples/scripts only and requires no native rebuild.

Public pair: Open3D DemoICPPointClouds cloud_bin_0/1, 198835 source / 137833 target
points; three supplied prior poses; leaf .05 m, coarse gate .1 m, fine/evaluation
gate .02 m, cap30, translation/rotation thresholds 1e-4 m/rad, absolute MSE
stopping disabled. All-pair and 80%-retention runs share the same native extension
SHA256, input hashes, poses, pipeline-source hashes and all other settings.
Both candidate searches agree with independent runs, and full aligned source XYZ
roundtrips exactly. Source fingerprints were verified after measurement.

| Selected candidate metrics | All pairs | Retain 80% |
| --- | --- | --- |
| Candidate index | 1 | 1 |
| Forward support within .02 m | 123513 / 198835 (62.1183%) | 123167 / 198835 (61.9443%) |
| Reverse support within .02 m | 129117 / 137833 (93.6764%) | 128664 / 137833 (93.3477%) |
| Forward gated RMSE | 6.5657 mm | 6.8244 mm |
| Coarse stop | limit at 30 | limit at 30 |
| Fine stop | transform threshold at 24 | limit at 30 |

The observed proximity metrics are slightly worse with trimming for this selected
candidate. This does not establish true-pose error: no independently verified
ground truth is available. It reinforces the synthetic counterexamples and keeping
trimming optional. Times in the report are sequential individual observations,
not repeated comparative performance benchmarks.

The offline comparison tool accepts --vary trim-fraction, normalizes omitted
fractions to 1.0, and rejects different inputs, priors, fixed parameters, native
extension fingerprints or pipeline-source hashes. Trimming comparisons require
a native fingerprint on every receipt. Tests exercise these rejection paths.
It still supports older iteration-cap/fine-distance reports.

Artifacts:
/workspace/SpatialRust-python-delivery/target/public-trim-workflow-all/receipt.json
/workspace/SpatialRust-python-delivery/target/public-trim-workflow-80/receipt.json
/workspace/SpatialRust-python-delivery/target/public-trim-workflow-comparison/report.html

Subjective maturity: 56%, provisional, based on usable file integration and
controlled validation, not improved real-pose accuracy or external-library parity.
Remote CI remains unconfirmed because GitHub API access is denied by policy.

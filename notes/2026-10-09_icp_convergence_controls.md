# Unit-aware ICP stopping controls and native boundaries

Added Rust IcpConvergenceCriteria with independent translation, rotation and
absolute MSE thresholds, exposed as keyword-only options on both Python ICP
functions and CLI flags in both alignment examples. Existing Rust IcpConfig
literal fields, Python positional arguments and default thresholds are preserved.
Zero disables a test; transform stopping requires both thresholds. Explicit
criteria are recorded in each stage and checked against the recorded final update
by the standalone trace renderer. The examples validate options before file IO.

Rust ICP now rejects invalid gates (including squared underflow/overflow), zero
iteration budgets, fewer than three minimum correspondences, nonfinite thresholds,
nonfinite/short XYZ clouds, invalid initial quaternions, invalid estimated poses,
and transformed-coordinate overflow. Errors replace meaningless numerical output.

Validation: 19 registration tests with all CPU registration features; 254 Python
tests with the rebuilt ONNX-enabled release extension; strict registration and
extension Clippy; Python stubtest. Tests cover disabled and explicit-default
thresholds, each invalid threshold, rigid-transform and overflow errors, CLI
propagation/serialization, and renderer stop-reason consistency.

Public pair rerun at coarse .1 m, fine/evaluation .02 m and cap30 uses explicitly
chosen translation/rotation thresholds 1e-4 m/rad and disables absolute fitness.
This is a stopping-policy comparison, not an accuracy improvement. Independent
candidate runs agree and saved XYZ roundtrips exactly. The selected fine stage
can now stop when both update thresholds are met. Support changes slightly because
earlier stopping changes the estimated pose. No ground truth is available.

Artifacts:
/workspace/SpatialRust-python-delivery/target/public-icp-stopping-controls/report.html
/workspace/SpatialRust-python-delivery/target/public-icp-stopping-controls/receipt.json

One pair and chosen tolerances do not establish universal defaults or superiority
to Open3D/PCL/OpenCV. Remote CI remains unavailable through the GitHub API.
Subjective maturity: 53%, provisional. Numerical estimation across coordinate
scales and a reproducible multi-condition validation remain toward the 55% goal.

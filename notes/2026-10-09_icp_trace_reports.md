# Usable iteration reports and public-data observation

Both alignment examples accept --trace. Each stage gains owned icp_history and
stop_reason fields without changing ordinary reports. Candidate selection retains
the selected candidate's full history. HTML uses self-contained SVG with labeled
linear axes and an exact-measurement table, showing estimator/rematched counts,
gated RMSE, translation update and shortest rotation update. Optional history is
validated for sequence, counts, finite ranges, fitness differences, final fitness,
and convergence/stop-reason consistency. Older reports remain readable.

The rebuilt extension passed the full 241-test suite. An existing voxel GIL
observer test failed once to observe thread progress and passed on the full
rerun; no voxel production code changed. A subsequently added selected-candidate
trace test passed with the candidate suite. Trace/plain XYZ and reports agree
after removing optional history fields.

Public Open3D cloud_bin_0/1 rerun (three supplied poses; leaf .05 m, coarse gate
.1 m, fine/evaluation gates .02 m, 30 iterations) exactly matches the preceding
ordinary report after excluding trace fields and file paths. Independent runs
agree and saved XYZ roundtrips exactly. Native build: opt-in diagnostics slice.

Selected candidate final update: 123506 estimator and 123507 rematched points,
translation 3.779987284819865e-5 m, rotation 1.285712432675122e-5 rad,
fitness 4.309507287107185e-5 m². Both update magnitudes exceed the default 1e-8
transform threshold, and fitness exceeds the absolute 1e-6 m² threshold.
Consequently the explicit reason is iteration_limit. Final five fitness changes
include negative values as membership changes; a fixed-set monotonic-error claim
would be inappropriate. This observation motivates separate translation/rotation
thresholds, not a claim that the final pose is correct.

Artifacts:
/workspace/SpatialRust-python-delivery/target/public-icp-trace-30/report.html
/workspace/SpatialRust-python-delivery/target/public-icp-trace-30/receipt.json

Limitations: one public pair, supplied initial poses, no independently verified
ground truth, no comparison to executing Open3D/PCL/OpenCV. Remote CI is unavailable.
Subjective maturity: 52%, provisional; 55% still requires numerical safeguards
and validated control of termination.

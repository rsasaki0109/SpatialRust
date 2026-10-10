# SR2 real-sensor continuation

The user authorized continuing toward 90%. The latest concrete evidence is
official RealSense L515 data, not a renamed synthetic benchmark. Fixed plan
`4ce5a76` precedes full processing; native projection and attribute IO runner
`581bd90` is committed before final explicit-CPU execution, following backend
control `35af742`. All 890 depth frames and 264,115,152
points pass an independent geometry audit. Eighteen exact typed IO roundtrips
and nine fixed MVP executions pass. Source hashes, calibration, raw time domains,
recorded/unapplied transforms and all per-frame outcomes are retained.

See `/workspace/SpatialRust/docs/REALSENSE_OPERATION_VALIDATION.md` and the
committed compact receipt. Raw inputs and runs are kept below
`/workspace/SpatialRust/target/real-sensor-reference`, never staged for Git.
The full default-wheel suite passes 647 tests with exactly three ONNX skips;
12 CI helper tests pass. All four stage backends are CPU with zero transfers.
Intermediate run-v2's nine incorrect backend-name assertion failures remain
recorded; the corrected final run exactly reproduces initial geometry, IO and
pipeline metrics. There are no Rust/native API or dependency changes.

A separate hash-frozen timestamped trajectory evaluator is implemented and
tested on analytic cases. It preserves integer nanoseconds, failures, unmatched
coverage and independent ATE/RPE counts, with no scale fit. It has not yet scored
a real TUM sequence. The current official host returns proxy CONNECT 403; new
host requirements are saved through the supported environment configuration
workflow, with prior hosts retained. No proxy/TLS bypass is used.

PR #115's exact source remains `816f76c41b7b557b54e97937dfca05c958c1dd4c`.
The latest complete job snapshot at this checkpoint has 82 successes, three
running and 18 queued main jobs out of 103. Its seven wheel build/runtime jobs
pass; publication is intentionally skipped. That is incomplete main CI and is
not a merge gate success. New changes are based on #115 and must be delivered
with their own exact-head complete CI before merge. Prior #112–#114 remain
merged with complete 103-job evidence.

Assessment: **86% provisional**, supported by the new limited real-data
ingestion/geometry/IO operation. No TUM ATE, independently measured clock
synchronization, color alignment, original LiDAR mounting calibration or broad
90% production parity is inferred from these results. The old canonical bag
remains blocked without its exact-source measured clock and mounting artifacts.
Next: apply the saved TUM hostname additions, verify publisher data/calibration
and timestamps, preregister complete-sequence generation before using truth,
freeze all estimates, then evaluate coverage, ATE and RPE.

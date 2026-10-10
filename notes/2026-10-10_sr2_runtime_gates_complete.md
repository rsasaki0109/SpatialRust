# Complete SR2 hosted runtime gate

The latest tested heads for PRs #112, #113 and #114 all have complete successful
103-job main CI matrices. Each wheel run has seven successful build/runtime jobs
and one intentionally skipped `Publish to PyPI` job. Windows UTF-8 runtime
validation, macOS HTTP streaming, Linux Python 3.8/3.14, Web/WASM browser output,
feature matrices and integration checks are observed, not inferred from configured
workflow entries or from a first page of jobs.

| PR | Tested head | Main run / attempt | Main results | Wheel run |
| --- | --- | --- | --- | --- |
| #112 | `ce1fcd8a5e4da126423bb186ebcbafae6c9411ad` | 38011747105 / 1 | 103 success | 38011747222 |
| #113 | `52e68d5b7405b4485e7c4db2ff3c0ff607799eaa` | 38015125009 / 1 | 103 success | 38015125015 |
| #114 | `2a6b511a528184540c6e687b86d829994dc68b12` | 38018628323 / 2 | 103 success | 38018628325 |

The first #114 run had 102 successes and a failed `Test (spatialrust-io-copc)`
job. That job stopped in `dtolnay/rust-toolchain@stable`; cache and build/test
steps were skipped. The failure's root cause is not established: the detailed
log destination `results-receiver.actions.githubusercontent.com` was denied,
and API annotations contain only exit code 1. A targeted failed-job rerun
succeeds through toolchain setup and the actual build/test step. Unaffected
passed jobs are not described as having been reexecuted.

The default/latest job-list route returned repeated HTTP 502 after rerun while
the authoritative `/actions/runs/38018628323/attempts/2/jobs` route worked.
`inspect_github_ci.py` now requests the observed run's explicit attempt, with
the existing complete pagination and exact run/head/attempt checks preserved.
Three regression tests verify endpoint selection, rejection of stale jobs, and
rejection of invalid attempts before any request. All 12 CI-helper tests pass.
Actual API verification fetches all 103 jobs across two pages, checks matching
head and attempt, and rechecks the run identity after fetching. The final receipt
is `/workspace/SpatialRust/target/goal-continuation/pr114-ci-attempt-scoped.json`;
the original failure is retained in `pr114-ci-current.json`. Prior full receipts
are `pr112-ci-current.json` and `pr113-ci-current.json` in the same directory.
Official API routes are used with inherited proxy and TLS trust unchanged.

## Delivery

The merge sequence preserves the stacked commits and their source identities:

- #112: merge `1b6787aec42677edc86b7480a790ccd1426c8ac7`.
- #113: retargeted to main; merge `e0b2eefe557e2c09fdedef9f85098ee371680278`.
- #114: retargeted to main; merge `9c8662b4b7832547e0bfe265e16a5e813fc2517a`.

No branch is force-pushed or deleted. All merges match the reviewed/tested head
explicitly. `git merge-tree` confirms the proposed #113 and #114 merge trees
are exactly their tested head trees. Historical dataset receipts and frozen
registration helpers are unchanged.

## Remaining calibration gate and assessment

All four existing canonical calibration-manifest entries pass exact byte-size
and SHA-256 checks, including the 713,670,656-byte DB3. Source identity matches
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`, but
`registration_ready:false` remains correct: source-bound measured clock and
root-to-`lidar_front`/`lidar_rear` artifacts are absent. Cached synthetic test
fixtures, a different capture's TF, or assumed identity transforms do not satisfy
this gate. The separate synthetic Redwood study likewise cannot certify these
sensor clocks or mountings.

A fresh execution of `rosbag2_calibration_evidence` against those exact bytes
exits with the documented blocked code 2, with `identity_matches:true`, empty
frame edges and `registration_ready:false`. Its new JSON/HTML/four-file manifest
is preserved at
`/workspace/SpatialRust/target/goal-continuation/calibration-evidence-current`.
This confirms that the remaining gate is missing source-bound evidence, rather
than a lost input or a failed input hash check.

The provisional engineering assessment is **85%**, up from 82% on the new
complete hosted runtime/integration evidence together with the already frozen
separate synthetic benchmark. This is not measured PCL/OpenCV/Open3D feature
parity. Source-bound measured calibration and broader real-sensor full-operation
validation remain necessary for 90%; no arbitrary score increase closes them.

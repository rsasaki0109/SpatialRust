# COPC fixture accepted-socket regression

The resumed PR #126 head `395b12298e8adb88a09d7d5fd5f67cc4f913b023`
finishes 106 of 107 main CI jobs successfully. The remaining job,
`114401293124` in run `38116139690`, fails on macOS. All seven required
wheel jobs succeed; PyPI publication is intentionally skipped. This is a
failed integration gate, not an acceptable complete matrix.

A normal proxy-backed, verified-TLS request now retrieves the 53,029-byte
job log from `productionresultssa4.blob.core.windows.net`, despite the
runtime metadata still omitting that destination. Actual operation succeeds;
no proxy, TLS or network-policy bypass is used. The optional run-log archive
redirects to another denied destination and is not needed for this diagnosis.

The detailed log identifies the two newly added fragmented HEAD/Range tests.
Each reads EOF (`response before complete request headers: 0`) before sending
the remaining headers. Records and bounded IO tests succeed before this
integration fixture failure; later workflow commands are not executed.

Darwin can inherit `O_NONBLOCK` from the nonblocking listener onto accepted
sockets. A read timeout does not restore blocking semantics. The synchronous
fixture worker could therefore receive `WouldBlock` between fragments and
close the connection. The already separate unit-test server explicitly sets
blocking mode; this integration fixture had omitted that step.

Two added real TCP regressions force accepted sockets into nonblocking mode
on every platform and preserve the existing strict HEAD/no-body and exact
206/range-payload assertions. **Both fail before the repair on Linux with the
same EOF diagnostic as the hosted macOS failures.** The worker now explicitly
sets blocking mode before its existing five-second timeout and bounded 4 KiB
header loop. After repair, the complete streaming IO feature profile passes
76 tests, including all six HTTP integration cases, with no failures or
ignores. `cargo fmt --all --check` also passes. Linux reproduction and repair
do not substitute for verification on a new hosted macOS head.

Local, hash-bound receipts and logs are retained under
`/workspace/SpatialRust/target/goal-continuation/`:

- `pr126-oct12-resume-ci.json`
- `pr126-oct12-macos-job.log`
- `pr126-oct12-log-access.json`
- `pr126-oct12-nonblocking-before-fix.log`
- `pr126-oct12-blocking-after-fix.log`
- `pr126-oct12-blocking-regression.json`

This repair changes only test infrastructure. Production COPC readers,
trajectory generation, frozen TUM estimates and scoring thresholds are
unchanged. All 12 real study/comparison cases still fail at least one fixed
coverage/accuracy condition, and source-bound canonical Autoware clock and
mounting calibration remains unavailable. The maturity estimate remains
**87%, provisional**; 90% is unsupported. The repaired, integrated head must
pass all 107 main and seven required wheel jobs and match a fresh candidate
merge tree before PR #126 can merge.

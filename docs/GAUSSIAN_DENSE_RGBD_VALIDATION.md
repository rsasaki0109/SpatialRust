# Explicit Gaussian3 dense RGB-D fitting policy

Operational slice 146L. Commit this contract and both full-scene plans before
new fitting. The parent native causal control completes xyz/rpy but rpy ATE
0.114793m misses 0.10m. The external Open3D CPU control completes both and passes,
with rpy ATE 0.099684m/RPE 0.048494m. That comparison contains several model
differences; it establishes no single causal preprocessing explanation.

## Fixed model and ownership

Add an explicit typed native preprocessing policy and a Python keyword whose
default remains raw. Original frame/options/estimate types and raw behavior
retain their contracts. The Gaussian3 choice changes only full-resolution fit
inputs: outer product of [1,2,1], divided by 16, replicated edge coordinates,
float64 gray/depth output. A missing/out-of-range/nonfinite depth tap invalidates
the filtered depth pixel; there is no hole fill or depth-weight renormalization.
Keep gray normalization at raw u8/255, with no pairwise mean normalization.
Apply the existing pyramid, bilinear derivatives, 20/10/5 updates, depth weight,
stopping rule and intermediate solvability/count policy unchanged.

Retain owned raw full-resolution levels for the final pose statistics and the
unchanged >=30 / >=0.5 correspondence gate. Charge the additional four float64
raw planes (32*width*height bytes) in checked preflight before any fit-buffer
allocation, under the same 128MiB logical bound. Raw mode retains its original
bound. Python owns four input arrays before releasing the GIL; all transfers
remain explicit CPU ownership. No core/heavy dependency, GPU or unsafe change.

Use identity then the exact prior accepted source-to-target increment, retaining
one 128-byte float64 seed per chain. Preserve all original full-cloud support,
world composition, timestamp and typed-IO checks. Any loss preserves every
remaining row and makes no further estimator call. Record the selected fitting
policy; raw diagnostics stay separate from its filtered fitting objective.

## Complete prospective experiment

Reuse original 798 xyz and 722 rpy frame plans from the causal preparation through
verified hard links. Preserve native/external controls, failed preparations,
archives, metadata, binaries and all wheels. Canonicalize only the declared
parent identity to its exact LF byte form; require uniform LF/CRLF source and
reject all other plan/model changes. Existing scientific plans are never edited.

Freeze both new full timelines before reading either new reference score.
Keep 100% generation, >=95% reference coverage, first-pose SE3 ATE <=0.10m and
1s translation RPE <=0.05m. Retain all three/six publisher reference gaps and
unmatched RPE denominators. Independent auditing recomputes original raw
projective statistics, seed/world lineage and trajectory scores at existing
tolerances. A failed full-scene outcome remains failed; no post-score tuning,
selection or threshold change.

## Storage and delivery

New build/package/test root: `E:/SpatialRustBuild/native-dense-gaussian3-v1/`.
Reuse `E:/SpatialRustBuild/maturity-80-target/` as compilation cache, while hashing
and preserving every frozen binary/wheel before and after. Cap new package/test
output plus incremental cache growth at 2GiB; retain at least 20GiB free. Use
four build/CPU workers, owned commands bounded at one hour, and no package/data
download. New data roots use `gaussian3-dense-v1`; study root is
`E:/SpatialRustData/maturity-80/gaussian3-dense-study-v1/`, with combined new
scientific output capped at 1GiB. Share existing PNG/index payloads; never copy
them on link failure. Keep Python/native package checks independent of fitting.

Only exact owned passing pytest scratch may be removed after passing XML and
complete absolute-path/no-reparse checks. Preserve all scientific failures and
measurements. After implementation, tests and source/plan/binary freezing, run
both full scenes, audit, commit outcomes, PR and merge. Native rpy/held-out,
canonical measured clock/physical frame accuracy, both raw Linux hardware
classes and full mapping/semantic/release acceptance remain explicit 80% gates.

## Pre-fit implementation validation

Model/plan commit `8414866` precedes implementation and fitting. Numerical source
`ef77631` adds the typed `DenseRgbdPreprocessing` and checked working-byte policy;
Python's `preprocessing` keyword remains `raw` by default. Gaussian3 row tiles
share the existing private four-worker pool and are serial on WASM.

Native dense tests pass 13/13, including impulse/border weights, missing taps,
raw final denominators, exact added-plane charging and serial/parallel bit
equality with partial tiles. Native and Python-bridge Clippy pass. Focused Python
tests pass 49/49; the full installed-wheel suite passes 1,043 with three existing
optional ONNX skips. The new 25 cases independently recompute raw statistics,
check ownership/strides/concurrent calls, fixed controls, causal state and
generator/auditor loss behavior. Installed wheel bytes, ICP/support and PCD
round-trip checks pass.

Five identical synthetic raw-default cases were run separately through the
preserved old module and the new one: identity, one-pixel and fractional-pixel
motion, a depth hole and an explicit seed. Serialized poses and original
diagnostics agree exactly, excluding only the new policy identifier. New module
SHA-256 is `46f4434b0b6456a3ea564d2a7f1f7c4c942ce4979348d6a06e9eed36ef9611a6`.
All old frozen modules/wheels/executables retain their hashes.

The first native-test launcher failed before creating a process because Windows
lookup did not use the child PATH; its empty log is preserved and Cargo now uses
an absolute path. One Clippy observer encountered an atomically removed compiler
temporary file; after confirming no matching Cargo process remained, the bounded
verification was repeated successfully with disappearing-entry-aware inventory.
No fit/model/threshold was altered. Passing scratch alone reclaimed 397,598,601
logical bytes; keep all XML/logs and both startup/observer records. These checks
precede complete-scene estimation and do not establish trajectory acceptance.

## Complete planned experiment outcomes (2026-10-20 JST)

Workflow source `e311d87` freezes both timelines at
`2026-10-19T21:12:54.861663+00:00` before either new reference score. The
[source, storage and measurement receipt](receipts/2026-10-20_gaussian3_dense_rgbd.json)
binds the separately frozen numerical package, all plans/inputs, original raw
controls, every estimate/error, both independent audits and the failed rpy chain.

| Implementation / scene | Generated | First-pose ATE RMSE (m) | 1s RPE RMSE (m) | Existing gates |
| --- | ---: | ---: | ---: | --- |
| Native raw causal / xyz | 798/798 | 0.089993 | 0.026591 | Pass |
| External CPU / xyz | 798/798 | 0.051174 | 0.023110 | Pass |
| Native Gaussian3 / xyz | 798/798 | 0.082755 | 0.025186 | Pass |
| Native raw causal / rpy | 722/722 | 0.114793 | 0.048145 | ATE fails |
| External CPU / rpy | 722/722 | 0.099684 | 0.048494 | Pass |
| Native Gaussian3 / rpy | 349/722 | 0.119374 (prefix) | 0.049937 (prefix) | Incomplete; fails |

XYZ independently evaluates 795/798 reference poses and 762/768 one-second pairs.
RPY retains all 722 planned rows: 349 poses and 373 errors. Its reported prefix
errors cover only those generated poses and 319/692 planned one-second pairs;
they do not establish full-trajectory accuracy. All three/six publisher reference
gaps stay explicit in the full denominators.

RPY first fails at zero-based index 349. Returned-pose raw support is
120,612/242,491 = 0.49738753190840074, below the unchanged 0.5 gate. No pose is
admitted at that frame, and no later estimator call/reset occurs. Do not waive
this as rounding or lower the threshold. The policy improves xyz against the
raw control but does not resolve rpy. It stays an explicit experimental choice;
the raw default remains unchanged. A future separately fixed model can isolate
gray filtering from depth filtering; these outcomes identify no single cause.

All 797+348 admitted causal/projective pairs pass independent original-raw
recomputation. Counts/fractions and preceding seeds agree exactly; maximum
gray/depth RMSE differences are 2.09e-16 / 1.05e-17, world-composition difference
4.52e-14, and independent trajectory score differences 1.11e-15m / 2.56e-13
degrees. All eight generation/audit/score commands complete successfully,
including scientific recording of the rejected rpy chain. Numerical receipts
report four workers, 22,743,840 charged working bytes and 3,072,000 owned input
copy bytes on every admitted pair; these are logical native payloads, not RSS.

Final scientific output plus new preparation metadata is 229,398,835 logical
bytes under the fixed 1GiB budget. Existing PNG/index payloads share 918,149,680
logical bytes through 3,045 checked links; new image copies/downloads are zero.
The final new build root plus incremental reused-cache growth is 655,029,109
logical bytes, with maximum successful-command sample 1,048,366,695, under 2GiB.
Keep the original observer/startup failures and their logs; sampling is not a
continuous peak measurement. All old/new frozen native files, six wheels and
271 old JSON files remain unchanged. Passing scratch alone reclaimed
397,598,601 bytes. Rpy/held-out, canonical calibration, Linux raw classes and full
mapping/semantic/release acceptance remain open; this does not prove 80%.

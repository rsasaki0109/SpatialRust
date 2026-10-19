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

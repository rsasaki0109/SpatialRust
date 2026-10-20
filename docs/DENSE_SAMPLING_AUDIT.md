# Dense RGB-D sampling and derivative audit

Operational slice 146O. The pair-mean development experiment completes both
timelines but rpy ATE 0.130595m still fails the original 0.10m gate. Preserve
every previous variant and result. This slice tests operator semantics before
another estimator model is proposed; it does not shrink the 80% target.

## Fixed comparison

Commit `benchmarks/dense-sampling-audit-plan.json` before collecting numerical
probes. Bind the seven official Open3D v0.19.0 source files to upstream commit
`1e7b17438687a0b0c1e5a7187321ac7044afe275` and the existing CPU module's hash.
Use the unchanged native production functions through a test-only Rust probe,
and the installed CPU Image pyramid/filter and hybrid-row APIs. No new public
API, runtime dependency, default, estimator or device transfer is introduced.

Use an 8x8 affine gray image, constant metric depth, a 7x5 odd-size image,
explicit missing-depth neighbors and fixed border/subpixel coordinates.
Independently calculated expectations distinguish centered Gaussian stride-two
sampling from Gaussian-then-2x2 averaging, ceil from floor dimensions and
four-corner admission from a single integer-depth lookup. These are operator
tests on identical supplied bases, not a complete estimator equivalence test.

Validate all twelve native residual-Jacobian entries against an independent
analytic rigid-action calculation with common supplied point/gradients.
Observe the external hybrid row's residual/weight vectors. If its Python API
does not return mutated Jacobian arrays, record that limitation explicitly;
unchanged caller input arrays cannot prove a C++ Jacobian result. Keep source
arithmetic and float32/float64 operation-order differences separate.

The source-defined rounded coordinate lookup is reported as a source model,
not as an exported runtime correspondence set. No observed difference is
declared the cause of rpy error without further evidence.

## Resource and provenance gates

Build/test root: `E:/SpatialRustBuild/dense-sampling-audit-v1/`; reuse the
existing offline `E:/SpatialRustBuild/maturity-80-target/` cache. Bound new
build/test bytes plus incremental cache growth to 1GiB, owned commands to
20 minutes and worker pools to four. Keep at least 20GiB free. Freeze the
test executable in its own build directory before collecting its probe.
Reuse the installed wheel and reference CPU module; do not rebuild or replace
any previous wheel, executable or library. Prove that the Rust production
prefix and Python production sources are unchanged.

Result root: `E:/SpatialRustData/maturity-80/dense-sampling-audit-v1/`;
cap audit output at 16MiB. Download/copy no sensor payload. Preserve all old
JSON files, scientific outputs, failures and frozen binaries. Only explicitly
named passing pytest scratch may be removed after XML, absolute-path and
full no-reparse verification. Record inputs, commands, source/executable/module
hashes, observed values, tolerances and available disk bytes.

Full native rpy accuracy, held-out recording robustness, measured canonical
clock/physical frame calibration, both raw Linux hardware classes and full
mapping/semantic/release acceptance remain unproven. Operator agreement and
green CI cannot establish operational 80%. Deliver the audit, tests, bounded
evidence, PR and merge as this single named family.

## Observed outcome

The protocol was committed as `2bfd55d` before probing. The test executable
was built from `04fc306`; the successful Python auditor is `9b147bc`. The
production Rust prefix, Python bridge/stub and core/GPU trees retain the
`0573432` baseline. No production estimator or default changed.

| Fixed operator | Native observation | Pinned CPU reference observation |
| --- | --- | --- |
| Interior level-one gray | 0.09019607843137256 (23/255) | 0.10588235408067703 (27/255 within float32 tolerance) |
| 7x5 source, level-one dimensions | 4x3 | 3x2 |
| Missing depth at the coarse sample | Weighted finite depth remains 2m | Box-averaged depth is nonfinite |
| Integer sample beside a missing neighbor | Four-corner sampling rejects | Source-modeled integer lookup has finite selected depth |
| Last-column sample | Four-corner sampling rejects | Source-modeled integer lookup has finite selected depth |
| Affine gray gradient | (3/255, 5/255) | Sobel3 gradient agrees within 1.17e-9 |

The native bilinear value at (2.25, 2.5) is 26.25/255 with the expected exact
affine derivatives. The reference source rule rounds that coordinate to
(2, 3), whose gray is 28/255 within float32 tolerance. The lookup rows describe
the pinned source arithmetic and actual image values; they do not observe a
runtime correspondence set or prove full correspondence admission.

All twelve native residual-Jacobian entries match the independent analytic
projection/rigid-action calculation for the common supplied point and gradients
(maximum error zero). The actual reference hybrid-row API returns residuals
0.0017888526760153527 and 0.009838689718084435, with weights [1, 1], matching
the fixed expectations. Its Python list does not expose the C++ Jacobian
vector mutation: the receipt records `jacobian_observable=false` and a null
runtime Jacobian. No numerical reference-Jacobian agreement is claimed.

The first external adapter call failed because the Python enum spells the
filter `Sobel3dx`, whereas C++ uses `Sobel3Dx`. Its log is preserved. Only the
Python enum spelling changed before the successful second call; the protocol,
test executable and tolerances stayed fixed.

Native tests: 21 passed. Python: 31 new audit cases passed; the full suite
passed 1,134 cases with three existing optional ONNX skips. Clippy, Rustfmt
and Python 3.8 syntax checks passed. The reused native wheel was not rebuilt.
The [measurement receipt](receipts/2026-10-20_dense_sampling_audit.json) binds
source, executable, module, command, log and XML hashes, including the failure.
Actual audit results remain at
`E:/SpatialRustData/maturity-80/dense-sampling-audit-v1/observations-v2/audit.json`.

New result bytes total 116,313; retained new build plus incremental cache growth
is 5,579,325 bytes. Maximum successful-command sampled growth was 430,144,092
bytes, below the fixed 1GiB bound. Verified passing pytest scratch cleanup
reclaimed 438,281,388 logical bytes. No sensor payload was downloaded or copied;
all prior JSON outputs and 32 frozen native artifacts remain hash-verified.
After cleanup, E had 676,487,598,080 free bytes, above the 20GiB floor.

This establishes the tested operator differences, not their causal contribution
to rpy error. Native trajectory acceptance and operational 80% remain false.
The next estimator experiment needs a separately fixed, coherent raster,
gradient and pyramid model before fitting; all original full-timeline,
raw-final, world-chain and accuracy gates must remain.

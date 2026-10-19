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

# Isolated wheel and source distribution verification

Local artifacts: `/workspace/SpatialRust-python-delivery/target/runtime-wheel-validation/`.
Built the default release wheel with frozen Cargo lock and cached dependencies,
created a fresh virtualenv, installed the wheel and dependency archives offline,
and ran 356 tests successfully. Exactly 3 ONNX tests skip because the default
wheel has no ONNX feature. No local ONNX library path is needed for this wheel.

The verifier compares native module, package initialization, type stub and
PEP 561 marker against actual wheel bytes, rejects editable/mismatched installs,
checks imported paths, and exercises NumPy conversion, ICP, indexed support and
exact saved XYZ. JSON records wheel/native hashes, runtime versions, platform
and JUnit counts. No zero-test/disabled-suite readiness claim.

Source-distribution rebuild exposed a real packaging defect: its pyproject is
relocated to the bundle root, but spatialrust.pyi only remained in the crate
subdirectory. Runtime tests passed yet the rebuilt wheel omitted __init__.pyi
and py.typed. An explicit sdist root include fixes discovery. The repaired
source distribution builds offline with --locked; its installed wheel again
passes 356 tests with 3 expected skips and all archive-byte/runtime checks.

CI adds isolated x86_64 wheel runtime tests, source-rebuild validation and
receipts/artifacts to the existing wheel/sdist jobs. Publishing already depends
on both jobs, so failures block it. Workflow YAML parsed locally; remote CI
cannot be observed through the denied GitHub API, and no release was published.

Tested CPython 3.12.14, NumPy 2.2.6, PyArrow 26.0.0, Linux x86_64/glibc 2.41.
Local wheels tag manylinux_2_39: they are not evidence for older glibc. Production
CI uses its manylinux builder; aarch64 is cross-built, not locally runtime-tested.
Python 3.8 ABI compatibility, MSRV and GPU conformance remain separate checks.
Provisional maturity 63%, long-term target 90%.

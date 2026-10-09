# Opt-in ICP iteration diagnostics

Rust IcpRegistration::align_with_trace and Python register_icp_diagnostics share
the ordinary ICP update path. Ordinary alignment does not allocate history.
The owned result includes one row per completed update and an explicit stop
reason (absolute fitness, transform update, or iteration limit).

Rows include before-update estimator correspondence count, after-update rematched
count, mean squared distance, previous-minus-current fitness, translation length,
and shortest rotation angle. Fitness changes compare potentially different point
memberships and do not establish physical overlap or pose correctness. No
additional nearest-neighbor pass is added for diagnostics. Fitness accumulation
now directly sums the same f32 differences promoted to f64 in the same order,
instead of allocating two temporary correspondence vectors.

Rust tests cover all three stop reasons and exact ordinary/trace result equality,
including a centered pure rotation. Python checks post-update distances against
an independent NumPy all-pairs calculation, exact transform equality, input
validation, immutable owned history, lifetime independence, and concurrent reads.
All 228 Python tests passed with the rebuilt release ONNX-enabled extension.
Registration and extension strict Clippy and Python stubtest passed.

Files:
/workspace/SpatialRust-python-delivery/crates/spatialrust-registration/src/icp.rs
/workspace/SpatialRust-python-delivery/crates/spatialrust-py/tests/test_icp_diagnostics.py

Remote CI cannot be inspected because the GitHub API is denied by network policy.
This API is a diagnostic capability, not a new global registration algorithm.
Subjective maturity: 51%, provisional. The 55% goal still requires integration
into usable reports, numerical safeguards, and broader failure validation.

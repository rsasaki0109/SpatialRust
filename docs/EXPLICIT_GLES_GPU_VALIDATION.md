# Explicit GLES runtime and public GPU validation

Operational slice 146I. A capability-only probe of the existing Ubuntu 22.04
WSL2 system driver creates an OpenGLES 3.1 context through D3D12 and reports
1,024 compute invocations, 32KiB shared storage and 16 compute storage blocks.
The separate OpenGL 4.3 context request fails with EGL_BAD_MATCH. Neither result
executes application shaders or certifies hardware classes. Before this slice the
runtime selected `wgpu::Backends::PRIMARY`; the existing GLES implementation is a separate
explicit candidate. Preserve the prior Vulkan/Dozen failures and all Windows
hardware/Linux software results.

## Data model and execution contract

Add a safe `WgpuBackendSelection` enum with `Primary` as its default and an
explicit `Gl` selection. Add non-exhaustive `WgpuRuntimeOptions` with backend,
power preference and optional timestamp-request fields. Defaults retain the
existing primary/high-performance/no-timestamps behavior. New synchronous and
asynchronous options constructors own their runtime. Existing constructors and
the shared runtime keep their original selection and timestamp contracts.

Map `Primary` to exactly the existing primary backend set and `Gl` to GL only.
Use original instance flags, adapter admission, device limits and feature
requests. Do not select a backend from global environment overrides, request
software fallback, override Vulkan conformance flags or relabel adapter classes.
Require compiled GL support before creating an instance for an explicit GL
request. A WebGPU-only browser build otherwise lets wgpu select WebGPU despite
the requested GL mask; return an argument error before that selection. Primary
admission retains its original policy.
After adapter selection, require advertised compute-shader support for explicit
GL before creating a device; WebGL and graphics-only GL cannot satisfy this
compute runtime's contract. Original Primary admission remains unchanged.
No new heavy dependency, core change, unsafe code, shader or implicit device
transfer. Driver/platform support can still reject an explicit GL request.

The public RGB-D example accepts an optional trailing `--backend primary|gl`
while preserving the original three positional arguments. Reject malformed
selection before opening payloads or creating a runtime. Record requested and
actual backend separately, omitting adapter model names. All existing image,
voxel, neighborhood, transfer and resource checks stay unchanged.

The data-only report auditor gains a keyword-only expected backend, defaulting
to the original Vulkan expectation. Explicit GL auditing requires both actual
`Gl` and requested `gl`. Keep all original numeric/schema/process/frame gates,
both preferences and both distinct hardware classes. `Other`, software classes,
duplicate classes or incomplete runs cannot satisfy hardware coverage. A passing
operation receipt is distinct from full hardware coverage.

## Fixed public-data execution

Commit `tum-fr1-xyz-gles-hardware-plan.json` before the first wgpu GL admission
or shader call. Reuse the original checked public input directory
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/gpu-conformance-input-v1/`, manifest
SHA-256 `f0c2fe90f845de3bb434b7d0e24b10b4ddfe8754bb39455841bcaa654c98a7ae`
and all three full frames, indices 0/399/797. Run high and low preferences on
Linux with explicit GL only, retaining each failed process and frame. The
system D3D12/EGL driver remains installed as observed; do not replace drivers,
force version strings or change noncompliant-adapter flags.

Keep exact integer grayscale, 11x11/259x3 blur, histogram, voxel membership and
radius counts; centroid coordinate error remains <=1e-5m. Require original
explicit transfer receipts, zero mid-chain host image reads and 64MiB retained
pool budgets. Both complete runs and both raw IntegratedGpu/DiscreteGpu classes
are still required for hardware coverage. No inference from the capability
probe or renderer strings can substitute for those runtime receipts.

Before execution, freeze source/protocol/input/binary identities. Retain the
original frozen GPU executables and all five Python wheels/extensions. Build
new Windows and Linux executables under
`E:/SpatialRustBuild/explicit-gles-v1/` with at most four jobs, a 20GiB free
floor and a 16GiB combined build/cache/temp cap. Native defaults, explicit GL
selection, parser/backend mismatch rejection, old report acceptance and the
original independent CPU references must pass appropriate local checks before
dataset execution. Include WASM/feature compatibility checks.

Write new outputs only under
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/gles-hardware-conformance-v1/`,
with a 128MiB result cap and ten minutes per preference. Reuse inputs without
new image downloads or payload copies. Preserve logs/hashes and both failures
and successes. Publish actual selected classes, capabilities, operation errors,
transfers, pools and controlled storage without local adapter model names.
This is selected-frame/backend conformance, not broad GL certification,
bare-metal Linux, motion accuracy, calibrated canonical mapping or completion
of the operational 80% objective.

## Frozen execution outcome (2026-10-20)

The contract/plan was committed at
`db4b9018e89e56121ba4b18c5c4f2dc4d1fc5048`, before wgpu GL admission. Both
release executables use implementation
`ebf117dc781500bcdacb24e1afb1f95855155740`. The
[compact receipt](receipts/2026-10-20_explicit_gles_gpu.json) binds source,
input, binaries, raw outputs, audits, diagnostic sources and storage. The plan's
Git blob and working-file bytes have separately labeled hashes.

| Execution | Observed frames | Operation audit | Hardware coverage |
| --- | --- | --- | --- |
| Windows primary/high, Vulkan, DiscreteGpu | 0/399/797, all pass | Pass | Both Windows classes pass together |
| Windows primary/low, Vulkan, IntegratedGpu | 0/399/797, all pass | Pass | Both Windows classes pass together |
| Linux explicit GL/high | None; device creation fails, process exits 1 | Fail | Unaccepted |
| Linux explicit GL/low | None; device creation fails, process exits 1 | Fail | Unaccepted |

The Windows control passes all six unchanged image/point/transfer/pool cases.
Maximum centroid coordinate error is `3.936750760269447e-6 m`, within the fixed
`1e-5 m` limit. Both Linux stderr files report
`failed to create wgpu device: Parent device is lost`; neither process emits an
operation report. Retain all six unobserved Linux frame cases and both failures.
The EGL capability probe does not establish any Linux application execution.

A separate bounded admission-only diagnostic uses pinned wgpu 24.0.5 and its
own offline lock. Its redacted result reports raw class `Other`, GL 4.2 core,
GLSL 4.20 and advertised compute support. Device creation reproduces the failure
while wgpu builds its internal indirect-validation pipeline:
`The selected version doesn't support Features(DYNAMIC_ARRAY_SIZE)`.
Inspection of the pinned primary source explains the difference: wgpu-hal
prefers desktop GL when EGL advertises it; Naga 24 requires desktop GLSL 4.30 or
ES 3.10 for dynamic arrays. The separate ES 3.1 capability context therefore
does not imply that wgpu selected ES. Source paths/hashes and the private log's
identity are bound in the receipt; private renderer/model strings are omitted.
Two diagnostic launcher failures occurred before GPU calls and are also retained.
No driver replacement, version override, disabled validation, alternate backend
fallback or class relabeling was used to make this result pass.

After the frozen runs, review added the compiled-GL and compute preflights above;
the receipt separately binds its final source and checks. The measured dataset
results and executables remain frozen at the implementation commit above.
This admission-only guard changes no kernel, numeric limit or transfer.

Validation passes eleven runtime tests, three example tests (including the
independent CPU references), 34 report/backend cases, GPU all-target Clippy,
WASM `gpu-image` checking and workspace formatting. The full Python suite passes
973 tests with three optional ONNX skips. The numerical kernels and original
execution gates are unchanged.

No dataset is downloaded and no image payload is copied. Only the completed
passing Python test scratch at
`E:/SpatialRustBuild/explicit-gles-v1/pytest-full-v1/` is removed after checking
the XML, exact absolute parent and every child for reparse points: 275,674,423
logical bytes across 1,048 files. Logs, XMLs, original inputs, all five
wheels/extensions, old/new GPU executables and failed/successful results remain.
After diagnostics and cleanup, controlled build/cache/temp storage is
4,965,811,047 logical bytes (16GiB cap); the result directory is 78,351 bytes
(128MiB cap), with more than 632GiB free on E (20GiB floor). These are logical
file counts and point-in-time free space, not peak process memory measurements.
After the final compiled-backend guard checks, the separately recorded
controlled total is 6,040,968,384 bytes, still below the same 16GiB cap with
more than 632GiB free on E. All release executables are checked unchanged.

Typed explicit GL selection is implemented. Linux hardware execution and both
raw hardware classes remain open; this slice makes no GL conformance or
operational 80% claim. A future ES-context integration needs its own ownership
contract, fixed protocol and unchanged operation/hardware gates before admission.

# Explicit GLES runtime and public GPU validation

Operational slice 146I. A capability-only probe of the existing Ubuntu 22.04
WSL2 system driver creates an OpenGLES 3.1 context through D3D12 and reports
1,024 compute invocations, 32KiB shared storage and 16 compute storage blocks.
The separate OpenGL 4.3 context request fails with EGL_BAD_MATCH. Neither result
executes application shaders or certifies hardware classes. The current runtime
selects `wgpu::Backends::PRIMARY`; the existing GLES implementation is a separate
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

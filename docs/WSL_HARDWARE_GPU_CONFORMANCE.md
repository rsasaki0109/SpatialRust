# Public RGB-D hardware conformance through WSL2

Date: 2026-10-19. Operational slice 146F. The
[fixed plan](../benchmarks/tum-fr1-xyz-wsl-hardware-plan.json) is committed
before the first device/dataset execution with the isolated driver. Building
the driver alone is not evidence of hardware execution.

This extends [the existing public GPU protocol](PUBLIC_GPU_RGBD_CONFORMANCE.md)
without changing its frames, scalar references, exact mismatch gates, 1e-5m
centroid tolerance, explicit transfers or retained-pool budgets. Both power
preferences must complete indices 0, 399 and 797. The original requirement
for distinct integrated and discrete hardware classes remains. A CPU adapter
or two preferences selecting one hardware class cannot pass that gate.

The old Linux executable and input manifest are reused with their fixed
SHA-256 values. Before execution, verify the current GPU/core/math/image/search
Git trees against the original protocol revision, and freeze driver library,
ICD, executable and input hashes in an external receipt. Dataset images,
derived point arrays and all older results stay outside Git and unchanged.

## Isolated runtime

The test environment is Ubuntu 22.04 under WSL2. Mesa Dozen translates Vulkan
to the host D3D12 bridge through `/dev/dxg`; this does not establish bare-metal
Linux support. The official [Mesa 26.2.5 source checksum](https://docs.mesa3d.org/relnotes/26.2.5.html)
binds the driver source. Mesa's pinned DirectX-Headers and libdrm sources are
recorded separately. Build only `microsoft-experimental` with Gallium,
OpenGL/EGL/GLX/GBM, LLVM, tools and driver tests disabled. Source, tool virtual
environment, build, temporary files and private runtime prefix stay in
`E:/SpatialRustBuild/mesa-dozen-v1/`.

Stage only the built Dozen library, its private libdrm runtime and generated
ICD into that prefix. Select it for each child process with `VK_DRIVER_FILES`
and `VK_ICD_FILENAMES`, and include the private library directory plus
`/usr/lib/wsl/lib` in `LD_LIBRARY_PATH`. See
[Mesa's process-local driver instructions](https://docs.mesa3d.org/install.html).
Do not update system drivers, install globally or run `ldconfig`.

Invoke the frozen `gpu_public_rgbd_conformance` executable with the existing
input directory, manifest SHA-256 and `high`, then `low`. Preserve stdout,
stderr and status for both attempts, including initialization failures. The
runner emits backend, hardware class and limits without adapter model names.
Its independent CPU references and public upload/readback receipts determine
correctness; successful enumeration or compilation cannot establish it.

The data-only `C:/Users/rsasa/Workspace/SpatialRust/scripts/audit_public_gpu_conformance.py`
also checks complete preference/frame sets, process status, exact operation
counts, finite centroid errors, transfer sizes, retained-pool bounds and
cross-preference image hashes. It reproduces the prior hardware/software
distinction from actual published receipts and rejects inconsistent success
flags. It checks report consistency; execution authenticity still depends on
the separately frozen process/binary/input/driver provenance.

## Storage and scope

Reuse `E:/SpatialRustData/maturity-80/tum-fr1-xyz/gpu-conformance-input-v1/`
directly. This slice downloads and copies no dataset payload. New stdout,
stderr, runtime caches and receipts use
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/wsl-hardware-conformance-v1/`.
Require at least 20GiB free, cap controlled build outputs at 8GiB and result
outputs at 128MiB, and bound each preference to 600 seconds. Monitor those
bounds during execution. These counters exclude operating-system and host
driver allocations and are not peak-memory measurements.

This is three development frames under one bridge configuration. Accurate
motion, measured canonical sensor clock/calibration, full mapping and release
acceptance remain separate open gates. No throughput or competing-library
maturity percentage follows from these cases.

## Observed outcome

Protocol commit `952d34e2fee702717a172509430702f62912f5c5` and execution-source
commit `893ba09665ba5eca59453644cca2f147c91b4f1c` precede both attempts.
The original executable, input payloads and earlier Windows/Linux receipts
remain byte-identical. The relevant GPU/core/math/image/search Git trees also
match the original protocol revision. The isolated source build succeeds;
that success does not establish operation conformance.

| Preference | Process | Operation frames executed / planned | Hardware acceptance |
| --- | --- | ---: | --- |
| high | Exit 1: no compatible wgpu adapter | 0 / 3 | fail |
| low | Exit 1: no compatible wgpu adapter | 0 / 3 | fail |

Both stdout files are empty. Complete stderr is preserved and reports that
Dozen is a testing implementation followed by the unchanged runtime's adapter
admission error. No image, voxel or radius shader runs, readback correctness,
or centroid error was observed. Missing measurements are not zero errors.

A separate post-failure Vulkan capability probe enumerates one `DiscreteGpu`,
API 1.2.354, graphics/compute support and conformance version 0.0.0.0. It emits
no adapter model strings and creates no logical device or shader. wgpu-hal
24.0.4's
[adapter admission source](https://github.com/gfx-rs/wgpu/blob/v24.0.5/wgpu-hal/src/vulkan/adapter.rs)
excludes a non-MoltenVK driver with conformance major zero unless an explicit
noncompliant-adapter flag is set. The frozen runtime uses the default flags.
These observed properties and source checks explain its exclusion; neither
driver certification nor admission is overridden to obtain a passing result.
The probe also reports no `VK_KHR_swapchain`; later device creation is untested.
Even enumeration of one class cannot meet the unchanged two-class gate.

The [compact receipt](receipts/2026-10-19_wsl_hardware_gpu.json) records both
initialization failures, every planned but unexecuted frame, runtime/source
hashes, the capability diagnostic, storage observations and report auditing.
The final auditor uses `null` when no centroid error was observed; the original
raw execution receipt retains the earlier auditor's zero initializer and is
preserved with its original source hash. This reporting correction does not
rerun or revise operation results.

The earlier six Windows hardware and six Linux software operation cases retain
their outcomes. WSL2 hardware operation conformance and bare-metal Linux
hardware support remain unproven. Any experimental change to admission or
driver configuration requires a separate protocol and new result directory.

Final local validation passes 892 Python cases, including 24 new report-audit
cases and the unchanged 13 input/projection cases; three optional ONNX cases
skip. Both new Python files parse with the Python 3.8 grammar. Native Rust,
wheel, shaders and Cargo dependencies are unchanged; no new native build is
needed for these data-only checks. The Mesa build completes with ten compiler
warnings retained in its log; it is not a Vulkan certification test.

The final controlled E-drive source/tools/build/runtime tree contains
596201206 logical bytes, with the 68652756-byte Mesa archive cached separately.
New execution/diagnostic results contain 23709 logical bytes. Removing only
three fully checked passing-test scratch directories reclaims 404474664 logical
bytes; XMLs, source archives, wheels, frozen binaries, data and all old/new
execution results remain. These are observed regular-file sizes, excluding
filesystem symlink targets, not disk-allocation or peak-memory measurements.

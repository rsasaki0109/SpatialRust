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

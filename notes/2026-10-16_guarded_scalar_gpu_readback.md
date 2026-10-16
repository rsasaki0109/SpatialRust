# Unwind-safe scalar GPU readback

Parent: `8cb7556bce4a74dc2885fdbc8226590e2485cfe6` (merged PR 242).

`read_staging_f32` previously unmapped its staging buffer explicitly after
copying the mapped prefix into an owned vector. If prefix indexing panicked,
the mapped view dropped during unwinding but the final `unmap` was skipped.
The buffer remained mapped, preventing a later map or GPU upload until the
caller manually unmapped it. This is an internal helper boundary; valid
production lengths already followed the successful cleanup path.

The reader now reuses the existing private `StagingUnmap` drop guard, as XYZ
and multi-attribute readers already do. The guard is created after successful
mapping and before the mapped view. Reverse local destruction order drops the
view before the guard unmaps on normal return or panic unwinding. There is no
new `unsafe`, API, dependency, parallel execution policy or resource pool.

Original mapping callbacks, poll/wait, map-error text, prefix bounds behavior,
bitwise copy, result length/capacity and owned return value are preserved.
The same buffer is mapped once and unmapped once on a successful call. This
adds no GPU pass, submit, transfer, staging buffer, readback or extra host wait.
Shader math and public voxel first-value/average callers are unchanged. The
scope is cleanup during unwinding; this does not change the length contract.

## Regression evidence

Three native GPU tests cover:

- **6,685,002 exact word comparisons** across zero, one, three, seventeen,
  65,537 and 1,048,577-element prefixes, the complete frozen parent function,
  full-buffer tail checks, exact capacities, reupload and owned lifetimes.
  Inputs include signed zeros, infinity, NaN payloads, subnormals and arbitrary
  words. Reupload changes the buffer while previous vectors retain their bits.
- **Original failure and recovery:** a deliberate bounds panic in the frozen
  parent leaves a readable map active. The control test inspects that map and
  manually releases it. Three new-reader panic cycles (one-past, `usize::MAX`
  and a larger excess) each permit subsequent zero/prefix/full maps, GPU
  reupload and another successful frozen-parent read without manual cleanup.
- **Four concurrent independent buffers:** each worker unwinds a bounds panic,
  remaps and reuploads its own staging storage. 524,500 result words
  remain exact after GPU buffers and the runtime have been dropped.

The reference is the complete scalar reader from the merged parent, including
its original cleanup behavior. Its normal-path bits and capacity are compared
directly; the source audit retains original map-error handling. The failure
test also exercises the live production helper.

Validation in `C:/Users/rsasa/Workspace/SpatialRust`:

- All-features/all-targets Clippy with `-D warnings`.
- **323 native tests:** 132 image, 171 voxel/readback, 18 filtering and 2 facade.
  This includes ignored GPU correctness tests and excludes benchmark tests.
- CPU-only build; minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM
  builds, with WASM warnings denied; workspace formatting and whitespace checks.
- Source audit: **478 original GPU/image Rust/WGSL files unchanged**. The sole
  original source change is scalar cleanup plus native test registration.
  All shaders, callers, other readers and mapping/copy/error bodies are exact.
- **482 source/executable hashes** pin the validated files. Tracked files and
  delivery text omit the local adapter name.

This is a resource-lifetime fix. No latency or allocation improvement is claimed
and no performance benchmark was needed to establish panic recovery.

Implementation paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback_scalar_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback_scalar_reference.rs`

Ignored logs and receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/scalar-readback-*`.

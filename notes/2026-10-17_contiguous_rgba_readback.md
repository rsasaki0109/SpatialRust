# Copy contiguous RGBA readbacks directly into owned output

Parent: `eca4726c6430197cc5897e5dca70cfdbf181cd6c` (merged PR 251).

`GpuImage::readback_u8::<4>` previously zero-initialized the complete output and
copied each logical row over it, even when mapped rows had no padding. The
private unpack wrapper now copies the validated logical prefix directly into
an owned vector when channels equal four and padded stride equals width times
four. All other layouts use the original row helper and worker selection.
Under the existing 256-byte staging row alignment, positive image widths that
are multiples of 64 qualify.

## Source work ledger

For qualifying width W and height H, output bytes B = 4*W*H:

| Source operation | Original | New |
| --- | --- | --- |
| Owned output vectors | 1 | 1 |
| Explicit zero initialization | B bytes | none |
| Logical output copies | H row copies / B bytes | 1 prefix copy / B bytes |
| GPU texture copies / submits / maps / waits | 1 each | 1 each |
| Logical device-to-host payload | B bytes | B bytes |

At 1920 x 1080, B is 8,294,400 bytes and H is 1,080. This removes a source-level
zero-initialized vector construction and row loop; it does not remove the owned
allocation or final payload copy. Compiler/allocator optimizations and physical
memory traffic are not established by source counts.

Validation order, padded staging extent, valid-prefix bounds, mapped-view guard,
unmap-on-unwind, output capacity/ownership, metadata, successful transfer receipts
and public APIs remain. No shaders, GPU dispatch, pool, dependency or core changes.
One-/two-/three-channel packing, padded RGBA rows and uploads retain their code.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 output_drop_assertions_excluded=true`. Each shape/boundary has three warmup pairs
and thirty measured pairs with alternating old/new order. Assertions and output
destruction occur after timing. The original public readback and unpacking
helpers are frozen from the immediate parent; these comparisons isolate this
change from prior readback improvements. Both complete APIs retain matching
receipts and bytes. The host-only boundary excludes GPU work, maps and waits;
the complete boundary includes staging creation, texture copy, submit/map/wait,
host output construction, metadata and receipt updates.

Pre/post observations met the configured quiet thresholds: **false**. Busy-process counts were 1 before and 1 after. GPU utilization samples (GPU%, memory%) were `[[[0, 0]], [[0, 0]]]` before and `[[[0, 0]], [[0, 0]]]` after.

Quiet observations require no other sampled process above 0.25 CPU cores and
GPU/memory utilization at most 5% in every sample. Each survey waits two seconds
between process/GPU snapshots; snapshot overhead also contributes to the actual
CPU-time interval. Process sampling excludes the measurement coordinator itself.

Pre/post samples cannot prove isolation during every timed sample. These shared-
host medians are diagnostic observations, not a general speed guarantee or a GPU
kernel throughput result. Padded rows are unchanged control cases. Different
allocators, memory state and backend synchronization can alter the results.
Tiny host copies can fall below the timer resolution; zero-duration samples
are retained in raw receipts and no paired ratio is reported for those rows.

| Boundary | Pixels | Contiguous RGBA | Original median ms | New median ms |
| --- | --- | --- | ---: | ---: |
| host_unpack_only | 64 x 3 | yes | 0.000100 | 0.000100 |
| completed_public_readback | 64 x 3 | yes | 1.030350 | 1.059250 |
| host_unpack_only | 65 x 3 | no | 0.000200 | 0.000200 |
| completed_public_readback | 65 x 3 | no | 1.042950 | 1.032150 |
| host_unpack_only | 1920 x 1080 | yes | 1.726550 | 1.454700 |
| completed_public_readback | 1920 x 1080 | yes | 4.112250 | 3.795850 |
| host_unpack_only | 1921 x 1080 | no | 1.991850 | 1.988650 |
| completed_public_readback | 1921 x 1080 | no | 6.231550 | 6.309250 |
| host_unpack_only | 3840 x 2160 | yes | 7.268400 | 6.253800 |
| completed_public_readback | 3840 x 2160 | yes | 23.742050 | 23.169650 |

## Correctness and portability

- **341 native correctness tests:** 137 image, 184 voxel/readback, 18 filtering
  and 2 facade, including ignored GPU tests and excluding benchmarks.
- **64,564,840 original/CPU bytes** across 48 channel/layout cases. The
  complete original unpacking helpers are frozen independently. Cases span zero
  output, channels one to four, row alignment boundaries, one-megapixel worker
  thresholds, 1920/1921 and 3840 widths, source prefix/suffix guards, exact
  capacities and owned outputs after source mutation.
- **50,499,904 original/CPU RGBA bytes** across ten complete public
  readbacks with aligned/padded rows, packed/ROI input, retained BGRA/alpha/range
  metadata, matching upload/readback receipts, texture recycling and runtime
  drop. Existing mapped decoder, concurrency, rejection and panic/remap tests
  remain and pass.
- The ignored paired benchmark separately passes one test with ten sets of
  thirty pairs and exact byte/metadata/receipt assertions outside timing.
- All-features/all-targets Clippy with warnings denied, CPU-only and minimal/CI
  WASM builds with WASM warnings denied, formatting and whitespace checks pass.
- **497 unchanged original GPU/image Rust/WGSL files** and **502 validated
  source/executable hashes**. Source audit removes only the four-line shortcut
  to restore the original transfer helper, retains all prior native fixtures,
  and verifies the complete frozen immediate-parent public readback/unpacking.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_packing.rs`.
Fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/mapped_readback_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/rgba_readback_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/rgba_readback_benchmark.rs`.
Ignored receipts: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/rgba-readback-*`.
The local adapter name is omitted from tracked files and delivery text.

# GPU frame partitioning and transfer accounting — 2026-10-11

## Behavior

`run_aoso_voxel_normal_frame` now combines uploaded XYZ chunks and partitions
source points directly. It previously called the host-visible centroid API,
computed one centroid per occupied voxel, copied the padded four-float centroid
records back to the host, allocated three CPU coordinate arrays, and discarded
them. That work was unnecessary because the frame retained the original source
positions and estimated one normal per source point.

The shared partition helper is also used by the existing centroid API. Its
host-visible output and averaging semantics are unchanged. Chunk boundaries do
not split voxels; source point count, position order, voxel membership, and
normal behavior remain intact.

For `n` source points and `c` occupied voxels, construction removes the centroid
dispatch and `16 * c` bytes of device-to-host payload. GPU segment arrays remain
resident. Each nonempty partition still synchronizes its four-byte cell count;
construction builds a voxel partition and a separate radius grid, so the frame
receipt now reports eight bytes rather than incorrectly reporting zero. This
slice does not eliminate those scalar synchronizations.

## Receipt contract

- XYZ upload and chunk concatenation each report `12 * n` bytes, measured from
  logical point counts. Empty construction reports zero bytes in every direction.
- `GpuVoxelSegments::construction_readback_bytes()` records the bytes actually
  copied by scalar synchronization, excluding later metadata readbacks and
  transfers of caller inputs.
- Rebuilding a nonempty radius grid adds four device-to-host bytes. Reusing a
  matching grid while estimating normals adds no transfer bytes.
- Attribute reduction records both the concatenation device copy and the reduced
  attribute payload readback. Uploads made before attaching attributes happen
  outside frame methods and remain caller-accounted.
- Receipts cover payload copies and scalar metadata performed by frame methods.
  Kernel parameter uniforms, shader storage reads/writes, and operations made
  directly through exposed buffers or attached objects are excluded.

## Validation

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.

`cargo test -p spatialrust-gpu --features gpu-aoso-staging,gpu-image -- --test-threads=1`
passes all 53 tests on NVIDIA GeForce GTX 1660 Ti / Vulkan. The tests include
seven image GPU regressions (known pixels, CPU luma parity, retained chains,
pipeline caching, and upload/readback accounting) as well as point-cloud kernels.
The related filtering and segmentation run below passes another 18 and 10 tests,
respectively, for 81 passing tests in total:

```text
cargo test -p spatialrust-filtering -p spatialrust-segmentation --features spatialrust-filtering/filter-voxel-gpu,spatialrust-segmentation/segment-euclidean-gpu -- --test-threads=1
```

New and expanded point-cloud checks cover:

- Negative/duplicate voxel keys split across chunk sizes 1, 2, and 64, compared
  with CPU partition metadata while retaining every source point in order.
- A planar frame whose voxel count is smaller than its point count, with correct
  per-point normals and explicit position readback added to the eight-byte ledger.
- Valid empty XYZ columns, empty grid rebuilds, and zero transfer accounting.
- Attribute copies/readback, cached-grid reuse, changed-radius rebuilds, and
  invalid-radius rejection without altering the receipt.
- Construction readback bytes for counts 0, 1, 31–33, 255–257, 511–513, and 65,537,
  together with the existing CPU radix/segment reference comparison.

Focused all-feature/all-target Clippy, workspace formatting, and whitespace
checks pass. The full repository CI remains a separate PR gate.

## Performance comparison

The existing `aoso_frame_pipeline` benchmark now records adapter identity,
initializes shared pipelines before warmup, and waits for the final normal
dispatch to complete. Both the parent and revised implementation use this same
benchmark harness. Timings include upload, two key-sort/compact paths, radius
normal estimation, completion waiting, and frame recycling; they are not shader
timestamps.

Parent: `d646ab11795a91829d703485ef4c50f47c817880`. Environment: Windows 11,
Rust 1.98.1, wgpu 24.0.5, Criterion 0.5.1, NVIDIA GeForce GTX 1660 Ti / Vulkan,
driver 596.36. The synthetic fixture uses 16,384-point chunks, inverse voxel leaf
size 20.0, and normal radius 0.08. Each process uses 10 samples and two-second
warmup/measurement windows.

Criterion mean estimates, milliseconds; brackets show 95% bootstrap confidence
intervals of the mean. Confirmation runs each point count in a fresh process.

| Source points | Parent with centroid readback | Partition-only confirmation | Mean reduction |
| ---: | ---: | ---: | ---: |
| 10,000 | 10.684 [10.477, 10.924] | 8.636 [8.393, 8.902] | 19.2% |
| 65,536 | 13.358 [12.905, 13.782] | 10.369 [9.823, 10.970] | 22.4% |

An initial combined-size probe was noisy at 10,000 points and Criterion did not
detect a performance change there; its 65,536-point mean was 10.507 ms. The
isolated confirmation avoids relying on that noisy small-cloud result. These
short-window measurements describe this adapter and fixture, not a universal
speedup or CPU/GPU crossover.

Commands:

```powershell
# Parent plus the synchronized benchmark harness, before source changes:
cargo bench -p spatialrust-gpu --features gpu-aoso-staging --bench aoso_frame_pipeline -- --save-baseline frame-readback-before --sample-size 10 --warm-up-time 2 --measurement-time 2
# Revised implementation, each size in its own process:
cargo bench -p spatialrust-gpu --features gpu-aoso-staging --bench aoso_frame_pipeline -- aoso_frame/upload_voxel_radius_normals/10000 --save-baseline frame-readback-after-confirm --sample-size 10 --warm-up-time 2 --measurement-time 2
cargo bench -p spatialrust-gpu --features gpu-aoso-staging --bench aoso_frame_pipeline -- aoso_frame/upload_voxel_radius_normals/65536 --save-baseline frame-readback-after-confirm --sample-size 10 --warm-up-time 2 --measurement-time 2
```

## Next slices

Optional GPU timestamp measurements should serve both image and point-cloud
execution and distinguish stage work from submission/synchronization overhead.
Point-cloud compaction still scans the entire buffer at each stride, and
temporary-buffer reuse/retention needs a bounded-memory follow-up. These are
separate changes; no `ExecutionPolicy::Auto` threshold is changed here.

# GPU voxel acceleration — 2026-07-16

## Scope

This slice revalidates the end-to-end `point_xyzi` centroid voxel path and fixes
three GPU execution problems:

1. Headless compute previously requested the low-power adapter unconditionally.
   The default now prefers a high-performance adapter, with an explicit
   `WgpuPowerPreference::LowPower` constructor option retained for callers.
2. A recycled voxel-key output buffer lacked `COPY_DST`. An equal-sized later
   position upload could therefore trigger a wgpu validation panic.
3. Bitonic sort and prefix-scan stages submitted one command buffer per stage.
   Sort stages and scan stages are now batched, and large zero initialization
   uses GPU `clear_buffer` instead of allocating and uploading host zero vectors.

No GPU model name is recorded in this public receipt.

## Environment and workload

- OS: Windows 11, x86-64
- CPU topology: 6 cores / 12 logical processors
- GPU class: high-performance discrete adapter
- Backend: Vulkan
- Rust: 1.97.0
- Schema: `StandardSchemas::point_xyzi()`
- Mode: centroid with average attributes
- Leaf size: `4.0`
- Build: Criterion release profile

Canonical benchmark:

```powershell
cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample
```

GPU probes were run as isolated 10-sample Criterion processes so device and
driver allocations were released between point counts.

## Optimized GPU medians

| Input points | GPU before final submit batching | GPU after | Change |
| ---: | ---: | ---: | ---: |
| 100,000 | 23.62 ms | 21.047 ms | -10.9% |
| 200,000 | 33.21 ms | 24.549 ms | -26.1% |
| 500,000 | 51.816 ms | 35.829 ms | -30.9% |
| 750,000 | 74.63 ms | 54.962 ms | -26.4% |
| 1,000,000 | 78.258 ms | 65.890 ms | -15.8% |
| 2,000,000 | 129.57 ms | 104.82 ms | -19.1% |

The pre-batching 750k run was affected by the long-lived benchmark process and
is retained only as a local before/after optimization probe. Claims are limited
to the isolated post-change medians.

## CPU/GPU decision

The current CPU medians remain lower at every measured point count. At 2M, CPU
is 47.290 ms and GPU is 104.82 ms. The centroid Auto threshold is therefore set
to `usize::MAX`: Auto stays on CPU until a new measured crossover exists.
Explicit GPU execution remains available through
`VoxelGridDownsampleConfig::without_gpu_min_points()`.

This decision applies to the round-trip filter API. It does not imply that
GPU-resident chains are slower, because those chains can avoid repeated upload
and readback.

## Correctness and safety

- The recycled-buffer regression test forces a key-output buffer to back a
  later, equal-sized position upload.
- All 26 `spatialrust-gpu` `gpu-wgpu` tests pass.
- GPU segment construction continues to match the CPU reference.
- Public APIs remain safe and no implicit CPU/GPU transfer was added.

## Next optimization

Bitonic sorting still performs `O(n log² n)` compare/swap work even after submit
batching. A stable GPU radix sort for packed voxel keys is the next material
performance step before reconsidering automatic round-trip GPU selection.

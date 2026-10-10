# Stable GPU voxel radix sorting — 2026-10-11

## Scope

Voxel segment construction now uses a stable four-bit LSD radix sort instead
of the bitonic compare/swap path. The existing public functions and GPU-resident
segment representation are retained. Centroid, approximate-first, AoSoA/frame
and sparse-radius-grid consumers all use the shared implementation.

- Signed i32 XYZ keys are ordered lexicographically by flipping the sign bit
  and processing z, y, then x. Increasing source indices are preserved for equal
  keys. Host i64 keys retain their existing i32 saturation behavior.
- Each digit uses workgroup lane masks for deterministic local ranks, a GPU
  histogram prefix, and a stable scatter. No optional subgroups, atomic append
  ordering or cross-workgroup spin waits are used.
- All 24 digit passes share one command submission and scratch allocations.
  Work scales linearly in point count for the fixed 96-bit coordinate width.
- Exact-sized entries remove power-of-two padding and the post-sort filtering
  scan, gather, and four-byte valid-entry count readback. The existing segment
  compaction cell-count readback remains; this is not a completely asynchronous
  GPU-resident pipeline.
- The legacy `padded_count` argument remains accepted when at least the point
  count, but no padding is allocated. Internal callers pass the actual count.
  Buffer size, storage usage, index width and dispatch limits are checked before
  sorting to return errors instead of wgpu validation failures.

## Environment

- Windows 11, x86-64
- Intel Core i7-9750H, 6 cores / 12 logical processors
- NVIDIA GeForce GTX 1660 Ti; Vulkan; NVIDIA driver 596.36
- Rust 1.98.1, Criterion 0.5.1, wgpu 24.0.5
- Parent baseline: `52c547f6f761f134c4693fe21b65eea79f9d377c`
- Workload: existing `point_xyzi` centroid benchmark, leaf size `4.0`, release
  profile; explicit GPU policy with its minimum-point threshold disabled

The benchmark now prints the selected adapter/backend and initializes cached
pipelines before warmup. Timings cover the filter round trip, including host
attribute handling and output readback, not just the sorting shader.

## Same-session baseline and isolated confirmation

Criterion mean estimates, milliseconds. The parent baseline used 10 samples,
one-second warmup and measurement windows. Confirmation ran each point count in
a fresh process with 10 samples and two-second warmup/measurement windows.
The brackets give 95% bootstrap confidence intervals of the mean.

| Points | Parent bitonic | Radix confirmation | Mean reduction |
| ---: | ---: | ---: | ---: |
| 100,000 | 15.400 [13.656, 17.499] | 11.878 [10.887, 13.100] | 22.9% |
| 1,000,000 | 56.022 [55.109, 56.858] | 38.097 [36.282, 41.052] | 32.0% |

An initial one-second radix probe produced means of 7.94 ms at 100k and
32.16 ms at 1M. These faster observations are not used as the confirmed result.
The 1M confirmation had four Criterion outliers. The sample windows and
iteration counts differ, so these short probes demonstrate a local improvement
but do not establish a general hardware-independent speedup.

A separate mixed CPU/GPU sweep with one-second windows produced:

| Points | CPU mean | GPU radix mean |
| ---: | ---: | ---: |
| 100,000 | 7.646 ms | 17.282 ms |
| 1,000,000 | 39.271 ms | 36.629 ms |
| 2,000,000 | 56.902 ms | 60.310 ms |

The sweep overlapped local repository maintenance at its first point count, and
its GPU process retained allocations between sizes. Its 100k result is slower
than the parent probe and illustrates the timing variation; it is not used for
the before/after claim. The 1M CPU/GPU reversal is small and not a validated Auto
crossover. Automatic centroid placement remains on CPU pending a larger,
controlled CPU/GPU rebaseline. The July README table/plot is labeled as historical
and is not silently replaced with these short probes.

## Reproduction

At the parent revision, save the comparison baseline:

```powershell
cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample -- '^voxel_downsample/(100000|1000000)/gpu_centroid$' --sample-size 10 --measurement-time 1 --warm-up-time 1 --save-baseline bitonic
```

At the radix revision, run each size in a separate process against that baseline:

```powershell
foreach ($points in 100000, 1000000) {
    cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample -- "^voxel_downsample/$points/gpu_centroid$" --sample-size 10 --measurement-time 2 --warm-up-time 2 --baseline bitonic
}
```

The local confirmation invoked the already-built Criterion executable with
`--bench` and these same arguments, after all builds and tests completed.
Raw estimates are retained under Criterion's ignored target output directory.

## Verification

```powershell
cargo test -p spatialrust-gpu -p spatialrust-filtering -p spatialrust-segmentation --features spatialrust-gpu/gpu-aoso-staging,spatialrust-gpu/gpu-image,spatialrust-filtering/filter-voxel-gpu,spatialrust-segmentation/segment-euclidean-gpu -- --test-threads=1
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline gpu_voxel_matches_cpu -- --test-threads=1
cargo clippy -p spatialrust-gpu -p spatialrust-filtering -p spatialrust-segmentation --all-features --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

- 51 GPU tests, 18 filtering tests and 10 segmentation tests pass.
- Both facade MVP tests pass for normal and RGB attributes: 81 related tests in
  total. AoSoA chunk merging, frame receipts, sparse-grid normals and clustering
  remain compatible.
- Five new sorting regressions cover empty/singleton inputs, lane/workgroup and
  prefix-chunk boundaries through 65,537 points, full-width signed keys and
  reversed input, saturated extrema, invalid counts/usage and device limits.
  Device indices are inspected directly for both all-equal and interleaved
  duplicate keys, so host-side per-cell sorting cannot conceal instability.
- Clippy with warnings denied, repository formatting and whitespace checks pass.
- No unsafe code or implicit host/device transfer is introduced.

## Remaining performance work

Use GPU timestamps and longer isolated CPU/GPU runs to separate the radix prefix,
scatter, segment scan, allocations and final readback costs. Segment compaction
still uses a full-buffer scan and scalar synchronization; removing those limits
is a separate change. The fixed 24 digits also have launch overhead on small
inputs, so this receipt does not claim acceleration at every point count.

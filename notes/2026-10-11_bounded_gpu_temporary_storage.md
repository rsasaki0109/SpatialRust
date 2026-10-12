# Bounded GPU temporary storage — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Parent slice: `codex/parallel-image-point-execution` at
`5a07e0d6686c46a32af6453a249ee98e14fc1ee2` (PR 123).
This branch implements the temporary-storage row in
`C:\Users\rsasa\Workspace\SpatialRust\docs\ROADMAP.md`.

## Behavior and ownership

- `GpuBufferPool` now retains at most its byte budget and eight buffers per
  exact size. Its default is 64 MiB; `with_budget`, `set_budget`, `stats`, and
  `clear` are additive provisional APIs. Mismatched sizes or missing storage /
  copy-src / copy-dst usage are discarded. A directly constructed public pool
  must be used with one device, after submitting all users and relinquishing
  other handles; this existing explicit ownership obligation remains.
- `WgpuRuntime` owns an independent private 64 MiB scratch pool. Radix scratch,
  histograms, offsets, ranks, compaction flags/prefix/atomic counts/dummy binding,
  and recursive scan totals/prefixes use it. Uniform and staging allocations
  remain outside this slice. Segment outputs and original entries are not
  recycled, so live result objects cannot be overwritten by subsequent frames.
- `ScratchBuffers` retains private buffer leases until `submit_and_recycle`
  submits their encoder. Abandoned batches drop their handles without returning
  them to the cache. Acquired buffers are uninitialized: each shader overwrites
  all read ranges, and compaction atomic counts are explicitly cleared.
- The four-byte final prefix copy is appended to the compaction encoder before
  scratch is released. Future submissions on the same queue can reuse the
  prefix without racing that copy. This removes the separate scalar-copy
  submission while preserving the existing scalar map/wait and receipt. No
  extra synchronization or hidden host/device copy is introduced.
- Image textures have a global 64 MiB RGBA8 payload budget, replacing the old
  unbounded-across-resolutions per-key cache. Their eight-per-resolution limit
  remains. Existing image/CHW reuse is preserved; image allocation reduction
  versus a deliberately disabled cache is not a new speedup versus PR 123.
- Image and AI-tensor objects retain the runtime's identity token. Moving a
  runtime preserves valid ownership; foreign readback/recycle targets reject
  them before reaching another runtime's texture or buffer pool.
- Pools reject incoming resources when full rather than evicting existing
  entries. Lowering a budget below current retention clears that pool. Zero
  disables retention, and explicit clearing preserves lifetime counters. Empty
  size buckets are removed; rejected size/resolution churn creates no buckets.
- Budgets cover recycled buffer bytes or RGBA8 texel payloads. They do not cap
  active outputs, uniforms, mapped staging, cache metadata, driver overhead,
  actual device reservations or process RSS. Three independently saturated
  default pools would retain at most 192 MiB of payload. Dropping cached wgpu
  handles leaves actual release subject to outstanding command lifetimes.
- `GpuResourcePoolStats` snapshots budget, cached bytes/resources, requested
  allocations, cache hits and discarded resources for one pool. Counters are
  lifetime totals, not complete frame-allocation or peak-memory receipts.
- CPU analysis already uses bounded worker-local 256-bin arrays (2 KiB each)
  after PR 123. This slice adds no CPU execution behavior or dependencies.

Implementation files:

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\resource_cache.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\upload_cache.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\scratch.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\runtime.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\voxel_radix.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\voxel_compact.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\voxel_scan.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\gpu_image.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\ai_tensor.rs`

## Correctness checks

New tests cover the exact budget boundary, per-key limit, zero/oversized values,
10,000 candidate size keys, shrink/clear counter semantics, incompatible buffer
usage/length, live buffer writes after budget changes, resolution churn,
retention-disabled image readback, eight repeated resident image chains, runtime
moves and foreign image/tensor recycling, and an abandoned encoder that is
submitted after a second lease was acquired.

Point tests compare against the CPU implementation through 65,537 elements with
empty, singleton and 256-element boundaries; repeatedly overwrite scratch with
different key distributions, verify zero warmed scratch allocations, retain an
old output through subsequent frames and cache clearing, exercise four-byte and
zero budgets, and run four concurrent threads with three builds each. Existing
hierarchy parity through 1,000,003 elements and timing/transfer contracts also
run unchanged.

## Measurements

Hardware: Windows, Intel Core i7-9750H, local GPU, Vulkan,
NVIDIA driver 596.36. Rust/Cargo 1.98.1, wgpu 24.0.5. Other desktop/Python work
continued; timings are local observations, not portable performance guarantees.
Benchmark builds and GPU tests finished before the Criterion measurement.

`gpu_pool_reuse` compares budget 0 with 64 MiB on this implementation, after three
warmup runs. Twenty completed frames follow, with uploads outside timing.
Point timings include entry construction, 24 radix digits, compaction, output
allocation and four-byte scalar readback. Every iteration explicitly waits for
completion. The image source is uploaded once and the output tensor is recycled
after explicit completion; it has zero payload readback.

| Workload | Budget | Pooled device allocations/frame | Reuses/frame | Retained payload |
| --- | --- | --- | --- | --- |
| 65,536 points | 0 | 11 scratch | 0 | 0 |
| 65,536 points | 64 MiB | 0 scratch | 11 | 1,869,832 bytes / 10 buffers |
| 1,048,576 points | 0 | 13 scratch | 0 | 0 |
| 1,048,576 points | 64 MiB | 0 scratch | 13 | 29,917,320 bytes / 12 buffers |
| 640×480 → 320×240 image | 0 | 5 textures + 1 CHW buffer | 0 | 0 |
| 640×480 → 320×240 image | 64 MiB | 0 textures + 0 CHW buffers | 5 textures + 1 buffer | 614,400 texture bytes + 307,200 buffer bytes |

The pooled source image remains live and is excluded from cached texture bytes.
These zeros apply to the measured pools; uniforms, readback staging and three
caller-owned segment buffers still allocate. The sequential 20-frame timing
probe reported 2.674/2.481 ms for 65,536 points, 13.508/11.325 ms for 1,048,576
points and 0.718/0.768 ms for images (disabled/enabled). The small image probe
showed no timing gain, and allocation counts are its useful evidence. Confidence
intervals for the longer point comparison follow.

Criterion: 20 samples per case, 3-second warmup, at least 5-second measurement,
uncached then cached cases in one release executable. The same public resident
keys API, key distribution and completed-work boundary are used. Values are
Criterion time estimates (slope) and 95% bootstrap intervals. No old Criterion
baseline is used for this comparison.

| Points | Retention disabled, ms [95% CI] | 64 MiB retained, ms [95% CI] | Local time reduction |
| --- | --- | --- | --- |
| 65,536 | 2.1691 [2.0946, 2.2421] | 2.0731 [2.0358, 2.1255] | 4.4%; intervals overlap |
| 1,048,576 | 12.732 [12.642, 12.844] | 10.971 [10.922, 11.027] | 13.8% |

The small-cloud comparison is inconclusive given overlapping intervals and
shared desktop load. The million-point observation is encouraging, but these
sequential cases are not a controlled cross-device performance claim. The
repeat-frame allocation counts and enforced retention bounds are deterministic
acceptance evidence. Cached/uncached cases compare retention behavior within
this implementation; batching the scalar copy is common to both cases.

Local ignored measurement artifacts:
`C:\Users\rsasa\Workspace\SpatialRust\target\automation\bounded-scratch-criterion.log`
and
`C:\Users\rsasa\Workspace\SpatialRust\target\automation\bounded-scratch-estimates.json`.

## Reproduction and validation

Run from `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
cargo run -p spatialrust-gpu --release --features gpu-image --example gpu_pool_reuse
cargo bench -p spatialrust-gpu --features gpu-wgpu --bench voxel_segments -- --sample-size 20 --warm-up-time 3 --measurement-time 5
cargo test -p spatialrust-gpu --features gpu-image,gpu-aoso-staging
cargo test -p spatialrust-filtering -p spatialrust-segmentation --features spatialrust-filtering/filter-voxel-gpu,spatialrust-segmentation/segment-euclidean-gpu -- --test-threads=1
cargo clippy -p spatialrust-gpu --features gpu-image,gpu-aoso-staging --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-wgpu --all-targets -- -D warnings
$env:RUSTFLAGS = '-Dwarnings'
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --features gpu-wgpu
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --features gpu-image,gpu-aoso-staging
cargo check -p spatialrust-gpu --no-default-features
```

All-features and minimal-wgpu native Clippy profiles passed after correcting test-module ordering.
GPU unit tests: 66 passed; shared timing integration tests: 7 passed. Browser
builds are compile checks, not real browser execution or GPU performance tests.
Wgpu-only WASM and image/AoSoA WASM compile with warnings denied; default GPU
build, web WASM/WebGPU integration build, and formatting pass. Downstream filtering tests: 18 passed; segmentation
tests: 10 passed. Total relevant native tests: 101 passed. Criterion completed
all four cases; CI is the remaining delivery gate.

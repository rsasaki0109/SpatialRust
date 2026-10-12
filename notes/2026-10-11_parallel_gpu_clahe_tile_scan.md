# Parallel GPU CLAHE tile clipping and cumulative scan

Date: 2026-10-11. Repository:
`C:\Users\rsasa\Workspace\SpatialRust`.
Branch: `codex/parallel-gpu-clahe-tile-scan`, based on the frozen CLAHE PR #131
source `5855c2ffe497e5d295f76fca9049e557dfc3722a`.
The user's continuing GPU image/point and CPU parallelization instructions,
`C:\Users\rsasa\Workspace\SpatialRust\AGENTS.md` and
`C:\Users\rsasa\Workspace\SpatialRust\.cursor\rules\long-horizon.mdc`
authorize implementation, checks and guarded dependency-ordered delivery.

## Change and invariants

The previous tile lookup shader assigned clipping, residual redistribution and
two 256-bin loops to lane zero. The new private helper at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\clahe_clip_scan.wgsl`
uses all 256 lanes for a work-efficient exclusive tree scan of pairs:
`(min(count, clip), count - min(count, clip))`, with clip zero retaining the
original count. Eight tree levels ascend, the root saves total clipped excess,
and eight levels descend to per-bin exclusive prefixes.

For inclusive bin index `b`, add its original bounded count, the uniform
redistribution batch times `b+1`, and `min(residual, floor(b/step)+1)`.
The CPU distributes residual pixels to bins `0, step, 2*step, ...`;
the analytic term counts exactly those positions through bin `b`. Zero residual
contributes zero with step one. This avoids a second scan. Each intermediate
bounded/excess component and the final prefix is at most the validated tile area
(`u32::MAX`), including maximum-count hot bins.

Histogram atomics occupy 1 KiB and vector scan entries 2 KiB, plus small shared
excess/rounding state. This fits the existing conservative 4 KiB device limit;
the planner and accepted device capabilities are unchanged. Each tile resets
all histogram bins and scan entries. Workgroup barriers bracket initialization,
tree stages, shared rounding state and subsequent tile reuse. Bounded tile-grid
striding, reflected pixel traversal and hot-bin atomics remain unchanged.

The existing exact lookup rounding and binary64 geometry/interpolation helpers
remain unchanged. There is no CPU reference upload in production. Public APIs,
three passes/one submission, ownership/recycling, metadata, timing capacity,
receipt stages and transfer bytes remain unchanged. Global temporary storage is
still `1024*tiles_x*tiles_y + 16*(width+height)` bytes. Dependencies are unchanged.

## Correctness and feature checks

The new native proof at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\clahe_lookup_tests.rs`
executes the production scan and rounding helpers for 4,096 synthetic histograms:
1,048,576 bins, comparing both cumulative and lookup words to an independent
serial u64 CPU reference. It covers all 256 residuals at hot-bin positions
0/127/255, small and near-maximum excess, dense/sparse/full-width distributions,
disabled clipping, small/intermediate/saturating clip counts and maximum tile
mass. Several histograms per workgroup exercise shared-memory reuse. Every last
cumulative bin equals the original area. Synthetic counters are test-only.

`cargo test -p spatialrust-gpu --all-features` passed 107 native tests:
83 unit, four CDF, five CLAHE, five equalization, three histogram and seven shared
image/point timing tests. The paired benchmark is ignored by default and passed
on explicit invocation. Existing odd-grid/ROI/metadata/constant/half-rounding,
65,536 binary64/geometry rows, 66,049 interpolation pixels, 2,309,120 global LUT
comparisons, concurrent live outputs, stale scratch and point-cloud regressions
all pass. The native adapter is a local GPU, Vulkan, NVIDIA 596.36.

All-feature and default all-target Clippy with warnings denied passed. Minimal
`gpu-image` tests compiled; minimal `gpu-image` and `gpu-wgpu` WASM library Clippy
passed; the facade's minimal `gpu-image` check, default native test, image-only
adapter-free planner test, formatting and whitespace checks passed. The serial
shader fixture matches the frozen parent shader with normalized newlines.
Browser shader execution
is not established by WASM compilation. CI already compiles the new GPU proofs
through the existing image profile; its adapter-free planner checks are retained.

## Paired tile-pass measurement

The explicit ignored diagnostic at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\clahe_lookup_benchmark.rs`
uses the same runtime, source texture, dimensions, uniforms, layouts and bounded
dispatch for the old and new lookup shaders. The old shader is the frozen PR
#131 implementation, stored only as a test fixture at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\tests\fixtures\clahe_lookup_serial.wgsl`.
There is no production selection or extra public API for the baseline.

Each fixture has four warmup pairs and 40 measured pairs, alternating which
shader runs first. Both passes are in the same command submission; explicit
collection reads 32 query bytes and waits before the next pair. All final LUT
entries from both shaders are independently checked against CPU histograms,
clipping, residual positions and binary64 rounding after measurement.
Timestamp intervals exclude host setup, allocation, upload, readback and gaps;
these are pass measurements, not whole-image latency or throughput.

Thirty fixtures vary resolution, grid, mixed/hot distribution and clip 0/2/40.
The first diagnostic overlapped a locally started GPU regression suite and was
discarded. The reported run started after that suite completed, with no other
agent-started GPU workload. The user's existing Python jobs were left running.
Desktop load and GPU clocks can still vary; sequential fixture comparisons and
earlier separately run CLAHE numbers do not establish speedup.

The table reports median microseconds and median paired serial/parallel ratios.
Brackets are a percentile 95% bootstrap interval of paired-ratio medians,
10,000 resamples, seed 20261011. They describe this local sample, not a hardware
performance guarantee. Raw observations and ranges are retained in the JSON.

| Image | Grid | Pattern | Clip | Serial us | Parallel us | Paired ratio [95% interval] |
|---|---|---|---:|---:|---:|---:|
| 17x19 | 8x8 | mixed | 0 | 73.968 | 11.584 | 6.381 [6.120, 6.670] |
| 17x19 | 8x8 | mixed | 2 | 73.552 | 11.440 | 6.427 [6.012, 6.642] |
| 17x19 | 8x8 | mixed | 40 | 73.104 | 11.328 | 6.450 [6.209, 6.644] |
| 17x19 | 8x8 | hot | 0 | 73.104 | 11.328 | 6.472 [5.972, 6.638] |
| 17x19 | 8x8 | hot | 2 | 73.344 | 11.408 | 6.451 [5.853, 6.640] |
| 17x19 | 8x8 | hot | 40 | 73.184 | 11.488 | 6.393 [6.149, 6.630] |
| 640x480 | 8x8 | mixed | 0 | 87.232 | 27.952 | 3.124 [2.986, 3.200] |
| 640x480 | 8x8 | mixed | 2 | 87.216 | 28.160 | 3.106 [2.972, 3.174] |
| 640x480 | 8x8 | mixed | 40 | 87.392 | 28.096 | 3.117 [3.015, 3.185] |
| 640x480 | 8x8 | hot | 0 | 87.648 | 28.416 | 3.085 [2.948, 3.148] |
| 640x480 | 8x8 | hot | 2 | 87.632 | 28.976 | 3.035 [2.937, 3.121] |
| 640x480 | 8x8 | hot | 40 | 87.776 | 28.784 | 3.061 [2.941, 3.122] |
| 640x480 | 64x64 | mixed | 0 | 3338.592 | 352.288 | 9.487 [9.455, 9.513] |
| 640x480 | 64x64 | mixed | 2 | 2672.352 | 289.392 | 9.208 [9.145, 9.231] |
| 640x480 | 64x64 | mixed | 40 | 2635.344 | 277.744 | 9.478 [9.403, 9.541] |
| 640x480 | 64x64 | hot | 0 | 2633.328 | 275.504 | 9.541 [9.493, 9.609] |
| 640x480 | 64x64 | hot | 2 | 2638.832 | 288.736 | 9.137 [9.110, 9.186] |
| 640x480 | 64x64 | hot | 40 | 2644.704 | 286.864 | 9.195 [9.166, 9.268] |
| 3840x2160 | 8x8 | mixed | 0 | 472.720 | 425.936 | 1.110 [1.107, 1.113] |
| 3840x2160 | 8x8 | mixed | 2 | 471.776 | 425.184 | 1.112 [1.108, 1.113] |
| 3840x2160 | 8x8 | mixed | 40 | 471.296 | 424.464 | 1.110 [1.108, 1.112] |
| 3840x2160 | 8x8 | hot | 0 | 486.592 | 438.768 | 1.109 [1.108, 1.111] |
| 3840x2160 | 8x8 | hot | 2 | 484.704 | 438.336 | 1.106 [1.105, 1.108] |
| 3840x2160 | 8x8 | hot | 40 | 485.280 | 438.448 | 1.106 [1.104, 1.111] |
| 3840x2160 | 32x32 | mixed | 0 | 1388.640 | 392.128 | 3.540 [3.531, 3.545] |
| 3840x2160 | 32x32 | mixed | 2 | 1388.912 | 394.128 | 3.522 [3.517, 3.531] |
| 3840x2160 | 32x32 | mixed | 40 | 1387.856 | 390.816 | 3.552 [3.544, 3.560] |
| 3840x2160 | 32x32 | hot | 0 | 1368.416 | 399.904 | 3.420 [3.410, 3.427] |
| 3840x2160 | 32x32 | hot | 2 | 1370.832 | 403.760 | 3.402 [3.395, 3.412] |
| 3840x2160 | 32x32 | hot | 40 | 1371.776 | 404.992 | 3.388 [3.380, 3.393] |

## Completed image-placement checks

The existing six-placement example ran after the paired diagnostic with the
production clip 2/grid 8x8, two warmups and ten rotating rounds, and a six-worker
CPU pool. Every completed result was byte/metadata checked against production
CPU CLAHE after its timer. Resident GPU modes wait inside timing and explicitly
read the full output for verification outside timing. Recycling/drop is outside
timing; upload/readback is included only in the listed placement paths.
These measurements have no matched old whole-image run and establish no overall
speedup. In particular the 4K mixed resident range varies substantially.

| Image | Pattern | Placement | Median ms | Observed range ms | Timed H2D / D2H bytes |
|---|---|---|---:|---|---:|
| 640x480 | mixed | CpuHost | 9.951 | [7.566, 13.114] | 0 / 0 |
| 640x480 | mixed | GpuResident | 0.747 | [0.675, 0.934] | 0 / 0 |
| 640x480 | mixed | GpuReadback | 2.281 | [1.859, 2.709] | 0 / 1228800 |
| 640x480 | mixed | UploadRoundTrip | 2.994 | [2.583, 3.422] | 1228800 / 1228800 |
| 640x480 | mixed | GpuInputCpuReturnGpu | 12.868 | [10.575, 14.342] | 1228800 / 1228800 |
| 640x480 | mixed | GpuInputCpuOutput | 11.446 | [9.969, 15.723] | 0 / 1228800 |
| 640x480 | hot | CpuHost | 9.454 | [7.225, 12.894] | 0 / 0 |
| 640x480 | hot | GpuResident | 0.654 | [0.558, 1.690] | 0 / 0 |
| 640x480 | hot | GpuReadback | 2.671 | [2.258, 4.941] | 0 / 1228800 |
| 640x480 | hot | UploadRoundTrip | 3.421 | [2.960, 5.548] | 1228800 / 1228800 |
| 640x480 | hot | GpuInputCpuReturnGpu | 13.169 | [10.165, 20.795] | 1228800 / 1228800 |
| 640x480 | hot | GpuInputCpuOutput | 12.386 | [9.273, 17.431] | 0 / 1228800 |
| 3840x2160 | mixed | CpuHost | 43.049 | [39.949, 58.112] | 0 / 0 |
| 3840x2160 | mixed | GpuResident | 11.222 | [3.966, 13.342] | 0 / 0 |
| 3840x2160 | mixed | GpuReadback | 42.555 | [32.059, 51.439] | 0 / 33177600 |
| 3840x2160 | mixed | UploadRoundTrip | 71.388 | [51.590, 79.652] | 33177600 / 33177600 |
| 3840x2160 | mixed | GpuInputCpuReturnGpu | 93.805 | [90.424, 110.696] | 33177600 / 33177600 |
| 3840x2160 | mixed | GpuInputCpuOutput | 79.758 | [74.138, 91.051] | 0 / 33177600 |
| 3840x2160 | hot | CpuHost | 47.873 | [37.678, 63.169] | 0 / 0 |
| 3840x2160 | hot | GpuResident | 7.066 | [6.624, 7.233] | 0 / 0 |
| 3840x2160 | hot | GpuReadback | 43.287 | [35.831, 48.262] | 0 / 33177600 |
| 3840x2160 | hot | UploadRoundTrip | 71.244 | [62.039, 91.809] | 33177600 / 33177600 |
| 3840x2160 | hot | GpuInputCpuReturnGpu | 100.194 | [87.489, 137.035] | 33177600 / 33177600 |
| 3840x2160 | hot | GpuInputCpuOutput | 85.971 | [76.153, 120.473] | 0 / 33177600 |

Warmed private scratch/bin allocation deltas were zero for all four fixtures.
Per-call global scratch remains 83,456 bytes at VGA and 161,536 bytes at 4K;
retention reached 179,456 bytes with both geometry sizes present. Texture pool
allocation counts include setup/placement outputs: 4/4/19/33, with discarded
counts 0/0/12/26. Cached 4K textures occupied 38,092,800 bytes in five resources
within the 64 MiB budget. This does not claim zero texture allocation for all
placements. Explicit profiling collected 48 bytes per separate three-pass sample.

## Evidence and reproducibility

All artifacts below are under
`C:\Users\rsasa\Workspace\SpatialRust\target\automation\`:

- `gpu-clahe-scan-all-tests.log`: full native suite, 107 passing tests.
- `gpu-clahe-scan-proof.log`: first isolated million-bin proof; the final suite
  includes the subsequently expanded independent clip/count combination.
- `gpu-clahe-scan-clippy.log` and `gpu-clahe-scan-profiles.log`: feature/lint checks.
- `gpu-clahe-scan-paired.log`: final isolated paired measurements and CPU checks.
- `gpu-clahe-scan-paired-estimates.json`: all 40 pairs per fixture, ranges,
  medians and deterministic bootstrap intervals.
- `summarize-clahe-scan.py` and `gpu-clahe-scan-paired-table.md`: aggregation.
- `gpu-clahe-scan-paired-overlap-discarded.log`: discarded initial run.
- `gpu-clahe-baseline.exe`, `gpu-clahe-baseline-estimates.json` and
  `gpu-clahe-baseline-interleaved.log`: preserved previous CLAHE artifacts before
  rebuilding; historical values are not used for the paired speedup claim.
- `gpu-clahe-scan-completed-estimates.json`: all 24 completed placement medians,
  ranges and timed transfer bytes.
- `gpu-clahe-scan-interleaved.log`: completed image-placement run after the paired
  diagnostic, using the existing CPU/reference and GPU image fixture.

Commands (PowerShell, with Rust on PATH):

```powershell
cargo test -p spatialrust-gpu --all-features
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo test -p spatialrust-gpu --all-features --release --lib paired_serial_parallel_clahe -- --ignored --nocapture --test-threads=1
python C:\Users\rsasa\Workspace\SpatialRust\target\automation\summarize-clahe-scan.py
$env:RAYON_NUM_THREADS = '6'
cargo run -p spatialrust-gpu --release --features gpu-image --example gpu_clahe
```

## Delivery and next work

At this slice's survey, PR #130 was open on main and PR #131 was open on #130's
branch, with their existing frozen-source CI/merge guards active. This slice
stacks on #131 without changing either source. Delivery checks the frozen source
and parent heads, all 114 named successes, no pending/failed checks and a clean
merge before retargeting/merging in dependency order. User-modified PR prose and
local main are preserved by the existing guard conditions.

The remaining contrast dispatch work is per-tile histogram contention, padded
pixel traversal and possible geometry reuse. No automatic CPU/GPU placement or
dispatch heuristic is inferred from these single-adapter pass measurements.

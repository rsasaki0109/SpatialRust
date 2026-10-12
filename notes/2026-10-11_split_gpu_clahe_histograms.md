# Bounded GPU split-tile CLAHE histograms

Parent main is `15eaf532e04f61eb33986d97cfea91aab52b9985`, the merged
standalone point-estimator reuse slice. This slice improves resident image
CLAHE when too few large tiles expose little independent histogram work.

## Execution and bounds

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_split.rs`
selects splitting only for at most 16 tiles, each with at least 65,536 padded
pixels. Each tile has `min(ceil(area/8192),32)` contiguous chunks.
Selection checks partial index arithmetic and device buffer/storage binding
limits before allocation. Insufficient partial capacity selects the original
direct path without rejecting an otherwise supported call. Original input,
coordinate, grid, u32 tile-area, dispatch and 4 KiB workgroup limits remain.

The partial histogram pass assigns one independent workgroup per chunk, with
at most 512 groups; reduced device dispatch limits stride through all chunks.
Each group clears 256 local bins, covers its disjoint pixel range and writes
all 256 counts. Division uses quotient/remainder instead of an overflowing
`area+chunks-1`. Offset and group increments use subtraction guards. Reflect101
matches the original padded image domain, including one-pixel axes.

The next lookup pass merges chunk counts in u32: every nonnegative partial sum
is bounded by the validated tile area. It uses the original clipping/excess
tree scan, analytic residual positions and exact CPU-compatible lookup
rounding. Axis geometry, interpolation, metadata and the direct shader remain
unchanged. Split pipelines initialize lazily inside the owning runtime's
existing CLAHE family cache; unrelated runtimes and moving the runtime retain
the established ownership behavior.

Additional private storage is `tiles*chunks*1024` bytes, at most 512 KiB.
The existing submitted scratch leases hold geometry, lookups and partials
until application is submitted, then recycle them with queue ordering.
Retained buffers follow the existing byte budget and per-size cache bounds;
live leases are outside that retention budget. Every call rewrites all partial
counts, including empty ranges in the forced diagnostic. No image content,
histogram or geometry is retained as a reusable result.

Direct execution uses geometry/lookup/application in one submission. Split
execution uses geometry/histogram/merged lookup/application in one submission.
The resident image still performs no payload readback or implicit host wait.
Receipts add `clahe_histogram_gpu` and the partial write bytes on split calls.
Ordinary and split parameter buffers are each 48 bytes; image payload upload
and download accounting remains the existing explicit image contract.

`clahe_gpu_profiled` now reserves three or four free stages before allocation.
Explicit query collection returns 48 or 64 bytes respectively. Failed capacity
checks preserve prior recording; errors inside execution invalidate collection
as before. Public signatures and dependencies are unchanged. The adaptive GPU
APIs remain provisional; callers allocating exactly three stages for all CLAHE
grids should reserve four for split-capable work.

## Correctness and ownership

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_split_tests.rs`
checks 4,030,464 raw partial bins across 144 forced split layouts. Sizes include
empty chunk ranges, 255/256/257 lane boundaries, small/large tile areas,
one-pixel axes and 1,025x1,027 images. Chunk counts 2/7/32 and one dispatch group
force non-divisible ranges, repeated bin clearing and complete output writes.
The CPU oracle iterates each range serially and reflects signed coordinates
repeatedly, independently of shader modulo and workgroup traversal. Dirty
output counters start at u32::MAX so missing zero writes are observable.

Another proof compares 73,728 count/cumulative/LUT words over 96 cases,
including u32::MAX total mass, concentrated/spread/random partials and clipping
limits zero/one/area-based/full mass. Existing independent clipping, binary64
and interpolation proofs remain active.

Mixed direct/split cold calls, moved and independent runtimes, lazy cache
initialization, warmed scratch reuse, retained concurrent outputs and zero
retention budgets are checked against CPU image bytes and receipts.
The pure planner checks area and 16-tile boundaries, 8/9/32 chunk selection,
maximum u32 tile area, one-group dispatch and exact partial binding limits.
The existing adapter-free GPU-image CI command executes these planner checks.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/image_clahe.rs`
checks the new four-pass capacity, prior-recording preservation, 64-byte query
readback and CPU/plain/profiled equality. Existing strided ROI and RGB-gray
chains continue checking byte/metadata and transfer accounting.

## Measurement method

Native release uses Rust/Cargo 1.98.1, wgpu 24.0.5 and the local Vulkan discrete
adapter without publishing the adapter model. The diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_split_benchmark.rs`.

The baseline lookup is frozen from the parent in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/clahe_lookup_single_histogram.wgsl`.
Both variants use the same resident source and execution helper; the baseline
forces the original three-pass shader while the current path follows production
selection. Geometry/application shaders and host execution are shared. This
controls the histogram strategy comparison, rather than claiming two literal
historical host APIs. Runtime/shader creation and input upload occur before
timing. Each condition uses four warmups and twenty alternating old/new pairs.

Histogram time sums histogram and merged-lookup passes for split calls, or
the original lookup pass for direct calls. Total GPU time sums all three/four
passes. Completed resident latency spans the operation call through device
idle, including validation, uniforms/bind groups, output/scratch acquisition,
command construction and submission. Profiler allocation, query collection,
payload readback, byte verification and recycling are outside that timer.
Every result in every round matches CPU bytes and metadata. Query readback
is checked separately at 48/64 bytes. Input/upload and output/download costs
are excluded, so these numbers do not describe host round-trip latency.

Regression GPU work finishes before the diagnostic starts. Existing user jobs
remain untouched, so timings describe local observed conditions. Ratios are
medians of paired old/current observations, which can differ from ratios of
timing medians. Intervals bootstrap paired medians 10,000 times with seed
20261011 and 2.5/97.5 percentiles. All raw samples and observed timing ranges
are retained locally. First-use latency, browser execution and cross-device
speedups are not measured.

An initial 16,384-area selection improved histogram GPU time but regressed some
completed small calls. The final 65,536-area floor amortizes the additional
pass/setup; grids above 16 tiles retain direct execution because pilot timing
showed sufficient existing parallelism.

All 48 final conditions completed with 20 positive paired samples per metric.
The 28 selected split conditions had histogram paired medians 2.212–21.750
and completed resident paired medians 1.048–10.156. At 4K with a 1x1
tile grid, completed ratios were 9.205–10.156. One selected completed
interval (256x256, mixed, clip=2) overlaps one; local medians are not universal
latency guarantees. The 20 direct conditions retain the original shader and
mostly show ratios near one; small completed-time differences are retained
in the full table rather than called an acceleration.

| Image | Grid | Distribution | Clip | Chunks | Histogram previous/current (us), ratio [95% CI] | GPU total ratio [95% CI] | Completed previous/current (ms), ratio [95% CI] |
|---|---|---|---:|---:|---|---|---|
| 255x257 | 1x1 | mixed | 0 | 1 | 84.304/84.432, 0.999 [0.997,1.001] | 1.002 [0.997,1.005] | 0.461/0.462, 1.012 [0.973,1.053] |
| 255x257 | 1x1 | mixed | 2 | 1 | 84.256/84.128, 1.000 [0.996,1.006] | 1.001 [0.998,1.006] | 0.446/0.466, 0.963 [0.907,1.062] |
| 255x257 | 1x1 | hot | 0 | 1 | 89.248/89.120, 1.002 [1.000,1.005] | 1.003 [1.000,1.006] | 0.440/0.447, 1.004 [0.943,1.083] |
| 255x257 | 1x1 | hot | 2 | 1 | 88.944/89.248, 1.000 [0.995,1.002] | 0.999 [0.996,1.001] | 0.386/0.408, 0.955 [0.922,1.039] |
| 256x256 | 1x1 | mixed | 0 | 8 | 81.424/23.456, 3.480 [3.421,3.498] | 2.508 [2.485,2.525] | 0.422/0.419, 1.048 [1.008,1.090] |
| 256x256 | 1x1 | mixed | 2 | 8 | 81.600/23.312, 3.501 [3.466,3.519] | 2.529 [2.518,2.545] | 0.407/0.373, 1.069 [0.980,1.149] |
| 256x256 | 1x1 | hot | 0 | 8 | 86.640/24.480, 3.542 [3.520,3.553] | 2.601 [2.590,2.621] | 0.383/0.382, 1.112 [1.044,1.123] |
| 256x256 | 1x1 | hot | 2 | 8 | 86.624/24.288, 3.568 [3.536,3.614] | 2.631 [2.605,2.642] | 0.428/0.381, 1.093 [1.055,1.178] |
| 257x257 | 1x1 | mixed | 0 | 9 | 94.352/24.816, 3.796 [3.766,3.841] | 2.655 [2.637,2.669] | 0.435/0.393, 1.122 [1.047,1.214] |
| 257x257 | 1x1 | mixed | 2 | 9 | 94.336/24.800, 3.807 [3.795,3.857] | 2.663 [2.648,2.680] | 0.405/0.368, 1.097 [1.033,1.137] |
| 257x257 | 1x1 | hot | 0 | 9 | 98.688/25.392, 3.886 [3.876,3.962] | 2.723 [2.704,2.739] | 0.464/0.445, 1.087 [1.028,1.150] |
| 257x257 | 1x1 | hot | 2 | 9 | 98.752/25.440, 3.892 [3.873,3.931] | 2.721 [2.712,2.735] | 0.419/0.371, 1.128 [1.037,1.204] |
| 257x259 | 1x1 | mixed | 0 | 9 | 94.528/24.672, 3.829 [3.787,3.904] | 2.675 [2.656,2.689] | 0.437/0.404, 1.128 [1.051,1.179] |
| 257x259 | 1x1 | mixed | 2 | 9 | 94.528/24.944, 3.793 [3.784,3.814] | 2.656 [2.635,2.679] | 0.401/0.358, 1.159 [1.076,1.186] |
| 257x259 | 1x1 | hot | 0 | 9 | 98.912/25.264, 3.922 [3.906,4.001] | 2.725 [2.712,2.752] | 0.453/0.377, 1.136 [1.085,1.207] |
| 257x259 | 1x1 | hot | 2 | 9 | 98.912/25.328, 3.910 [3.883,3.923] | 2.725 [2.711,2.758] | 0.402/0.352, 1.137 [1.081,1.249] |
| 257x259 | 3x5 | mixed | 0 | 1 | 16.000/15.984, 1.003 [0.990,1.009] | 0.995 [0.983,1.011] | 0.368/0.368, 0.994 [0.928,1.096] |
| 257x259 | 3x5 | mixed | 2 | 1 | 16.352/16.496, 0.989 [0.984,0.994] | 0.995 [0.987,1.005] | 0.380/0.385, 1.008 [0.974,1.043] |
| 257x259 | 3x5 | hot | 0 | 1 | 46.976/47.024, 0.992 [0.990,1.001] | 0.996 [0.987,1.000] | 0.636/0.634, 0.972 [0.943,1.029] |
| 257x259 | 3x5 | hot | 2 | 1 | 48.080/48.176, 0.998 [0.991,1.009] | 0.998 [0.995,1.002] | 0.588/0.584, 1.017 [0.972,1.040] |
| 640x480 | 1x1 | mixed | 0 | 32 | 2607.152/150.192, 17.357 [17.278,17.496] | 10.041 [10.003,10.104] | 3.557/1.145, 3.114 [3.023,3.201] |
| 640x480 | 1x1 | mixed | 2 | 32 | 2632.432/151.696, 17.351 [17.185,17.457] | 10.018 [9.974,10.075] | 3.575/1.177, 3.088 [3.032,3.180] |
| 640x480 | 1x1 | hot | 0 | 32 | 2711.552/148.928, 18.201 [18.094,18.243] | 10.452 [10.423,10.482] | 3.674/1.176, 3.113 [3.048,3.190] |
| 640x480 | 1x1 | hot | 2 | 32 | 2745.664/152.000, 18.060 [17.998,18.186] | 10.390 [10.354,10.425] | 3.700/1.156, 3.207 [3.096,3.274] |
| 640x480 | 8x8 | mixed | 0 | 1 | 108.912/108.912, 0.999 [0.993,1.004] | 1.000 [0.999,1.001] | 1.444/1.442, 0.998 [0.977,1.018] |
| 640x480 | 8x8 | mixed | 2 | 1 | 110.096/110.624, 0.999 [0.994,1.002] | 0.999 [0.998,1.000] | 1.486/1.479, 1.008 [0.985,1.038] |
| 640x480 | 8x8 | hot | 0 | 1 | 109.232/109.328, 1.000 [0.997,1.004] | 1.000 [0.995,1.002] | 1.074/1.071, 0.991 [0.979,1.043] |
| 640x480 | 8x8 | hot | 2 | 1 | 113.456/113.712, 0.998 [0.991,1.008] | 0.998 [0.996,1.000] | 1.041/1.015, 1.012 [0.991,1.038] |
| 3840x2160 | 1x1 | mixed | 0 | 32 | 13520.688/647.456, 20.882 [20.864,20.888] | 13.829 [13.821,13.840] | 14.464/1.574, 9.205 [9.110,9.374] |
| 3840x2160 | 1x1 | mixed | 2 | 32 | 16130.784/754.608, 21.465 [21.397,21.527] | 14.013 [13.993,14.053] | 17.251/1.888, 9.691 [9.450,9.755] |
| 3840x2160 | 1x1 | hot | 0 | 32 | 16541.488/762.816, 21.750 [21.691,21.775] | 14.855 [14.826,14.873] | 17.549/1.749, 10.156 [9.919,10.312] |
| 3840x2160 | 1x1 | hot | 2 | 32 | 16541.776/761.088, 21.734 [21.708,21.761] | 14.856 [14.845,14.878] | 17.549/1.751, 10.063 [9.911,10.286] |
| 3840x2160 | 2x2 | mixed | 0 | 32 | 9660.640/1876.032, 5.153 [5.096,5.256] | 3.137 [3.108,3.196] | 12.204/4.582, 2.687 [2.650,2.719] |
| 3840x2160 | 2x2 | mixed | 2 | 32 | 10946.576/1935.536, 5.608 [5.443,5.755] | 3.390 [3.306,3.452] | 13.930/4.899, 2.823 [2.765,2.847] |
| 3840x2160 | 2x2 | hot | 0 | 32 | 10549.040/1531.120, 6.892 [6.876,6.904] | 4.568 [4.558,4.571] | 12.635/3.626, 3.507 [3.452,3.550] |
| 3840x2160 | 2x2 | hot | 2 | 32 | 11035.856/1548.304, 7.125 [7.110,7.146] | 4.710 [4.701,4.724] | 13.173/3.693, 3.579 [3.504,3.624] |
| 3840x2160 | 4x4 | mixed | 0 | 32 | 4942.816/2138.208, 2.362 [2.351,3.076] | 1.362 [1.360,1.530] | 12.534/9.757, 1.295 [1.277,1.349] |
| 3840x2160 | 4x4 | mixed | 2 | 32 | 4720.512/2132.288, 2.212 [2.192,2.256] | 1.337 [1.330,1.343] | 11.848/9.327, 1.279 [1.266,1.287] |
| 3840x2160 | 4x4 | hot | 0 | 32 | 4718.000/1542.320, 3.078 [3.065,3.091] | 1.748 [1.745,1.755] | 8.993/5.885, 1.531 [1.516,1.548] |
| 3840x2160 | 4x4 | hot | 2 | 32 | 4960.240/1577.680, 3.156 [3.148,3.176] | 1.787 [1.779,1.796] | 9.220/5.929, 1.559 [1.545,1.577] |
| 3840x2160 | 8x8 | mixed | 0 | 1 | 1503.888/1506.592, 0.999 [0.996,1.002] | 1.000 [0.999,1.001] | 13.137/13.162, 0.992 [0.986,0.999] |
| 3840x2160 | 8x8 | mixed | 2 | 1 | 1461.824/1458.016, 1.001 [0.999,1.004] | 1.000 [0.999,1.001] | 10.413/10.480, 0.998 [0.977,1.003] |
| 3840x2160 | 8x8 | hot | 0 | 1 | 1462.720/1463.680, 0.999 [0.998,1.000] | 1.001 [0.999,1.001] | 5.868/5.825, 0.999 [0.990,1.015] |
| 3840x2160 | 8x8 | hot | 2 | 1 | 1464.752/1465.024, 0.999 [0.998,1.001] | 1.000 [1.000,1.001] | 5.851/5.790, 1.011 [1.002,1.019] |
| 3840x2160 | 32x32 | mixed | 0 | 1 | 1651.344/1645.920, 0.998 [0.995,1.002] | 1.000 [1.000,1.002] | 11.925/11.825, 1.005 [0.998,1.017] |
| 3840x2160 | 32x32 | mixed | 2 | 1 | 871.280/871.200, 0.999 [0.997,1.003] | 1.000 [0.999,1.001] | 9.720/9.742, 1.000 [0.996,1.002] |
| 3840x2160 | 32x32 | hot | 0 | 1 | 838.704/837.904, 0.999 [0.998,1.001] | 1.000 [0.998,1.001] | 3.049/3.030, 1.004 [0.988,1.021] |
| 3840x2160 | 32x32 | hot | 2 | 1 | 1547.584/1547.248, 0.999 [0.998,1.001] | 1.001 [0.998,1.002] | 5.882/5.880, 0.996 [0.988,1.011] |


## Validation and reproduction

136 native tests pass: 109 unit, 4 CDF, 6 CLAHE, 5 equalization, 3 histogram and
9 shared-timing tests. Eight diagnostics are explicitly ignored in normal
runs. Native all-feature/all-target and default Clippy, minimal image and
combined image/AoSoA WASM library Clippy pass with `-D warnings`.
WASM validation compiles only; browser shader execution remains unverified.

Run from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1
cargo test -p spatialrust-gpu --features gpu-image clahe_plan_bounds_grid_padding_buffers_and_device_dispatch --lib
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-wgpu,gpu-image --target wasm32-unknown-unknown --lib -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo test -p spatialrust-gpu --features gpu-image --release --lib paired_direct_split_clahe_completed_operations -- --ignored --nocapture --test-threads=1
cargo fmt --all -- --check
git diff --check
```

Receipts under `C:/Users/rsasa/Workspace/SpatialRust/target/automation` are
ignored: `gpu-clahe-split-regressions.log`, `gpu-clahe-split-clippy.log`,
`gpu-clahe-split-default-clippy.log`, `gpu-clahe-split-wasm-minimal.log`,
`gpu-clahe-split-wasm-combined.log`, `gpu-clahe-split-ci-pure-plan.log`,
`gpu-clahe-split-paired.log`, `gpu-clahe-split-paired-estimates.json`,
`gpu-clahe-split-paired-table.md` and `summarize-clahe-split.py`.

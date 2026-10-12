# GPU voxel radix prefix reuse and large-input specialization

Date: 2026-10-11. Workspace:
`C:\Users\rsasa\Workspace\SpatialRust`.
Branch: `codex/parallel-gpu-voxel-radix-prefix`.
Frozen parent: PR #132 source `5345011408d3982e3930c0b576a07d7ae9c40952`.
The user's continuing image/point GPU and CPU parallelization request,
`C:\Users\rsasa\Workspace\SpatialRust\AGENTS.md` and
`C:\Users\rsasa\Workspace\SpatialRust\.cursor\rules\long-horizon.mdc`
authorize implementation, verification and dependency-ordered guarded delivery.

## Algorithm and ownership

The shader at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\shaders\voxel_radix.wgsl`
previously recomputed each digit's global base by loading every earlier digit's
histogram in the prefix pass. For large inputs, the histogram workgroup now
publishes its 16 digit counts in shared memory. One additional barrier makes
them visible; the first 16 lanes sum local earlier-digit counts and write those
subtotals into the existing offset buffer. No global atomic ordering is used.

The ordered prefix pass reads `(own digit count, earlier-digit subtotal)` for
each group in its lane's contiguous chunk. A work-efficient exclusive tree scan
of 256 chunk totals yields both the global digit base and local chunk prefix.
Each digit's workgroup reads only its own buffer slice. All lanes finish reading
at the initial workgroup barrier before any output overwrite, so there is no
cross-workgroup read/write race. The final loop writes scatter offsets in place.
Both components and final offsets stay at most the validated point count.

For G input workgroups, the source-level prefix counter reads decrease from
152G to 48G per digit, including the final offset-writing loop. The histogram
adds 16G offset writes. Combined histogram/prefix scalar counter accesses
(reads plus writes) decrease from 184G to 96G, excluding entries, ranks and
scatter traffic. These are source-level counts; physical traffic depends on
GPU caches and is not established by hardware bandwidth counters.

The initial always-optimized version regressed at 257 and 65,537 points despite
improving million-point sorts. The final implementation specializes the legacy
histogram/prefix through 2,048 input workgroups (524,288 points), retaining at
most eight counts per prefix scan lane without the new histogram barrier.
Larger inputs use the cached-subtotal/tree path. This is an internal GPU
selection; explicit CPU/GPU placement and Auto admission policies are unchanged.

At
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\pipeline_cache.rs`,
the legacy pipelines compile with the default false `cached_prefix` override.
Two optimized histogram/prefix pipelines compile with that override true and
are initialized lazily once per runtime via `OnceLock`; scatter is shared.
This avoids a per-invocation shader branch on the small path and avoids creating
the optimized pair for runtimes that only process small point clouds.

Stable lane-mask ranks, signed i32 XYZ ordering, increasing original indices for
equal keys, 24 digits/72 passes, uniform/binding layouts and dispatch limits are
retained. Private scratch remains `20*N + 128*ceil(N/256)` bytes (ping-pong
entries, ranks, histograms and offsets), leased through submission and then
recycled. Caller-owned entries/segments are separate. There is no new production
payload transfer, clear, GPU copy, wait, dependency or unsafe code. The existing
explicit four-byte compaction cell-count synchronization remains.

## Verification

Four new native proofs at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\voxel_radix_tests.rs`
cover:

- Exact intermediate histogram counts, earlier-digit subtotals and stable
  local ranks, inspecting the optimized specialization directly.
- 2,429,504 exclusive offsets across 32 cases, with G=1/3/255/256/257/513/3907/
  32769, four sparse/dense distributions and total mass `u32::MAX`. Independent
  serial u64 CPU sums produce the reference. These synthetic counters exercise
  prefix arithmetic without allocating billions of points or changing public
  point/device limits.
- Direct GPU entry order for mixed full-width signed keys, all-equal hot keys
  and reverse/repeated keys at 257, 65,537, 524,288, 524,289 and 1,000,003 points.
  CPU reference sorts the tuple `(signed X, signed Y, signed Z, original index)`.
  No host per-cell sorting can mask GPU instability. Outputs survive subsequent
  sorts; a warmed million-point repeat makes zero additional scratch allocations.
- Four simultaneous sorts race the optimized pair's first initialization on
  one runtime, share the same pipeline object addresses and independently match
  CPU order. The forced specialization in this test is private/test-only.

The first isolated synthetic-prefix fixture incorrectly reused one dummy buffer
for both read/write bindings and was rejected by wgpu. Separate dummy buffers
fixed the fixture; production buffers were distinct throughout. The failed log
is retained, and all final proofs pass.

The final native all-feature GPU suite passed 111 tests: 87 unit, four CDF,
five CLAHE, five equalization, three histogram and seven shared image/point timing
tests. Two performance diagnostics remain ignored by default. Filtering's GPU
voxel profile passed 18 tests; GPU Euclidean segmentation passed ten; facade MVP
normal/RGB voxel parity passed two. Total related regression tests: 141.

Final all-feature/default all-target Clippy with warnings denied, minimal
`gpu-wgpu` test compilation, minimal wgpu and combined image/AoSoA WASM library
Clippy, and the minimal facade wgpu check passed. Formatting and whitespace
checks passed. The preserved test shader matches the frozen parent source with
normalized newlines. WASM compilation does not establish browser shader execution.

## Paired measurement method

The explicit ignored diagnostic at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\voxel_radix_benchmark.rs`
compares the final selected specialization with the frozen parent's shader at
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\tests\fixtures\voxel_radix_previous.wgsl`.
The preserved shader is test-only, not a production alternative API. Both use the
same runtime/device, keys, uniforms, layouts, dispatch and separate equal-sized
working buffers. Each fixture has three warmup pairs and 20 measured pairs,
alternating which implementation runs first.

Each variant starts from an identical resident seed, copied into its input
before all 72 passes. GPU timestamps exclude this reset, host setup, inter-pass
gaps and collection. The completed wall interval starts after command recording
and includes seed-to-input device copy, queue submission, all passes and wait.
The reset copies `16*N` bytes for both variants. These are pre-recorded resident
reset/sort measurements; they exclude public entry construction, allocation,
binding/uniform creation, segment compaction, CPU uploads and output readback.
They do not measure complete centroid/frame/API latency.

Each pair explicitly collects 144 stages/2,304 query bytes after both wall timers.
Final outputs from both implementations are fully compared with the independent
CPU tuple sort after measurement. Five point counts (including both sides of
the cutoff) times three distributions give 15 fixtures. Measurements use the local GPU,
Vulkan, NVIDIA 596.36; Rust 1.98.1/wgpu 24.0.5 on the i7-9750H Windows workspace.
Other agent-started GPU checks finished before the diagnostic. The user's Python
workloads were left running; desktop load and GPU clocks can still vary.

The table reports previous/current medians and median paired previous/current
ratios. GPU histogram/prefix/total columns are microseconds summed across the
24 digits; completed intervals are milliseconds. Brackets are percentile 95%
bootstrap intervals of paired-ratio medians, 10,000 resamples, seed 20261011.
These intervals characterize the local samples, not a cross-device guarantee.
Scatter, raw observations, observed ranges and reset bytes are retained in JSON.

| Points | Pattern | Histogram us: old/new; paired ratio [95%] | Prefix us: old/new; paired ratio [95%] | GPU total us: old/new; paired ratio [95%] | Completed ms: old/new; paired ratio [95%] |
|---:|---|---|---|---|---|
| 257 | mixed | 81.232 / 82.960; 0.973 [0.967, 0.988] | 114.336 / 115.520; 0.989 [0.986, 1.003] | 277.344 / 281.328; 0.985 [0.981, 0.990] | 0.910 / 0.897; 0.988 [0.962, 1.047] |
| 257 | hot | 81.776 / 83.856; 0.970 [0.960, 0.979] | 113.600 / 115.168; 0.990 [0.977, 0.998] | 270.592 / 277.168; 0.977 [0.966, 0.985] | 0.872 / 0.884; 0.992 [0.977, 1.030] |
| 257 | reverse | 82.224 / 83.120; 0.986 [0.973, 1.001] | 113.968 / 115.184; 0.990 [0.980, 0.998] | 273.424 / 278.240; 0.981 [0.975, 0.994] | 0.859 / 0.858; 0.998 [0.982, 1.015] |
| 65537 | mixed | 249.136 / 248.736; 1.001 [0.997, 1.003] | 146.544 / 146.448; 1.002 [0.997, 1.009] | 915.072 / 915.536; 0.998 [0.995, 1.001] | 1.509 / 1.525; 0.995 [0.986, 1.001] |
| 65537 | hot | 224.304 / 223.440; 1.004 [0.996, 1.007] | 146.080 / 147.200; 0.994 [0.986, 1.005] | 628.528 / 626.704; 1.003 [0.998, 1.005] | 1.239 / 1.237; 1.005 [0.986, 1.009] |
| 65537 | reverse | 226.448 / 227.248; 1.000 [0.995, 1.003] | 146.512 / 146.416; 1.001 [0.992, 1.005] | 688.480 / 686.000; 1.004 [1.002, 1.005] | 1.306 / 1.301; 1.000 [0.989, 1.012] |
| 524288 | mixed | 1146.304 / 1146.112; 0.999 [0.999, 1.001] | 660.544 / 663.072; 0.997 [0.994, 0.999] | 4567.008 / 4565.152; 1.000 [1.000, 1.001] | 5.211 / 5.222; 1.000 [0.996, 1.006] |
| 524288 | hot | 1091.328 / 1091.584; 1.001 [0.999, 1.001] | 666.080 / 668.320; 0.997 [0.995, 0.998] | 3548.320 / 3548.240; 1.000 [1.000, 1.001] | 4.211 / 4.219; 0.998 [0.993, 1.002] |
| 524288 | reverse | 1094.720 / 1092.928; 1.001 [0.999, 1.003] | 664.896 / 667.184; 0.996 [0.994, 0.998] | 3757.792 / 3753.312; 1.001 [1.000, 1.002] | 4.406 / 4.417; 0.999 [0.994, 1.003] |
| 524289 | mixed | 1127.488 / 1272.096; 0.887 [0.885, 0.888] | 617.232 / 218.240; 2.827 [2.814, 2.836] | 4515.680 / 4271.056; 1.058 [1.056, 1.059] | 5.173 / 4.929; 1.048 [1.042, 1.054] |
| 524289 | hot | 1089.248 / 1136.336; 0.958 [0.957, 0.960] | 617.680 / 218.448; 2.829 [2.825, 2.839] | 3495.232 / 3147.968; 1.110 [1.110, 1.111] | 4.143 / 3.783; 1.096 [1.092, 1.098] |
| 524289 | reverse | 1091.824 / 1155.904; 0.943 [0.943, 0.945] | 617.440 / 218.176; 2.833 [2.822, 2.841] | 3708.640 / 3375.632; 1.098 [1.097, 1.099] | 4.360 / 4.021; 1.087 [1.080, 1.093] |
| 1000003 | mixed | 2054.656 / 2351.424; 0.874 [0.873, 0.876] | 1239.104 / 413.104; 3.001 [2.995, 3.008] | 8489.968 / 7970.144; 1.065 [1.065, 1.066] | 9.230 / 8.682; 1.063 [1.059, 1.066] |
| 1000003 | hot | 1953.280 / 2037.920; 0.958 [0.958, 0.959] | 1238.480 / 411.968; 3.005 [2.991, 3.015] | 6572.928 / 5834.032; 1.127 [1.126, 1.127] | 7.280 / 6.544; 1.112 [1.108, 1.116] |
| 1000003 | reverse | 1961.888 / 2087.552; 0.939 [0.939, 0.940] | 1237.888 / 413.104; 2.998 [2.993, 2.998] | 7033.120 / 6344.160; 1.109 [1.108, 1.109] | 7.749 / 7.041; 1.100 [1.095, 1.104] |

The final million-point prefix is about 3.0x faster across all three patterns.
Histogram overhead remains visible (roughly 4–14% slower), while the combined
GPU sort intervals improve 1.065x/1.127x/1.109x (mixed/hot/reverse). Completed
reset/sort paired ratios are 1.063x/1.112x/1.100x. Just above the cutoff the GPU
ratios are 1.058x/1.110x/1.098x. The 65,537-point and cutoff legacy GPU totals
remain near parity. At 257 points, GPU totals are about 1.5–2.3% slower even with
the legacy specialization, while completed intervals' confidence intervals
include parity. The final path removes the initial extra-barrier regression;
it does not establish identical runtime or universal no-regression performance.
Median absolute times and median paired ratios are different statistics and
need not divide to the same ratio when clocks/load change during a fixture.

## Evidence and reproduction

Artifacts are under
`C:\Users\rsasa\Workspace\SpatialRust\target\automation\`:

- `voxel-radix-prefix-regressions.log`: final 141 related passing tests.
- `voxel-radix-prefix-specialized-proof.log`: isolated initial specialization
  proof before adding the cold-cache concurrency test; final suite includes it.
- `voxel-radix-prefix-final-profiles.log`: final feature/lint checks.
- `voxel-radix-prefix-paired.log`, `voxel-radix-prefix-estimates.json`,
  `voxel-radix-prefix-table.md` and `summarize-voxel-radix-prefix.py`: final paired
  raw samples, statistics and aggregation.
- `voxel-radix-prefix-always-cached-rejected.log`,
  `voxel-radix-prefix-always-cached-rejected-estimates.json` and
  `voxel-radix-prefix-always-cached-rejected-table.md`: rejected always-optimized
  measurements, retaining small-input regressions rather than hiding them.
- `voxel-radix-prefix-proof-fixture-alias-failure.log`: initial fixture error.
- `voxel-radix-prefix-baseline.wgsl`: parent's shader before edits.

```powershell
cargo test -p spatialrust-gpu --all-features
cargo test -p spatialrust-filtering --features filter-voxel-gpu
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline gpu_voxel_matches_cpu -- --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib paired_previous_current_voxel_radix -- --ignored --nocapture --test-threads=1
python C:\Users\rsasa\Workspace\SpatialRust\target\automation\summarize-voxel-radix-prefix.py
```

## Delivery and remaining work

At survey, PR #130 had merged after 114 successes; #131 subsequently merged
after 114 successes during this work. PR #132 remains the frozen parent of this
slice. Its existing guard retargeted it to main after #131 merged. New delivery
retains exact source/parent checks, all 114 named CI successes and no pending or
failed checks before dependency-ordered retarget/merge. User-modified PR prose
and local main are preserved by guard conditions.

Remaining point performance work includes launch/setup overhead, scatter and
per-distribution rank traffic; image histogram/padded traversal tuning remains
separate. This receipt makes no cross-device universal speedup or new Auto
CPU/GPU crossover claim.

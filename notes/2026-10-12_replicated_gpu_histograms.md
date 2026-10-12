# Bounded replicated GPU histogram counters

## Implementation and selection

Resident channel histograms previously used one workgroup-local array of 256
atomic counters. Concentrated values can make lanes update the same counters.
The selected path uses eight arrays with a 257-word pitch, assigning a lane to
replica `lane & 7`. Each lane initializes its bin in all eight replicas, counts
its original grid-stride pixels, then sums that bin across replicas after a
workgroup barrier. Padding words are never read. Used counters and every global
partial/output bin are explicitly overwritten.

Production selects replicas only for **4,194,304..=8,388,608 pixels** when the
device supports at least **8,224 workgroup bytes**. Smaller/larger inputs and
lower shared-memory limits retain the original shader. A native prototype survey
found small-input overhead; public-call measurements at 16,777,216 pixels did
not establish stable improvement. These bounds are a measured selection policy,
not an optimality or portable speedup guarantee.

The original two pipelines remain. One additional partial pipeline initializes
lazily per runtime only when selected. The original 16-byte uniform, binding
layout, group count (at most 256), final reduction, private global partials (at
most 256 KiB) and independently owned 1 KiB output remain unchanged. There are
still two passes in one submission. Logical channels, exact u32 counts, source
transfer history, physical stage-write receipts, 1 KiB explicit bin readback and
32-byte timestamp collection are preserved. Local replication adds no global
scratch or payload transfer. No public signature, dependency or unsafe code is
introduced; the feature remains `gpu-image` in `spatialrust-gpu`.

The unchanged index/stride wrap guard visits the same pixels. Each pixel belongs
to exactly one lane/replica/group. Nonnegative counts and all intermediate sums
are bounded by the accepted global pixel count, at most u32::MAX. Integer atomic
ordering does not affect these counts. Shared-memory selection falls back before
compiling the additional pipeline when its local-storage requirement is unmet.

Production sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram/replicated.wgsl`

## Correctness and builds

Four new native tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram_replica_tests.rs`
cover:

- **307,200 exact bin comparisons** across 300 cases: ten singleton/lane/group/
  odd-tail shapes through 1001x1003, all logical channels 1..=4, and mixed, hot
  and two-bin strided ROIs. Frozen, replicated, per-lane-run and public outputs
  each match an independent CPU pixel lookup; complete receipts also agree.
- Manufactured replica arrays exercise **u32::MAX mass for each of 256 bins**
  through the actual production `sum_replicas` function on GPU. The bins are
  independent arithmetic probes, not one legal image with that total mass.
- Exact lower/upper pixel thresholds, adjacent-column/row increments beyond either
  threshold, exact 8,224-byte storage acceptance, one-byte-under fallback, the
  original 1 KiB fallback requirement and insufficient-storage rejection.
- Small calls leave the replica cache cold. Insufficient profile capacity on
  a large image preserves scratch counters/cold cache and empty recording
  state. Four concurrent large calls initialize one shared cache and retain
  independent live results. Mixed/hot/two-bin 2048x2048 inputs match frozen
  counts; selected large RGBA inputs verify all four channels. Profiled/public
  receipts agree with frozen execution, with two passes and 32 query bytes.

The frozen shader is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/histogram_previous.wgsl`.
It matches parent source `b595e771600fd71694320a927185328e29748b57`; the original
production shader remains byte-identical. Native execution in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram_experiment_execution.rs`
copies that parent executor, changing only the function name and injecting a
prepared pipeline in place of cache lookup. Its prepared factory injects shader
source and initializes the new empty cache field. The executor and shader are
verified against the parent. Production never invokes the native executor.

Native Windows, Rust/Cargo 1.98.1 and wgpu 24: **163 image/point GPU tests pass**
(136 unit plus 27 integration), zero failures and thirteen ignored diagnostics.
The facade smoke passes. Final selection tests pass in minimal `gpu-image`.
Strict default/all-feature all-target Clippy and facade Clippy pass. GPU image,
GPU image/AoSoA and facade GPU image WASM compilation pass; browser execution is
not established. Formatting and diff checks pass. The facade source is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_histogram_replica_smoke.rs`.

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust-gpu --no-default-features --features gpu-image --lib replica_tests -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features gpu-image --test gpu_histogram_replica_smoke
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features gpu-image --test gpu_histogram_replica_smoke -- -D warnings
cargo check -p spatialrust-gpu --no-default-features --features gpu-image --target wasm32-unknown-unknown
cargo check -p spatialrust-gpu --no-default-features --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown
cargo check -p spatialrust --no-default-features --features gpu-image --target wasm32-unknown-unknown
cargo fmt --all --check
git diff --check
```

## Completed-call evidence

The ignored native diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram_replica_benchmark.rs`.
It compares a prepared frozen parent with the public profiled operation. Timers
include validation, uniforms, output/scratch acquisition, bindings, encoding,
submission and device-idle wait. Source upload, pipeline cold setup, profiler
allocation/collection, bin readback/checks, and recycling are outside measured
samples. Both sides use the same host executor and pools. Four warmups per side
precede twenty alternating paired rounds. Every output matches independent CPU
bins; every report has exactly two stages and 32 query bytes. No other tests or
builds launched by this task overlap these surveys; desktop CPU/GPU activity is
uncontrolled. Cold-start cost and end-to-end upload/readback pipelines are not
measured here.

Fixtures are grayscale: mixed values `(i * 37 + i / 257) % 256`, constant 73,
and spatial checkerboard 0/255. Results are per-variant medians. Paired ratios
are medians of twenty frozen/public wall-time ratios; values above one favor
the new path. Intervals use 10,000 paired median bootstrap resamples, seed
20261012 and sorted indices 249/9749. They describe each run, not portability
or between-run variability. Cross-run raw times are not controlled comparisons.

### Final general survey (18 conditions)

| Image | Pattern | Old/current partial us | Old/current wall ms | Paired ratio (95% CI) |
| --- | --- | ---: | ---: | ---: |
| 17x19 | mixed | 4.51/4.45 | 0.208/0.209 | 0.991 (0.947–1.064) |
| 17x19 | hot | 4.40/4.30 | 0.237/0.226 | 1.020 (0.988–1.039) |
| 17x19 | two-bin | 4.30/4.51 | 0.208/0.204 | 0.993 (0.948–1.047) |
| 255x257 | mixed | 6.70/6.62 | 0.228/0.235 | 0.993 (0.942–1.021) |
| 255x257 | hot | 6.86/6.78 | 0.235/0.227 | 1.038 (0.988–1.068) |
| 255x257 | two-bin | 7.89/7.90 | 0.226/0.218 | 1.016 (0.982–1.059) |
| 640x480 | mixed | 11.52/11.58 | 0.234/0.237 | 0.995 (0.948–1.055) |
| 640x480 | hot | 11.84/11.97 | 0.238/0.232 | 1.026 (0.980–1.062) |
| 640x480 | two-bin | 16.03/16.02 | 0.240/0.238 | 1.033 (0.988–1.063) |
| 1024x1024 | mixed | 31.42/31.26 | 0.261/0.274 | 0.979 (0.945–0.994) |
| 1024x1024 | hot | 32.67/32.69 | 0.263/0.271 | 0.998 (0.954–1.031) |
| 1024x1024 | two-bin | 42.00/42.00 | 0.270/0.277 | 0.975 (0.954–1.012) |
| 1920x1080 | mixed | 55.63/55.62 | 0.294/0.291 | 1.009 (0.982–1.026) |
| 1920x1080 | hot | 58.16/58.21 | 0.294/0.289 | 1.000 (0.971–1.033) |
| 1920x1080 | two-bin | 74.88/74.90 | 0.307/0.317 | 0.976 (0.935–1.018) |
| 3840x2160 | mixed | 203.73/202.11 | 0.434/0.437 | 1.002 (0.973–1.034) |
| 3840x2160 | hot | 215.17/202.62 | 0.452/0.434 | 1.042 (1.007–1.065) |
| 3840x2160 | two-bin | 271.15/203.06 | 0.511/0.432 | 1.175 (1.164–1.222) |

### Final large repeat (21 conditions; 4096x4096 uses the original path)

| Image | Pattern | Old/current partial us | Old/current wall ms | Paired ratio (95% CI) |
| --- | --- | ---: | ---: | ---: |
| 2048x2048 | mixed | 107.74/107.10 | 0.337/0.338 | 0.983 (0.960–1.012) |
| 2048x2048 | hot | 111.63/106.11 | 0.343/0.338 | 1.011 (0.976–1.032) |
| 2048x2048 | two-bin | 140.50/106.32 | 0.380/0.334 | 1.130 (1.122–1.148) |
| 4096x1024 | mixed | 106.88/106.59 | 0.339/0.354 | 1.009 (0.957–1.018) |
| 4096x1024 | hot | 112.11/106.80 | 0.343/0.334 | 1.011 (0.986–1.036) |
| 4096x1024 | two-bin | 144.51/107.44 | 0.374/0.337 | 1.102 (1.077–1.126) |
| 1024x4096 | mixed | 106.34/106.46 | 0.350/0.343 | 1.006 (0.978–1.043) |
| 1024x4096 | hot | 111.20/106.13 | 0.356/0.335 | 1.043 (0.999–1.051) |
| 1024x4096 | two-bin | 143.52/106.40 | 0.380/0.345 | 1.113 (1.081–1.157) |
| 8192x513 | mixed | 114.16/114.26 | 0.348/0.347 | 0.999 (0.986–1.029) |
| 8192x513 | hot | 119.68/114.11 | 0.348/0.344 | 1.015 (0.983–1.042) |
| 8192x513 | two-bin | 142.85/114.61 | 0.369/0.340 | 1.089 (1.051–1.113) |
| 513x8192 | mixed | 105.23/104.70 | 0.326/0.327 | 1.006 (0.982–1.030) |
| 513x8192 | hot | 110.29/104.72 | 0.338/0.329 | 1.028 (1.016–1.046) |
| 513x8192 | two-bin | 143.62/104.80 | 0.370/0.334 | 1.118 (1.083–1.154) |
| 8192x1024 | mixed | 226.64/224.93 | 0.496/0.487 | 1.012 (0.981–1.032) |
| 8192x1024 | hot | 238.46/224.77 | 0.494/0.483 | 1.022 (1.001–1.061) |
| 8192x1024 | two-bin | 775.02/771.50 | 1.148/1.145 | 1.005 (0.980–1.019) |
| 4096x4096 | mixed | 1499.97/1500.22 | 1.922/1.920 | 1.006 (0.975–1.021) |
| 4096x4096 | hot | 1276.03/1278.58 | 1.682/1.695 | 0.990 (0.979–1.017) |
| 4096x4096 | two-bin | 1203.09/1204.82 | 1.614/1.631 | 0.993 (0.979–0.999) |

The five selected large two-bin shapes below 8,388,608 pixels improve in both final large surveys: first paired medians 1.080–1.125, repeat 1.089–1.130, with interval lower bounds above one. The general 3840x2160 two-bin case is 1.175 (1.164–1.222). At the exact 8,388,608-pixel boundary, two-bin ratios are 1.036 (1.004–1.133) and 1.005 (0.980–1.019): repeat improvement is not established. Mixed/hot outcomes vary. The original-kernel 1024x1024 mixed control has wall ratio 0.979 (0.945–0.994) despite nearly identical partial medians 31.42/31.26 us; the final original-kernel 4096x4096 two-bin repeat is 0.993 (0.979–0.999). These adverse completed-call observations remain in the tables; preserving a kernel does not establish universal neutral call overhead. The complete first final-large table and all samples remain in private artifacts.

Earlier eight-replica sweeps (18 general and 18 large conditions) motivated
selection. A second prototype compressed runs of equal values per lane while
retaining one counter array. Its 18-condition survey improved 1920x1080 two-bin
completion by 1.070 (1.050–1.095) and 3840x2160 hot completion by 1.040
(1.032–1.074), but 3840x2160 two-bin was 0.998 (0.966–1.024). Small/medium
partial passes often slowed, including 255x257 hot 6.83 to 8.43 us and two-bin
7.89 to 9.38 us; 640x480 two-bin wall ratio was 0.943 (0.897–1.030). This
distribution/geometry dependence did not justify production run compression.
It remains solely in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/histogram_runs_experiment.wgsl`.

Initial public surveys without an upper selection bound retained adverse
4096x4096 two-bin results: 0.991 (0.971–0.999) in the first large survey, and
1.010 (1.000–1.019) on repeat. Partial times in those samples rose to roughly
1.18–1.21 ms on both sides, unlike the earlier prototype sweep's 0.534/0.402 ms.
They are not discarded or treated as stable gains. Production therefore keeps
the original path beyond 8,388,608 pixels; the final table includes that larger
fallback control. Mixed/hot cases do not imply universal neutral overhead or a
fixed speedup, and no adapter-specific selection is used.

```powershell
$env:SPATIALRUST_HISTOGRAM_VARIANT = 'public'
cargo test -p spatialrust-gpu --no-default-features --features gpu-image --release --lib paired_frozen_replicated_histograms -- --ignored --nocapture --test-threads=1
$env:SPATIALRUST_HISTOGRAM_VARIANT = 'public-large'
cargo test -p spatialrust-gpu --no-default-features --features gpu-image --release --lib paired_frozen_replicated_histograms -- --ignored --nocapture --test-threads=1
```

Empty/default variant forces eight replicas; `replicas-large` extends prototype
shapes; `runs` selects the native run prototype. Full samples/analysis for every
survey, including earlier/adverse cases, remain private ignored files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-histogram-*`.
Published conditions omit the local adapter model. All 114 unchanged named
checks, the exact source commit and the merged parent dependency gate merge.

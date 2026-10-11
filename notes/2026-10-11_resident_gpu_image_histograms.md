# Resident GPU image histograms — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Parent: `a9eacf99e4554bbb39b7d4ebad46dcb406d85008`, PR 127,
`codex/bounded-parallel-point-queries`. One named GPU image-analysis slice;
point-query, contrast and prior GPU storage implementations remain unchanged.

## Implementation and contract

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\histogram.rs`
adds `histogram_u8_gpu`, `histogram_u8_gpu_profiled` and `GpuImageHistogram` under
`gpu-image`. Select any logical channel of a 1..=4-channel resident image;
`rgb_to_gray_gpu` is an explicit preceding operation for a luma histogram.
Integer `rgba8uint` texture loads count exact byte values without float rounding.

One command submission contains two passes:

1. At most 256 workgroups, each with 256 lanes and a 1 KiB workgroup-local atomic
   histogram, visit pixels with a grid stride. Every lane initializes one bin,
   both barriers are unconditional, and each group overwrites all 256 partial
   counts, including zero bins. A guarded index increment avoids u32 wraparound.
2. One 256-lane workgroup sums each bin across groups into a live 1 KiB output.
   Summation is exact; no global per-pixel atomic contention or CPU reduction
   occurs. Each final bin is overwritten even when output storage was recycled.

Private partial storage is capped at 256 KiB and additionally bounded by device
binding/buffer/workgroup limits. Inputs require 1..=u32::MAX pixels, so every bin
and their total fit u32. Pixel count and device limits are checked before GPU
work allocation; invalid channels and foreign-runtime sources also fail first.
Pipeline families are cached per runtime. Scratch is recycled only after both
passes have been submitted, using the existing private `ScratchBuffers` lease
protocol and ordered queue execution. Live outputs are separate from scratch.

The output exposes pixel count, selected channel and cumulative image receipts.
It retains its runtime identity across moves. Explicit `recycle` returns its
buffer to the original runtime's bounded upload/output pool; foreign recycling
drops the object instead of putting it in another pool. There is no implicit
output recycling, readback or host wait during histogram construction.
Native `readback_u32` synchronizes and copies exactly 1,024 bytes, returning
`[u32; 256]`; each successful readback adds another 1,024 bytes to its receipt.
Blocking histogram readback is not exposed on browser WASM. Device computation
still compiles there; browser execution/async readback is not validated here.

Receipts inherit source uploads/stages and count global partial writes plus
1,024 final-bin bytes as GPU work. Uniform/driver metadata and on-chip local-bin
initialization follow existing image receipt conventions and are not image
payload transfers. The profiled variant requires two unused query stages on the
same runtime before allocation. Explicit native timestamp collection returns
32 query bytes separately from payload receipts. Capacity rejection preserves
prior recording; an inner recording failure invalidates collection.

These APIs are provisional. No production/default/core dependency is added.
The production dependency graph was checked to exclude Vision and Rayon;
`spatialrust-vision/imgproc-analysis` and Rayon are dev-only benchmark references.
The current result has no public raw-buffer escape hatch; device-side consumers
are the next roadmap slice. CPU and GPU histogram count types intentionally
have different maxima (CPU u64, GPU bounded u32).

## Validation

`cargo test -p spatialrust-gpu --all-features -- --test-threads=2` passed:
72 unit tests, three new public histogram integration tests and seven existing
image/point timing integration tests: 82 tests total, on the real Vulkan GPU.
Six histogram unit tests cover:

- Counts for 1/255/256/257-pixel lane boundaries, one-column input, 257x257 and
  1001x1003 grids beyond one grid stride; all-zero/all-255 and mixed distributions.
- All 1/2/3/4 logical channels of a strided 17x19 ROI, padding exclusion, exact
  cumulative readback bytes, stage/device-write receipts and invalid channels.
- Counter overflow, empty dimensions and reduced/insufficient workgroup, storage
  and buffer limits, including accepted maximum u32 arithmetic without allocating
  an impossible full-size image.
- Stale scratch/output overwrite, retained old results while new calls recycle,
  zero/shrinking pool controls, warmed reuse and runtime moves/foreign devices.
- Four concurrent callers retaining three outputs each on one runtime, verifying
  exact counts and submission-safe scratch reuse.

Public integration tests cover RGB-to-gray-to-histogram CPU parity with no
intermediate image download; exact two-pass timing capacity/query bytes; failed
recording and foreign-profiler rejection. Default GPU tests and default/full
all-target Clippy also passed. GPU-image-only tests compile without AoSoA, and
the adapter-free planner fixture runs in that profile. Rustdoc with warnings
denied, facade `gpu-image` compilation, minimal WGPU and image+AoSoA WASM library
checks with warnings denied, rustfmt, diff and parsed CI YAML checks passed.

CI adds one `spatialrust-gpu-image` profile. Hosted GPU-less runners compile GPU
tests and execute the adapter-free planner fixture; actual shader/runtime tests
were executed locally rather than claimed as hosted GPU tests. The benchmark
compile job explicitly includes `image_histogram`. There are now 114 successful
checks required by this slice's merge guard, with all required names checked.

## Completed-work measurements

Machine: Windows, Rust 1.98.1 release/thin LTO, i7-9750H and GTX 1660 Ti,
Vulkan NVIDIA 596.36. Other user Python computations stayed active throughout;
none were stopped or reprioritized. The benchmark process alone sets
`RAYON_NUM_THREADS=6`. The production CPU histogram runs scalar below one
million pixels (VGA) and uses its existing bounded row bands at 1080p/4K.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\examples\gpu_histogram.rs`
prebuilds input images and uploads outside timing except in `UploadAndBins`.
Mixed byte values are `(i*37 + i/997) % 256`; hot-bin images contain only 43.
The native example runs two warmups per mode and ten rounds rotating all five
modes in one process. Timed regions include completed work, allocation, dispatch
and the named transfers; verification, result destruction/recycling and optional
timestamp collection are outside timing. Every measured result is verified.

| Mode | Starting input | Timed output placement | Timed image/bin payload transfers |
| --- | --- | --- | --- |
| Cpu | host grayscale | host CPU u64 bins via production `histogram_u8` | none |
| Resident | uploaded texture | completed resident u32 bins; explicit `wait_idle` | none |
| BinsReadback | uploaded texture | host u32 bins | 1,024 bytes D2H |
| UploadAndBins | host grayscale | host u32 bins | physical RGBA8 upload + 1,024 bytes D2H |
| ImageReadbackCpu | uploaded texture | host CPU u64 bins | full physical RGBA8 readback + CPU histogram |

Resident mode explicitly reads 1 KiB **after** its timing window for verification;
the example/benchmark as a whole therefore does perform that untimed transfer.
A separate uploaded source is used for full-image CPU readback so its cumulative
receipt does not contaminate the resident source. GPU-only pass timestamps are
not completed-call latency and omit CPU setup, waits, transfers and gaps.
Comparisons between different input/output placements do not imply automatic
GPU placement is preferable for a host image.

All entries are median [observed min, max] in ms.

| Image / distribution | CPU | Resident | BinsReadback | UploadAndBins | ImageReadbackCpu |
| --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 0.187 [0.167, 0.228] | 0.432 [0.322, 0.506] | 1.521 [1.289, 1.813] | 2.032 [1.807, 2.481] | 2.408 [2.075, 2.736] |
| 640x480 / hot | 0.543 [0.454, 0.620] | 0.315 [0.289, 0.595] | 1.319 [1.040, 1.754] | 1.785 [1.509, 2.266] | 2.413 [2.030, 3.018] |
| 1920x1080 / mixed | 0.481 [0.436, 0.556] | 0.527 [0.458, 0.660] | 2.210 [1.831, 2.939] | 5.964 [5.175, 7.078] | 7.835 [6.772, 9.729] |
| 1920x1080 / hot | 0.797 [0.672, 0.872] | 0.545 [0.432, 0.849] | 1.998 [1.612, 3.146] | 6.070 [5.618, 11.128] | 8.639 [7.811, 10.619] |
| 3840x2160 / mixed | 1.493 [1.341, 1.695] | 0.699 [0.549, 0.753] | 5.339 [4.467, 6.048] | 18.000 [17.310, 22.052] | 26.011 [22.896, 29.865] |
| 3840x2160 / hot | 2.802 [2.490, 3.394] | 0.710 [0.647, 0.832] | 6.339 [5.298, 7.405] | 21.825 [20.198, 24.501] | 31.234 [27.085, 35.933] |

Ranges are observed min/max of ten calls, not confidence intervals. At 4K mixed,
BinsReadback versus ImageReadbackCpu medians were 5.339 versus 26.011 ms, a local
ratio of about 4.9x for the same GPU-input/host-count destination. Physical D2H
payload shrinks from 33,177,600 bytes to 1,024 (32,400x fewer bytes). Resident
GPU bins completed in 0.699 ms; CPU host-input bins took 1.493 ms, with different
starting/output placements. UploadAndBins took 18.000 ms, **slower than CPU host
execution**; VGA mixed resident GPU was also slower than CPU (0.432 vs 0.187 ms).
These unfavorable cases are retained; no universal GPU speedup is claimed.

After warmups all six profiles had zero new histogram scratch/output-buffer
allocations during rotation and retained 262,144 scratch bytes. This covers
pooled histogram buffers, not uniforms, bind groups, encoders, image textures,
mapped staging or driver allocation counts. One profiled 4K mixed call recorded
206.816 us for partials and 25.248 us for reduction, with 32 explicit query bytes.

### Criterion evidence

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\benches\image_histogram.rs`
uses the same completed-work/verification boundary, 20 samples, two-second
warmups and four-second target measurement windows (extended where needed).
Cases run sequentially on a loaded desktop. Large transient CPU/transfer stalls
occurred in this run; intervals do not capture between-case load changes.
The rotating experiment is the main comparison and Criterion is supporting
evidence. The following arithmetic-mean estimates and 95% confidence intervals
are in ms; they differ from slope estimates printed for linear sampling.

| Image / distribution | CPU | Resident | BinsReadback | UploadAndBins | ImageReadbackCpu |
| --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 0.162 [0.155, 0.169] | 0.480 [0.471, 0.491] | 1.333 [1.318, 1.349] | 2.108 [2.026, 2.207] | 2.698 [2.585, 2.816] |
| 640x480 / hot | 0.444 [0.436, 0.452] | 0.630 [0.617, 0.643] | 1.854 [1.678, 2.071] | 12.703 [5.742, 21.594] | 65.907 [46.772, 84.375] |
| 1920x1080 / mixed | 4.145 [2.721, 5.610] | 1.163 [1.150, 1.176] | 1.298 [1.270, 1.329] | 6.203 [6.017, 6.428] | 10.802 [10.583, 11.057] |
| 1920x1080 / hot | 0.762 [0.738, 0.789] | 1.094 [1.067, 1.124] | 3.026 [2.782, 3.268] | 6.218 [5.955, 6.543] | 10.354 [10.183, 10.514] |
| 3840x2160 / mixed | 1.519 [1.497, 1.540] | 1.354 [1.306, 1.439] | 2.178 [1.968, 2.423] | 19.664 [19.129, 20.286] | 35.648 [35.013, 36.398] |
| 3840x2160 / hot | 2.717 [2.669, 2.773] | 3.139 [3.082, 3.195] | 6.711 [6.488, 6.958] | 21.500 [20.578, 22.601] | 36.778 [35.888, 37.872] |

The sequential 4K hot-bin resident estimate (3.139 ms) also exceeded the CPU
host-input estimate (2.717 ms). VGA hot-bin upload/full-image readback and the
1080p mixed CPU outliers remain in the table; no cases were discarded or rerun
to improve the reported comparison.

## Reproduction and artifacts

```powershell
$env:RAYON_NUM_THREADS = '6'
cargo run -p spatialrust-gpu --features gpu-image --release --example gpu_histogram
cargo bench -p spatialrust-gpu --features gpu-image --bench image_histogram -- --noplot
```

Ignored local receipts:

- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-histogram-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-histogram-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-histogram-estimates.json`

Delivery is stacked on frozen PR 127. Merge automation waits for that dependency,
retargets to main, requires all 114 named successes/no pending or failed checks
and a clean merge state, and stops on changed source/dependency commits or
conflicts. Pending CI is not reported as a completed merge. Device-side CDF
consumers and an explicit tested rounding contract for later equalization are
recorded as the next roadmap slice.

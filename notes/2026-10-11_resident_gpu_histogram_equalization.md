# Resident GPU histogram equalization — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Frozen parent: `3ad82f911bdd511268e656f54b785614487c112f`, PR 129,
`codex/resident-gpu-image-cdf`. One grayscale histogram-equalization slice.
CPU point-query PR 127 merged as `dd42c44775b9e32700b8d8235f6bb4a63b3f2392`;
GPU histogram PR 128 merged as `65d5e3e044516b6cbcdb720b3eb4773948315154`.
CDF PR 129 remains pending; its macOS HTTP HEAD test failed with ConnectionReset.
Its GPU source is unchanged. A frozen-head watcher waits for workflow completion,
requests one isolated retry of that job, and only restarts the named-check merge
guard after success. Other failed jobs/changed heads/repeated failure stop it.
The failure is preserved in ignored local job logs; no streaming code is changed.

## Execution and ownership

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\equalize.rs`
adds `equalize_histogram_gpu` and `equalize_histogram_gpu_profiled` under
`gpu-image`. Input must be a resident one-channel u8 image; RGB conversion is an
explicit preceding operation. Output is a separate, caller-owned `GpuImage`,
preserving dimensions and metadata. All-zero/all-255/arbitrary constant inputs
preserve their original byte, matching the CPU reference. Empty GPU images are
rejected by the existing upload contract; counts above u32::MAX and unsupported
compute/texture limits fail before allocating algorithm work.

Five compute passes use three ordered submissions: two histogram passes, one
CDF pass, and lookup plus image application together. The 256-entry u32 lookup
adds 1 KiB private submitted scratch to the existing bounded histogram partials
(up to 256 KiB). The temporary histogram/CDF outputs are 1 KiB each and recycle
only after dependent reads are submitted. Lookup scratch recycles after both
lookup and application submission, never while its encoder is unsubmitted.
The final RGBA8 texture stays live until the caller drops or explicitly recycles
it. Pool budgets restrict retention, not caller-held live resources.

Application uses at most 64x64 workgroups of 16x16 lanes, with each axis further
bounded by device dispatch limits. Two-dimensional grid strides remove per-pixel
division/modulo and guard increments against u32 wrap. Every output texel is overwritten, including unused logical components.
Pipeline families are cached lazily per runtime. CDF buffer access is restricted
to the image module; no public raw-buffer escape hatch is added. No dependency,
core API, CPU equalization output or automatic placement policy is changed.

Construction performs no image/bin payload readback or implicit host wait.
Receipts inherit source history and record histogram partial/final bins, CDF,
1 KiB lookup and the physical RGBA8 output write. Uniform/driver metadata follows
existing receipt conventions. Explicit output `readback_u8` copies the full
physical RGBA8 image; it cannot be replaced by the 1 KiB bin readback used for
histogram-only analysis. Native five-stage timestamp collection explicitly reads
80 query bytes separately from payload receipts. Capacity rejection reserves no
work and preserves prior samples; inner recording failure invalidates collection.
Browser WASM device code is compile-checked; browser shader execution/async
readback is not verified here. APIs remain provisional.

## CPU rounding contract

The CPU implementation in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\analysis.rs`
uses `n * (255.0 / d)` in binary64, followed by nearest-even byte rounding,
where d excludes the first occupied bin and n is the subsequent cumulative
count. This differs from rounding the exact rational at some half boundaries:
with d=50 and n=25, binary64 produces 127.49999999999999 and returns 127 rather
than 128. A 51-pixel image with one minimum pixel and two 25-pixel groups exercises
this behavior through the production CPU image API. CPU behavior is preserved.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\equalize_lookup.wgsl`
uses only u32/i32 words. Two-word multiplication computes 255*n without overflow,
and eight quotient trials give exact quotient/remainder. For non-half values,
binary64 error cannot change rounding: adjacent rational rounding-boundary
distance is at least 1/(2*d) >= about 1.16e-10, while reciprocal/multiply error
below 255*(2^-52 + 2^-106) is about 5.67e-14 for these bounded positive counts.

At exact halves, lane zero finds the reciprocal exponent, performs 45..77
modular remainder doublings and determines nearest-even binary64 division error
magnitude/direction. Each half-bin compares twice n*error against d times the
product's half-ulp scale, using two-word integer multiplication/shifts. A strict
exceedance produces the CPU's one-ulp drift; equality rounds back to the exactly
representable half. Otherwise nearest-even byte rounding applies. No GPU float32
approximation or optional shader float64 feature is involved. Reciprocal error
is computed once per lookup workgroup and broadcast after an unconditional
barrier. Constant images use an identity lookup, preserving their observed byte.

## Validation

Full native `cargo test -p spatialrust-gpu --all-features -- --test-threads=2`
passed 79 unit, five new equalization integration, four CDF integration, three
histogram integration and seven prior image/point timing integration tests:
98 total on the real GTX 1660 Ti/Vulkan device.

The batched lookup proof compares 2,309,120 actual GPU entries against independent
CPU binary64 evaluation/nearest-even rounding. It covers denominators 1..4096,
512 deterministic full-width random denominators, adversarial half cases and
their neighbors, values around 2^24, 2^31 and u32::MAX counts, monotone CDFs and
constant inputs. Nine thousand twenty CDF rows use the real production shader,
with one combined readback. No production lookup upload or test-only shader is
substituted. An initial WGSL reserved-word parsing error was corrected before
these checks; all final shaders execute successfully.

Integration tests verify packed grids through 1001x1003 and lane boundaries,
one-column input, zero/255/43 constants, low contrast, true halves and binary64
drift, a strided 1025x1027 ROI with metadata/padding, RGB-gray-equalization chains,
stale buffers/textures, retained outputs, runtime moves, zero budgets, four
concurrent callers retaining 12 outputs, foreign runtime/channel rejection
without work allocation, five-stage profiling/byte parity and capacity/failure
handling. Existing GPU point/image regression tests also passed.

Default GPU tests, image-only test compilation and the adapter-free equalization
planner passed. Full/default all-target Clippy, minimal WGPU and image+AoSoA WASM
library checks, facade `gpu-image`, and Rustdoc passed with warnings denied.
Rustfmt/diff/CI YAML checks are included. Hosted GPU-less CI compiles runtime
tests and executes three adapter-free planners; shader execution is local.
The benchmark compile job explicitly includes `image_equalize`. Required named
check inventory remains 114.

## Completed-work measurements

The first measurements below used the initial flat grid-stride application.
Its pixel division/modulo was then replaced with a bounded two-dimensional
application. These baseline receipts/estimates are preserved, not discarded.
Final measurements are recorded separately; the desktop load between runs is
not controlled, so they are not an isolated speedup comparison.

Windows, Rust 1.98.1 release/thin LTO, i7-9750H, GTX 1660 Ti/Vulkan NVIDIA 596.36.
Other user Python computations remained active and were not stopped/reprioritized.
Only the benchmark process sets `RAYON_NUM_THREADS=6`. CPU reference is production
`spatialrust_vision::equalize_histogram`, including its bounded parallel histogram
and image application for large inputs. These desktop-load observations are not
portable speedup guarantees. Do not compare the timings with earlier slices as
if external load were fixed.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\examples\gpu_equalize.rs`
uses prebuilt host images/uploaded sources, two warmups and ten rounds rotating
six modes in one process. Low-contrast mixed values are
`(i*37 + i/997) % 129 + 32`; hot-bin images contain 43. Every measured image is
byte-verified after timing. Verification, destruction/recycling and optional
pass timestamps are excluded. GPU resident modes wait for completion inside
timing and explicitly read the full output image after timing for verification;
the executable as a whole therefore performs those untimed downloads. Separate
CPU-readback sources keep their cumulative histories out of resident inputs.

| Mode | Starting input | Timed output | Timed payload transfers |
| --- | --- | --- | --- |
| CpuHost | host grayscale | host equalized image | none |
| GpuResident | resident texture | completed resident equalized texture | none |
| GpuReadback | resident texture | host equalized image | full RGBA8 D2H |
| UploadRoundTrip | host grayscale | host equalized image | full RGBA8 H2D and D2H |
| GpuInputCpuReturnGpu | resident texture | completed resident image via CPU | full RGBA8 D2H and H2D |
| GpuInputCpuOutput | resident texture | host image via CPU | full RGBA8 D2H |

Entries are median [observed min, max] ms over ten calls. Ranges are not
confidence intervals; all adverse cases are retained.

| Image / distribution | CpuHost | GpuResident | GpuReadback | UploadRoundTrip | GpuInputCpuReturnGpu | GpuInputCpuOutput |
| --- | --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 0.470 [0.350, 6.410] | 0.820 [0.575, 1.556] | 2.894 [2.217, 5.977] | 3.552 [2.906, 5.864] | 4.100 [3.017, 12.376] | 3.422 [2.216, 6.004] |
| 640x480 / hot | 0.640 [0.585, 0.847] | 0.770 [0.550, 1.247] | 3.279 [2.468, 5.439] | 3.932 [3.094, 15.612] | 5.138 [3.121, 21.734] | 4.113 [2.814, 5.273] |
| 3840x2160 / mixed | 4.332 [3.288, 13.322] | 7.675 [1.639, 19.379] | 40.137 [31.821, 53.627] | 66.234 [54.815, 84.522] | 57.098 [51.774, 122.543] | 51.260 [33.734, 79.144] |
| 3840x2160 / hot | 8.564 [4.246, 37.916] | 14.629 [13.764, 15.293] | 56.192 [42.546, 65.862] | 91.008 [70.137, 182.131] | 77.965 [57.201, 210.888] | 56.845 [40.140, 119.136] |

For the same resident-input/resident-output destination, 4K mixed GpuResident
was 7.675 ms versus 57.098 ms for downloading, CPU equalization and reuploading.
It avoids 33,177,600 bytes in each direction. Host-input CPU equalization was
4.332 ms with different starting/output placement and was faster than the
resident dispatch here; host UploadRoundTrip was 66.234 ms and much slower.
Returning a full host image still costs its full readback: 4K mixed GpuReadback
40.137 ms versus GpuInputCpuOutput 51.260 ms, with overlapping observed ranges.
No universal GPU advantage or automatic placement policy is inferred.

Warmup rotation produced zero new pooled histogram/CDF buffer and scratch
allocations for all four profiles; retained scratch was 263,168 bytes. This
does not cover uniforms, bind groups, encoders, mapped staging or driver storage.
Texture allocation/retention is reported separately: lifetime allocation counts
were 4/4/19/33 and discards 0/0/12/26 across VGA mixed/hot and 4K mixed/hot.
This run retains earlier-resolution textures under the normal 64 MiB budget;
large upload/return modes need multiple textures and cause real allocation churn.
The texture counters include fixture setup and warmup, so they are not a pure
measured-rotation allocation delta. No zero-texture-allocation claim is made.
4K profiles retained 38,092,800 cached texture bytes within the 67,108,864-byte
budget. Live source/output textures are separate from that retention budget.

Example-only pass times are kept in the raw log. In the loaded 4K mixed call,
partials/reduce/CDF/lookup/application were 2094.752/120.800/13.440/44.352/
10132.864 us; query collection was 80 bytes. These clock samples exclude CPU
setup, waits, transfers and gaps, and were collected separately from median
wall-call samples. GPU activity/load can change between those observations.

### Initial flat-grid Criterion evidence

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\benches\image_equalize.rs`
measures four host-output modes on VGA/4K mixed/hot images: 16 cases, 20 samples,
one-second warmup and three-second target measurement window (extended where
necessary). All measured work completes and every image is byte-verified after
timing; all Criterion outputs are on the host, so verification adds no extra GPU
transfer. Cases run sequentially under external load. Arithmetic mean and 95%
confidence intervals below are in ms, distinct from printed slope estimates.
Intervals do not account for changing between-case desktop load.

| Resolution / distribution | CPU host | GPU readback | Upload round trip | GPU input, CPU output |
| --- | ---: | ---: | ---: | ---: |
| 640x480 / mixed | 0.328 [0.317, 0.340] | 2.689 [2.614, 2.785] | 3.494 [3.402, 3.603] | 2.739 [2.697, 2.790] |
| 640x480 / hot | 0.469 [0.457, 0.481] | 2.880 [2.784, 2.977] | 3.367 [3.304, 3.433] | 3.177 [3.106, 3.247] |
| 3840x2160 / mixed | 3.710 [3.641, 3.778] | 31.123 [30.017, 32.363] | 52.892 [51.366, 54.791] | 40.829 [39.252, 42.352] |
| 3840x2160 / hot | 4.717 [4.492, 4.939] | 28.107 [27.636, 28.632] | 61.294 [58.601, 64.411] | 40.110 [39.542, 40.789] |

## Final two-dimensional application measurements

Same rotating protocol and timed-placement definitions as above; 24 final
median [observed min, max] ms results. All output bytes matched the CPU.

| Image / distribution | CpuHost | GpuResident | GpuReadback | UploadRoundTrip | GpuInputCpuReturnGpu | GpuInputCpuOutput |
| --- | --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 0.323 [0.276, 0.431] | 0.512 [0.458, 0.644] | 2.131 [1.894, 2.636] | 2.771 [2.378, 3.131] | 2.750 [2.439, 3.238] | 2.235 [1.948, 2.575] |
| 640x480 / hot | 0.531 [0.476, 0.564] | 0.563 [0.526, 0.869] | 2.592 [2.213, 2.852] | 3.182 [2.947, 3.867] | 3.214 [3.035, 3.443] | 2.688 [2.414, 3.422] |
| 3840x2160 / mixed | 3.072 [2.766, 3.703] | 1.084 [1.028, 3.570] | 24.079 [22.539, 26.848] | 36.486 [34.822, 43.039] | 39.445 [36.624, 44.336] | 26.779 [25.643, 33.167] |
| 3840x2160 / hot | 3.432 [3.232, 4.598] | 6.191 [2.588, 6.449] | 31.920 [25.172, 34.247] | 55.276 [41.779, 56.933] | 45.707 [39.297, 50.161] | 33.745 [28.702, 37.291] |

With resident input and resident output, 4K mixed measured 1.084 ms versus
39.445 ms for the CPU download/equalize/reupload alternative, avoiding
33,177,600 payload bytes in each direction. CPU host-only was 3.072 ms with
different input/output placement; UploadRoundTrip was 36.486 ms. VGA mixed
resident GPU (0.512 ms) was slower than host CPU (0.323 ms). Constant 4K resident
GPU (6.191 ms) was slower than host CPU (3.432 ms); hot-bin histogram contention
and external load are not separated. No automatic placement rule follows.

Final warmed histogram/CDF buffer and private scratch allocation deltas remained
zero, with 263,168 retained scratch bytes. Texture lifetime allocation counts
4/4/19/33, discards 0/0/12/26 and 4K retained 38,092,800 bytes were unchanged from
the initial run. The same real large-texture churn remains. These are retention
and lifetime counters, not per-call peak/live GPU memory measurements.

Separate final pass-clock samples for 4K mixed were
933.792/76.224/8.736/27.104/1782.112 us and for 4K hot
703.264/76.064/10.240/17.664/1084.000 us (partials/reduce/CDF/lookup/application).
Each explicitly collected 80 query bytes. Pass-clock sums and rotating wall-call
medians come from different calls under changing load; they must not be equated.

### Final Criterion evidence

All 16 final host-output cases completed with byte verification. Arithmetic
mean [95% confidence interval] ms, using the same protocol as the initial run:

| Resolution / distribution | CPU host | GPU readback | Upload round trip | GPU input, CPU output |
| --- | ---: | ---: | ---: | ---: |
| 640x480 / mixed | 3.408 [2.566, 4.250] | 15.278 [9.786, 21.195] | 24.217 [15.524, 33.984] | 4.561 [4.434, 4.680] |
| 640x480 / hot | 0.631 [0.622, 0.639] | 3.902 [3.806, 4.006] | 4.538 [4.484, 4.591] | 4.089 [3.985, 4.197] |
| 3840x2160 / mixed | 4.359 [4.308, 4.414] | 39.381 [37.367, 41.377] | 53.096 [51.410, 55.175] | 42.285 [41.515, 43.133] |
| 3840x2160 / hot | 7.234 [5.578, 9.559] | 51.086 [47.227, 55.434] | 81.972 [77.773, 86.446] | 54.373 [50.593, 58.250] |

Criterion printed regressions in several GPU cases and in the unchanged CPU
reference. Those observations are retained. The final sequential Criterion
run also differs materially from the immediately preceding rotation; desktop
load is uncontrolled and within-case bootstrap intervals do not capture it.
Neither the printed comparison against prior saved cases nor the faster final
rotating results establish a controlled before/after implementation speedup.
The concrete result is byte parity, completed resident execution and explicit
transfer avoidance; performance depends on placement, distribution and load.

## Reproduction and delivery

```powershell
$env:PATH = 'C:\Users\rsasa\.cargo\bin;' + $env:PATH
$env:RAYON_NUM_THREADS = '6'
cargo run -p spatialrust-gpu --features gpu-image --release --example gpu_equalize
cargo bench -p spatialrust-gpu --features gpu-image --bench image_equalize -- --noplot
```

Ignored local receipts:

- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-tests.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-estimates.json`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-before-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-before-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-equalize-before-estimates.json`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\pr129-macos-streaming-job.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\pr129-retry-http-head.log`

Stacked delivery waits for frozen PR 129, retargets to main and requires all 114
named successes/no pending or failed checks and a clean merge state. Changed
source/parent heads or conflicts stop automation. Pending CI is not claimed as a
completed merge. GPU CLAHE is the next independent roadmap slice, with clipping,
padding, interpolation, byte rounding and temporary budgets to validate first.

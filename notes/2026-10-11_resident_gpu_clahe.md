# Resident GPU CLAHE — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Named slice: grayscale resident contrast-limited adaptive histogram equalization.
Frozen parent: PR 130, `c591b847785f93a7853fbdafc8879e1116ddb207`,
`codex/resident-gpu-histogram-equalization`.

## Device execution and temporary ownership

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\clahe.rs`
adds `clahe_gpu` and `clahe_gpu_profiled` under `gpu-image`. Source must be a
resident one-channel u8 image; RGB-to-gray is an explicit preceding operation.
The independently owned output preserves dimensions and metadata. CLAHE may
change constant-image bytes, matching CPU CLAHE (global equalization has a
different constant-image contract). Empty GPU images remain unsupported.

Three passes run in one ordered submission: axis geometry, per-tile histogram /
clipping / cumulative lookup, then bilinear application. Geometry uses 16 bytes
per source row/column. Each tile has a 256-entry u32 lookup (1 KiB). For 4K 8x8,
per-call private storage is 96,000 + 65,536 = 161,536 bytes. Both scratch leases
remain exclusive until the application reads are submitted. Caller-held output
textures never return implicitly to pools; queue order protects submitted reuse.
Pool budgets constrain retention rather than caller-held/live driver memory.

Tile workers are bounded by tile count, device dispatch limits and 256 groups;
groups stride across additional tiles. Application uses at most 64x64 groups of
16x16 lanes, with guarded two-axis strides. Geometry groups are bounded by 256.
All geometry/lookup entries and physical output components are overwritten.
Each tile has at most u32::MAX padded pixels. Padded coordinate products, tile
count, geometry count/byte lengths, uniform/storage binding, workgroup, texture
and dispatch limits reject before algorithm allocation. Checked multiplication
prevents pathological grids from overflowing the LUT byte calculation.

Reflect101 uses positive-coordinate periodic reflection, including singleton
axes and grids larger than the source. GPU counts are clipped and redistributed
with the CPU's whole-bin batch plus uniformly stepped residual rule. The scalar
clip count is computed on the host from dimensions and the f64 clip argument;
clamping it to tile area is equivalent to CPU usize saturation at huge finite
limits. Zero (including negative zero) disables clipping. NaN, infinities,
negative limits, zero grids, other channels and foreign runtimes reject.

No image/bin/axis-geometry payload is downloaded or uploaded during construction;
geometry is generated on device. There is no implicit host wait. Receipts inherit
source history and record geometry, lookup and physical RGBA8 writes. Uniform
metadata follows the existing accounting convention; receipts do not measure
bind groups, encoders, uniforms, driver storage or peak live memory. Explicit
output readback copies the full physical RGBA8 texture. Three-stage timestamp
collection explicitly reads 48 query bytes separately from payload receipts.
Capacity validation occurs before work allocation and preserves prior recordings;
inner errors invalidate collection. Pipeline families are cached per runtime.
No dependencies, core APIs or CPU algorithms are changed. APIs are provisional.

## CPU byte-rounding contract

The reference is
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\analysis.rs`.
The lookup uses inclusive counts times binary64 `255.0 / tile_area`, including
reciprocal/multiply drift at exact half bytes. Existing global-equalization
integer-word helpers moved unchanged into the private
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\rounding_u32.wgsl`;
its 2,309,120 prior lookup comparisons passed after this extraction.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\binary64_positive.wgsl`
implements the finite non-negative normal-or-zero subset used by CLAHE, not a
public/general floating-point facility. Normalized 53-bit significands use two
u32 words. Addition/magnitude subtraction align guard/round/sticky bits;
multiplication forms a 106-bit product in four u32 limbs and rounds nearest-even.
Unsigned ratios use exact modular division and remainder rounding. Zero is
explicit. Operand/result ranges here stay far from subnormal/overflow domains;
signs, NaNs, infinities and general IEEE exceptional values are outside the helper.
No shader f32 approximations or optional shader float64 capability is required.

Axis generation reproduces rounded division, subtraction of 0.5, floor and
fractional subtraction in CPU order. The negative first half-tile uses the
symmetric magnitude subtraction before subtracting from one. Clamped neighbor
indices do not discard the fractional weight. Interpolation preserves each
rounded multiplication/addition of the top, bottom and vertical blends, followed
by nearest-even byte rounding/clamping. Equal integer corners bypass the blend:
CPU error is below 1e-12 there, far from a half-byte boundary, so the final byte is
invariant. Other corners take the full arithmetic path.

## Verification

The full native all-feature GPU suite passed 105 tests on GTX 1660 Ti/Vulkan:
81 unit, four CDF, five CLAHE, five equalization, three histogram and seven shared
image/point timing tests. An additional production interpolation test passed
66,049 synthetic pixels, for 106 passing distinct tests. Synthetic LUT/geometry
uploads belong only to that test; production builds both on device. It directly
runs the production application pipeline with exact/neighboring half boundaries,
true even/odd byte ties, zero/one weights and equal corners against independent
CPU binary64 evaluation. The final test fixture/lints also passed after changing
its column construction to iterators; production code was unchanged.

The arithmetic proof separately compares 65,536 actual GPU rows of sum, product,
magnitude difference, unsigned ratio and axis geometry against CPU bit patterns.
Half the rows densely cover practical integer positions and small tile sizes;
the rest include deterministic full-width numerators/denominators. Inputs cover
zero, halfway values, adjacent-one values and positive exponent ranges. A prior
local Python prototype also compared 200,000 operation pairs; GPU execution and
independent CPU comparisons are the delivered evidence.

End-to-end tests cover tiny/odd grids, grids larger than images, Reflect101,
0/43/255 constants, mixed low contrast, disabled/tiny/large/maximum finite clips,
lookup half drift, strided 1025x1027 ROI/metadata, RGB-gray chains, stale storage,
retained outputs, runtime moves, zero budgets, four concurrent callers retaining
12 outputs, invalid/foreign inputs without work, and three-stage profiling.
Existing point/image regressions passed. Full/default all-target Clippy, default
GPU tests, image-only test compilation, adapter-free planner, minimal WGPU and
image+AoSoA WASM library Clippy, facade `gpu-image`, Rustdoc with warnings denied,
rustfmt/diff and CI YAML checks passed. Browser shader execution/async readback
is unverified. Hosted GPU-less CI compiles native shader tests, runs four
adapter-free planners and compiles `image_clahe`; the named inventory stays 114.

## Completed-work measurements

Windows, Rust 1.98.1 release/thin LTO, i7-9750H, GTX 1660 Ti/Vulkan NVIDIA 596.36.
Other user Python computations remained active, without stopping/reprioritizing
them. Only the benchmark process sets `RAYON_NUM_THREADS=6`; CPU reference is
production bounded-parallel `spatialrust_vision::clahe` at clip 2.0, grid 8x8.
Mixed bytes are `(i*37 + i/997) % 129 + 32`; hot images contain 43. Desktop load
is uncontrolled. These observations do not define a portable placement policy.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\examples\gpu_clahe.rs`
rotates six modes for ten rounds after two warmups with prebuilt host/uploaded
sources. Timed GPU work waits for completion. Every output is byte-verified after
timing; drops/recycling and optional pass clocks are excluded. Resident-output
verification explicitly downloads the full output after timing, so the executable
as a whole still performs those transfers. A separate uploaded CPU-readback
source keeps its cumulative receipt history out of resident input.

| Mode | Starting input | Completed timed output | Timed payload transfers |
| --- | --- | --- | --- |
| CpuHost | host | host | none |
| GpuResident | resident texture | resident texture | none |
| GpuReadback | resident texture | host | full RGBA8 D2H |
| UploadRoundTrip | host | host | full RGBA8 H2D + D2H |
| GpuInputCpuReturnGpu | resident texture | resident texture via CPU | full RGBA8 D2H + H2D |
| GpuInputCpuOutput | resident texture | host via CPU | full RGBA8 D2H |

Rotating entries are median [observed min, max] ms over ten completed calls.
Ranges are not confidence intervals. All conditions are retained.

| Image / distribution | CpuHost | GpuResident | GpuReadback | UploadRoundTrip | GpuInputCpuReturnGpu | GpuInputCpuOutput |
| --- | --- | --- | --- | --- | --- | --- |
| 640x480 / mixed | 9.332 [8.406, 10.135] | 2.488 [2.399, 3.039] | 3.937 [3.702, 4.345] | 4.938 [4.565, 5.736] | 12.932 [11.318, 16.997] | 12.296 [11.104, 18.347] |
| 640x480 / hot | 7.340 [6.068, 11.413] | 2.459 [2.361, 2.688] | 3.705 [2.992, 4.078] | 4.634 [3.932, 4.822] | 11.476 [10.535, 17.866] | 10.720 [9.045, 12.426] |
| 3840x2160 / mixed | 40.716 [38.953, 45.795] | 15.282 [13.873, 16.026] | 43.943 [42.050, 45.542] | 70.385 [65.980, 72.600] | 91.101 [88.258, 99.208] | 75.537 [69.268, 86.139] |
| 3840x2160 / hot | 36.510 [33.060, 42.216] | 7.281 [7.137, 7.403] | 37.019 [32.702, 43.000] | 60.954 [57.228, 62.323] | 83.707 [79.192, 92.881] | 69.774 [64.370, 80.712] |

For identical resident-input/resident-output placement, 4K mixed GPU measured
15.282 ms versus 91.101 ms for the CPU download/process/reupload alternative,
avoiding 33,177,600 bytes in each direction. CPU host-only was 40.716 ms with
different placement. UploadRoundTrip was 70.385 ms, slower than host CPU;
GpuReadback was 43.943 ms versus 75.537 ms for CPU output from resident input.
Even the resident-input readback path was slower than host-only CPU in that
rotation. Sequential Criterion below differs, so no automatic policy follows.

Warm rotations allocated zero new private scratch or resident-bin buffers.
Per-call scratch is 83,456 bytes for VGA and 161,536 for 4K; retained scratch was
83,456 then 179,456 bytes because the previous geometry size remains cached.
Uniforms/bind groups/encoders/staging/driver allocations are outside these counters.
Texture lifetime allocations were 4/4/19/33 and discards 0/0/12/26 for VGA
mixed/hot then 4K mixed/hot. Earlier small textures occupy part of the 64 MiB
retention budget; multiple large upload/return textures cause real churn.
Final 4K cache retained 38,092,800 bytes within 67,108,864 bytes. Fixture/warmup
allocations are included; these are not warmed texture-allocation deltas or
per-call peak/live-memory measurements.

Separate geometry/lookup/application GPU-clock samples were
43.712/2722.144/10957.184 us for 4K mixed and 45.056/2829.696/2781.824 us for hot.
Queries explicitly copied 48 bytes. These samples exclude setup, waits, transfers
and gaps; sums must not be equated with rotating medians from different calls.

### Criterion host-output comparison

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\benches\image_clahe.rs`
completed all 16 cases, 20 samples, one-second warmup and three-second target
measurement windows. The calibrator extended some estimates; the VGA mixed
GpuReadback case selected 20 total iterations with a 42.565-second estimated
window. Actual completed-call durations, not that estimate, are reported below.
Verification/destruction/recycling are excluded from each returned duration;
every result is byte-verified afterward. All Criterion outputs are on the host,
so verification needs no extra GPU transfer. These are completed-call latency
measurements, not whole-executable throughput including verification.

Arithmetic mean [95% confidence interval] ms, distinct from printed slope
estimates. Cases are sequential under uncontrolled load; intervals do not account
for between-case desktop changes or establish a portable ordering.

| Resolution / distribution | CPU host | GPU readback | Upload round trip | GPU input, CPU output |
| --- | ---: | ---: | ---: | ---: |
| 640x480 / mixed | 9.394 [8.854, 10.048] | 5.515 [5.393, 5.644] | 3.020 [2.958, 3.089] | 10.951 [10.677, 11.229] |
| 640x480 / hot | 6.244 [6.009, 6.487] | 2.242 [2.172, 2.321] | 2.861 [2.797, 2.933] | 8.453 [8.264, 8.656] |
| 3840x2160 / mixed | 35.306 [34.403, 36.262] | 30.906 [30.306, 31.578] | 49.744 [49.312, 50.197] | 67.070 [65.750, 68.603] |
| 3840x2160 / hot | 32.975 [32.558, 33.452] | 30.363 [28.466, 32.414] | 44.343 [43.650, 45.378] | 68.691 [67.723, 69.597] |

The sequential samples differ materially from the earlier rotation, including
unchanged CPU timings and readback/round-trip ordering. Both datasets and adverse
host-transfer cases are preserved. Correct byte output and explicit transfer
avoidance are established; speeds depend on placement, distribution and load.

## Reproduction and guarded delivery

```powershell
$env:PATH = 'C:\Users\rsasa\.cargo\bin;' + $env:PATH
$env:RAYON_NUM_THREADS = '6'
cargo run -p spatialrust-gpu --features gpu-image --release --example gpu_clahe
cargo bench -p spatialrust-gpu --features gpu-image --bench image_clahe -- --noplot
```

Ignored local receipts:

- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-tests.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-unit.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-interpolation.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-clahe-estimates.json`

Dependency PR 129 already requested one isolated macOS HTTP fixture retry on
frozen head `3ad82f911bdd511268e656f54b785614487c112f`. The initial observer stopped
on three GitHub jobs-list HTTP 502 reads. Its requested attempt 2/job
114377474375 is now observed through frozen run/job endpoints, without a second
rerun; success restarts its named-check guard. PR 130 remains frozen to the parent
hash above. This slice waits for its merge, then retargets to main. Changed source
or dependency heads, failed/missing required checks or conflicts stop automation;
merge requires all 114 named successes, no pending/failed checks and CLEAN state.
Pending CI is not a completed merge. No streaming fixture code is changed here.

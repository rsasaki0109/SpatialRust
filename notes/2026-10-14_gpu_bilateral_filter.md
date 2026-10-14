# Exact resident bilateral filtering and CPU output bands

`GpuBilateralFilter::prepare(runtime, channels, options)` explicitly prepares
and uploads image-independent coefficients under `gpu-image`. The owned plan
can filter multiple source sizes with the same 1..=4 logical u8 channels and
options. `plan.filter(runtime, source)` submits one device pass and returns an
independently owned `GpuImage`. Upload, readback and texture recycling remain
caller operations. Image metadata/order are retained; alpha contributes to the
joint color distance and receives its own weighted mean.

`GpuBilateralOptions` admits diameter 1..=7, including even diameters with
radius `floor(diameter/2)`, finite positive color/space sigmas and existing
`GpuImageBorder::{Replicate, ConstantZero}`. The neighborhood is circular, with
1/5/13/29 taps at radii 0/1/2/3. Color distance is the square of the sum of
absolute logical-channel differences, matching the current CPU operator.

For u8 input the L1 difference is an exact integer in 0..=255*channels.
Host binary64 FMA/exp computes one row per distinct spatial squared distance,
using the same factors and operations as CPU. Tap descriptors retain dy/dx
order. Center taps can only sample color distance zero; unused center-row
entries are zero. At most seven rows and 29 descriptors occupy 57,640 bytes
for diameter 7 / four channels. The table is image-independent, not a sensor
payload or an image-sized staging buffer. Its explicit upload is recorded in
`plan.receipt()` separately from image receipts and occurs once per plan.

The admitted coefficient set is normal finite binary64 values or zero.
Nonfinite factors and positive subnormal coefficients reject before device
buffer/cache allocation, without a CPU fallback or precision relaxation.
Some finite positive sigmas are therefore unsupported by the GPU plan (for
example color sigma 20 with four channels/default spatial sigma). Generic CPU
calls retain their existing broader parameter behavior. Product/positive-sum
states remain normal or zero; the center weight is one, so the denominator
stays positive. The existing positive binary64 word helpers are reused without
changing CLAHE or its shared source.

The device computes channel differences and looks up exact coefficient bits.
Each channel uses ordered rounded binary64 products/additions implemented with
u32 words. Byte output matches `(sum / total_weight).round().clamp(0,255)`.
Eight integer decisions compare exact numerator/denominator ratios against
half-byte thresholds. The lower boundary is the midpoint between a threshold
and its preceding binary64 value; the threshold has an even significand, so
midpoint ties round up to it. The 0.5 boundary handles the smaller predecessor
ULP below a power of two. Exact up-to-128-bit product/comparison words avoid
software floating division and preserve CPU division-then-half-up rounding.
There is no shader float64 capability requirement or widened output tolerance.

The plan is bound to its runtime and logical channel count. Coefficient buffer,
uniform, sampled/storage texture, bind group and workgroup capabilities are
checked during preparation. Image dimensions, signed neighborhood coordinates
and both dispatch axes are checked before output allocation/lazy pipeline
initialization. A single runtime-owned pipeline serves all admitted options.
One ordered pass writes four physical bytes per destination pixel. The 32-byte
per-call uniform is control data excluded from payload receipts. There is no
explicit image-sized scratch or host completion wait; driver private storage
and possible register spills are outside this allocation statement. A source
image's cumulative payload receipt is inherited; each result adds one 4*pixels
output footprint. Plan coefficients remain in the separate plan receipt, so
callers can count preparation once across repeated frames.

The unchanged generic CPU `bilateral_filter` now uses bounded native bands for
at least 262,144 logical components and multiple rows, capped by the current
Rayon pool and height. Each band writes disjoint parts of one safely initialized
final vector, using a real input component without extra scalar conversions.
Per-pixel circular sampling, f64 FMA/exp, color/reduction order and dtype
conversion remain the original operations. Small, single-row, one-worker and
WASM calls invoke the original full serial loop. No new dependency, unsafe code
or core change is added. GPU placement remains caller-selected.

Base: merged PR #193, `3565bad8dee15bd60ec08602156ed4c27b92dfe4`.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/runtime.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/advanced_filter.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral.rs`

Validation and measurement paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_bilateral_smoke.rs`

Five native GPU tests pass. The final matrix contains 3,000 conditions /
1,158,000 exact logical bytes, covering channels, circular/even diameters,
normal/zero/uniform coefficient regimes, noise/constants/impulses and both
borders across singleton, strips, small windows and workgroup tails. An earlier
full matrix also checked 10,976 conditions / 11,501,280 exact bytes before the
test matrix was reduced; a chain test then failed because it passed a BGRA
four-channel image to the existing RGB-only grayscale API. The final chain
uses a separate RGB source and plan, while the BGRA ROI checks remain.

An independent native GPU division proof covers 79,822 positive-normal/zero
ratio cases: all 255 half-byte thresholds, nearby numerator bit patterns,
dyadic/odd divisors, random large-exponent divisors, exact half-up ties and
extreme numerator/denominator scales. Another entry point observes the actual
production accumulators: 14,960 raw channel/denominator binary64 values match
an independent CPU circular/FMA/exp loop exactly across apertures, borders and
sigma regimes. Pure coefficient checks cover all diameters 1..=7, all channels,
all reachable L1 entries and four sigma regimes; resource/dispatch and rejected
factor/subnormal cases are exercised before device allocation.

Cold concurrent filtering initializes one family, preserves independent owned
outputs and survives input recycling. Strided BGRA/limited-range/straight-alpha
ROI output matches CPU. Foreign runtime/channel and invalid-plan calls leave
pipeline/pool state unchanged. A RGB bilateral -> gray -> Sobel -> CHW chain has
no intermediate host payload transfer. The facade smoke test passes. All 104
native image regression tests pass, including Gaussian/median/Canny/contrast/
transfer/temporary-resource tests.

Three new CPU tests include 9,600 small conditions plus threshold/strip/
1/2/4-thread pools, large u16/f32/f64 raw-bit comparisons and strided metadata
ROI checks. Empty/error and circular-support/even-diameter analytic cases use
the frozen complete parent function. All 35 filter tests pass. Vision/GPU
all-feature/all-target Clippy, native facade, minimal CPU-only GPU and WASM
filter/facade checks pass. The pre-existing filter-only
`LIGHT_PARALLEL_COMPONENTS` dead-code warning remains unchanged.

## Completed-call measurements

Two isolated native release CPU and two GPU surveys use Rust 1.98.1, 12
available CPU/Rayon workers, Vulkan discrete GPU execution and the default
64 MiB texture pool. All four surveys accepted attempt 1, with no detected
named-work overlap and empty benchmark stderr: CPU 191.18/146.52 s, GPU
26.45/26.87 s. Build/test runs, surveys and statistical processing are sequential.
Controllers require three quiet observations five seconds apart and monitor
named Cargo/compiler/linker/test/benchmark work between 500 ms waits. Any
overlap discards the entire attempt; the controller stops only its own native
benchmark and retains partial/rejected logs before retrying. This bounded
process guard does not prove every application, OS task or GPU activity is idle.

All conditions use color sigma 40, spatial sigma 3 and replicate borders, with
three warmups followed by 20 paired completed samples per route. Route order
rotates each round. Profile 0 is deterministic hash noise; profile 2 is all-255.
Fixtures, expected results and returned host image validation/drop are outside
timers. Every warm/timed output matches its exact reference. All timed samples
are retained without outlier removal.

CPU measurements compare the frozen complete parent implementation, forced
one-worker current code and the current public automatic-dispatch API. Both
serial implementations retain per-neighbor f64 exp and the original loop.
The automatic route actually schedules bands only above the component/row
threshold. Eight conditions cover small/sub-threshold images, exactly below/at
the cutoff, gray/RGB 3/5/7 diameters, 1024x1024, 1920x1080 and all-255 data.
CPU timings include result allocation and complete filtering. The frozen and
current functions compile into the same benchmark binary.

GPU measurements compare the improved public generic CPU API with:

- `full_new`: explicit plan preparation/coefficient upload, image upload,
  filtering, completed readback/unpack, input/output recycling and plan drop.
- `full_reuse`: the same image upload/filter/readback/recycling with a plan
  prepared outside timers. Its control preparation cost is amortized separately.
- `resident`: an already uploaded source and prepared plan, filtering/output
  checkout plus `wait_idle`. Validation readback and output recycling are outside
  this timer, giving a different transfer/output boundary.

The GPU algorithm combines coefficient lookup/reuse with actual device color
comparison, weighted accumulation and byte rounding. Its gains include those
algorithmic differences from the current generic CPU's per-neighbor exp.
They do not isolate device hardware alone or compare against an external
prepared-u8 CPU implementation. Eight conditions include gray/RGB/RGBA,
3/5/7 diameters, 65x33, 257x193, 1024x1024, 1920x1080 and all-255 input.
Plan upload bytes are recorded alongside samples; image receipts retain 4*pixels
H2D, one 4*pixels destination footprint and 4*pixels D2H for completed full calls.

Ratios are paired geometric means. Intervals use 10,000 bootstrap resamples of
20 log ratios and sorted zero-based endpoints 249/9749. CPU seed is
`2026101405 + 16*(survey-1) + 2*case_index + stage`, stage 0 serial / 1 automatic.
GPU seed is `2026101406 + 24*(survey-1) + 3*case_index + stage`, stage 0 new /
1 reuse / 2 resident. All conditions and adverse/inconclusive intervals appear
below. Tables use `parallel` for automatic CPU dispatch, including serial
fallback cases. Results are local synthetic measurements, not a placement
policy or cross-machine throughput guarantee.

For 1920x1080 RGB / diameter 5 noise, CPU old/current automatic ratios are
4.701 (4.298-5.169) and 4.360 (4.096-4.625); current CPU medians are
278.235/220.058 ms. Forced-one-worker ratios are 0.957/0.989, both intervals
crossing one. At 1024x1024 RGB / diameter 5, automatic ratios are 4.617/4.542;
all-255 ratios are 4.755/4.292. Small/sub-threshold conditions and the forced
serial routes have inconclusive intervals. Large image gains come from
parallel dispatch, with the original serial path retained.

Against the improved public CPU, 1920x1080 RGB / diameter 5 full-new GPU ratios
are 5.845 (5.048-6.699) and 4.767 (4.142-5.485), with medians
27.546/31.839 ms. Full-reuse ratios are 6.021 (5.322-6.790) and
4.687 (4.026-5.451), with medians 27.333/31.336 ms. Resident ratios are
10.089/8.997 with the different boundary above. These observations establish
no consistent timing improvement from coefficient reuse alone; its explicit
resource/preparation reuse contract holds independently of timing variation.
All seven non-tiny conditions have full-new and full-reuse intervals above
one in both surveys. At 65x33 gray / diameter 3, full-new ratios are
0.191 (0.176-0.216) / 0.184 (0.177-0.191), and resident ratios are
0.825 (0.757-0.922) / 0.906 (0.835-0.976). This small image favors CPU even
without full transfers. Absolute timings vary between surveys; no adverse
condition is removed or generalized into a speedup.

Raw accepted samples, execution/source provenance, all attempt logs and
controller/analysis scripts remain under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with
`cpu-bilateral-*` / `gpu-bilateral-*` prefixes. Final source/executable hashes
match those captured before measurement.

## Frozen/current CPU measurements

| Survey | Shape / channels / diameter / profile | Route | Baseline ms | Route ms | Paired ratio [95%] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | serial | 0.269 | 0.279 | 0.938x (0.814-1.043) |
| 1 | 65x33 / 1 / 3 / 0 | parallel | 0.269 | 0.419 | 0.922x (0.811-1.022) |
| 1 | 257x193 / 3 / 5 / 0 | serial | 30.240 | 29.406 | 0.983x (0.812-1.171) |
| 1 | 257x193 / 3 / 5 / 0 | parallel | 30.240 | 32.600 | 1.020x (0.866-1.189) |
| 1 | 511x513 / 1 / 3 / 0 | serial | 48.566 | 52.738 | 0.910x (0.789-1.051) |
| 1 | 511x513 / 1 / 3 / 0 | parallel | 48.566 | 48.603 | 0.963x (0.837-1.107) |
| 1 | 512x512 / 1 / 3 / 0 | serial | 52.763 | 51.309 | 1.034x (0.840-1.264) |
| 1 | 512x512 / 1 / 3 / 0 | parallel | 52.763 | 11.377 | 4.617x (3.911-5.473) |
| 1 | 1024x1024 / 3 / 5 / 0 | serial | 596.669 | 599.353 | 0.994x (0.950-1.040) |
| 1 | 1024x1024 / 3 / 5 / 0 | parallel | 596.669 | 121.862 | 4.617x (4.283-4.927) |
| 1 | 1920x1080 / 3 / 5 / 0 | serial | 1345.266 | 1371.435 | 0.957x (0.912-1.002) |
| 1 | 1920x1080 / 3 / 5 / 0 | parallel | 1345.266 | 278.235 | 4.701x (4.298-5.169) |
| 1 | 1024x1024 / 3 / 5 / 2 | serial | 633.312 | 647.285 | 0.993x (0.943-1.048) |
| 1 | 1024x1024 / 3 / 5 / 2 | parallel | 633.312 | 131.677 | 4.755x (4.375-5.134) |
| 1 | 1024x1024 / 1 / 7 / 0 | serial | 755.408 | 747.606 | 0.987x (0.940-1.032) |
| 1 | 1024x1024 / 1 / 7 / 0 | parallel | 755.408 | 171.265 | 4.303x (4.079-4.531) |
| 2 | 65x33 / 1 / 3 / 0 | serial | 0.262 | 0.266 | 1.099x (0.969-1.290) |
| 2 | 65x33 / 1 / 3 / 0 | parallel | 0.262 | 0.266 | 1.069x (0.957-1.227) |
| 2 | 257x193 / 3 / 5 / 0 | serial | 17.825 | 18.162 | 0.947x (0.786-1.107) |
| 2 | 257x193 / 3 / 5 / 0 | parallel | 17.825 | 18.959 | 0.969x (0.872-1.082) |
| 2 | 511x513 / 1 / 3 / 0 | serial | 33.718 | 33.754 | 0.993x (0.854-1.157) |
| 2 | 511x513 / 1 / 3 / 0 | parallel | 33.718 | 33.592 | 0.995x (0.849-1.185) |
| 2 | 512x512 / 1 / 3 / 0 | serial | 37.677 | 35.384 | 1.112x (0.953-1.310) |
| 2 | 512x512 / 1 / 3 / 0 | parallel | 37.677 | 8.981 | 3.782x (2.736-4.980) |
| 2 | 1024x1024 / 3 / 5 / 0 | serial | 475.767 | 453.078 | 1.060x (0.989-1.147) |
| 2 | 1024x1024 / 3 / 5 / 0 | parallel | 475.767 | 105.598 | 4.542x (4.160-4.967) |
| 2 | 1920x1080 / 3 / 5 / 0 | serial | 1008.340 | 965.241 | 0.989x (0.924-1.060) |
| 2 | 1920x1080 / 3 / 5 / 0 | parallel | 1008.340 | 220.058 | 4.360x (4.096-4.625) |
| 2 | 1024x1024 / 3 / 5 / 2 | serial | 434.940 | 452.274 | 1.001x (0.909-1.109) |
| 2 | 1024x1024 / 3 / 5 / 2 | parallel | 434.940 | 100.930 | 4.292x (3.817-4.875) |
| 2 | 1024x1024 / 1 / 7 / 0 | serial | 694.269 | 702.350 | 0.986x (0.945-1.029) |
| 2 | 1024x1024 / 1 / 7 / 0 | parallel | 694.269 | 164.771 | 4.182x (3.932-4.442) |

## GPU against current parallel generic CPU

| Survey | Shape / channels / diameter / profile | Route | Baseline ms | Route ms | Paired ratio [95%] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | full_new | 0.219 | 1.192 | 0.191x (0.176-0.216) |
| 1 | 65x33 / 1 / 3 / 0 | full_reuse | 0.219 | 1.203 | 0.189x (0.175-0.212) |
| 1 | 65x33 / 1 / 3 / 0 | resident | 0.219 | 0.286 | 0.825x (0.757-0.922) |
| 1 | 257x193 / 3 / 5 / 0 | full_new | 14.544 | 1.602 | 9.067x (8.657-9.486) |
| 1 | 257x193 / 3 / 5 / 0 | full_reuse | 14.544 | 1.392 | 10.315x (9.913-10.736) |
| 1 | 257x193 / 3 / 5 / 0 | resident | 14.544 | 0.604 | 23.956x (22.502-25.502) |
| 1 | 1024x1024 / 1 / 3 / 0 | full_new | 24.604 | 6.778 | 3.749x (3.470-4.010) |
| 1 | 1024x1024 / 1 / 3 / 0 | full_reuse | 24.604 | 6.457 | 3.975x (3.748-4.207) |
| 1 | 1024x1024 / 1 / 3 / 0 | resident | 24.604 | 1.282 | 19.825x (18.688-21.161) |
| 1 | 1024x1024 / 3 / 5 / 0 | full_new | 86.946 | 19.410 | 4.564x (4.155-4.972) |
| 1 | 1024x1024 / 3 / 5 / 0 | full_reuse | 86.946 | 18.992 | 4.691x (4.348-5.063) |
| 1 | 1024x1024 / 3 / 5 / 0 | resident | 86.946 | 11.714 | 7.973x (7.395-8.716) |
| 1 | 1024x1024 / 4 / 7 / 0 | full_new | 172.800 | 20.450 | 8.013x (7.081-9.056) |
| 1 | 1024x1024 / 4 / 7 / 0 | full_reuse | 172.800 | 23.067 | 7.685x (6.798-8.714) |
| 1 | 1024x1024 / 4 / 7 / 0 | resident | 172.800 | 13.684 | 12.463x (11.263-13.856) |
| 1 | 1920x1080 / 3 / 5 / 0 | full_new | 174.026 | 27.546 | 5.845x (5.048-6.699) |
| 1 | 1920x1080 / 3 / 5 / 0 | full_reuse | 174.026 | 27.333 | 6.021x (5.322-6.790) |
| 1 | 1920x1080 / 3 / 5 / 0 | resident | 174.026 | 16.297 | 10.089x (8.661-11.700) |
| 1 | 1024x1024 / 3 / 5 / 2 | full_new | 88.715 | 15.501 | 6.072x (5.471-6.702) |
| 1 | 1024x1024 / 3 / 5 / 2 | full_reuse | 88.715 | 15.016 | 6.094x (5.456-6.770) |
| 1 | 1024x1024 / 3 / 5 / 2 | resident | 88.715 | 8.875 | 10.894x (9.519-12.448) |
| 1 | 1024x1024 / 1 / 7 / 0 | full_new | 159.635 | 20.258 | 8.096x (7.544-8.701) |
| 1 | 1024x1024 / 1 / 7 / 0 | full_reuse | 159.635 | 20.171 | 8.135x (7.490-8.858) |
| 1 | 1024x1024 / 1 / 7 / 0 | resident | 159.635 | 12.222 | 13.078x (11.926-14.322) |
| 2 | 65x33 / 1 / 3 / 0 | full_new | 0.226 | 1.210 | 0.184x (0.177-0.191) |
| 2 | 65x33 / 1 / 3 / 0 | full_reuse | 0.226 | 1.182 | 0.183x (0.175-0.191) |
| 2 | 65x33 / 1 / 3 / 0 | resident | 0.226 | 0.250 | 0.906x (0.835-0.976) |
| 2 | 257x193 / 3 / 5 / 0 | full_new | 14.527 | 1.531 | 9.612x (9.223-10.030) |
| 2 | 257x193 / 3 / 5 / 0 | full_reuse | 14.527 | 1.349 | 10.536x (9.879-11.250) |
| 2 | 257x193 / 3 / 5 / 0 | resident | 14.527 | 0.597 | 24.945x (23.216-26.861) |
| 2 | 1024x1024 / 1 / 3 / 0 | full_new | 26.566 | 6.791 | 4.170x (3.718-4.641) |
| 2 | 1024x1024 / 1 / 3 / 0 | full_reuse | 26.566 | 6.388 | 4.414x (3.931-4.919) |
| 2 | 1024x1024 / 1 / 3 / 0 | resident | 26.566 | 1.303 | 22.376x (20.664-24.245) |
| 2 | 1024x1024 / 3 / 5 / 0 | full_new | 70.989 | 17.889 | 3.782x (3.352-4.227) |
| 2 | 1024x1024 / 3 / 5 / 0 | full_reuse | 70.989 | 19.501 | 3.667x (3.315-4.061) |
| 2 | 1024x1024 / 3 / 5 / 0 | resident | 70.989 | 10.518 | 6.751x (6.212-7.311) |
| 2 | 1024x1024 / 4 / 7 / 0 | full_new | 168.628 | 27.710 | 6.084x (5.481-6.766) |
| 2 | 1024x1024 / 4 / 7 / 0 | full_reuse | 168.628 | 27.354 | 6.149x (5.492-6.868) |
| 2 | 1024x1024 / 4 / 7 / 0 | resident | 168.628 | 18.574 | 9.108x (8.294-9.986) |
| 2 | 1920x1080 / 3 / 5 / 0 | full_new | 149.777 | 31.839 | 4.767x (4.142-5.485) |
| 2 | 1920x1080 / 3 / 5 / 0 | full_reuse | 149.777 | 31.336 | 4.687x (4.026-5.451) |
| 2 | 1920x1080 / 3 / 5 / 0 | resident | 149.777 | 14.766 | 8.997x (8.060-10.047) |
| 2 | 1024x1024 / 3 / 5 / 2 | full_new | 74.062 | 20.616 | 3.489x (3.173-3.843) |
| 2 | 1024x1024 / 3 / 5 / 2 | full_reuse | 74.062 | 20.572 | 3.360x (3.003-3.753) |
| 2 | 1024x1024 / 3 / 5 / 2 | resident | 74.062 | 11.833 | 6.383x (6.086-6.730) |
| 2 | 1024x1024 / 1 / 7 / 0 | full_new | 134.236 | 24.047 | 5.435x (4.831-6.075) |
| 2 | 1024x1024 / 1 / 7 / 0 | full_reuse | 134.236 | 24.925 | 5.326x (4.690-6.038) |
| 2 | 1024x1024 / 1 / 7 / 0 | resident | 134.236 | 13.861 | 9.561x (8.655-10.511) |

## Execution receipts

| File | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_vision-88eb52e016b36123.exe` | `cebd33a468b91404e631d23110791295068d12d5e5d0b9abbc64bfffa0417b16` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-d9539d55f12ed728.exe` | `a8bc40f9608fd8b6a53d618cc164d1fd42090c14521c9aede72cf5e7574da568` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral.rs` | `4f5e5339a348dfa9e0d3c7e0812b173db5aa5d7013c2357a1220253503d8a4b3` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral_tests.rs` | `5cab28ea0554e6af1f16f2dc78c7d5935b033529584185e2278c8c35afb76835` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/bilateral_benchmark.rs` | `f0e3cf7a3c7de2acbc3ff0b67ee52835860cf041049254fe573518dc01b2af97` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.rs` | `13f9cfffa1f249006027e05d36f483d1514bbd2ad56eed8e1f0a5e7df78eb334` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.wgsl` | `502000c889e07aba915af5f9d42d2b67fee8558a8aa705cfef49aaaa19584e1c` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round.wgsl` | `0332f72147ee8241df8475a3ed2593ef9ea08f92b9ef2fe06da12c4c8b0b253f` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_tests.rs` | `5fa599544ea421c0bed79a3cc21c38b5f4d3e98fb187255fe3530449500ac5cf` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_benchmark.rs` | `110d3e5280fd03d78bc8a4ef8bb60f8dc75726ab99fc22926026095ebe455c07` |

The frozen test reference and retained production serial body match the exact
merged parent function after whitespace normalization.

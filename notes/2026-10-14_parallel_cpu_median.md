# Exact CPU median selection and bounded output bands

`spatialrust_vision::median_blur` under `imgproc-filter` now selects the
middle element with `select_nth_unstable_by(f64::total_cmp)` for windows
larger than 3x3, avoiding their full sort. The library's optimized tiny full
sort is retained for 1x1/3x3 windows. Sampling, per-component `to_f64` and
`from_f64` conversions and total ordering remain the same. All odd square
apertures, supported scalar types, logical channel counts and five border
modes retain the existing public API. Alpha is an independent component.
There is no GPU placement decision or implicit device transfer.

At least 262,144 logical output components and more than one row allow the
existing native Rayon policy. Work is grouped into at most
`min(current_pool_threads, height)` contiguous output bands. Each band reuses
C sample vectors, with `8 * C * kernel_size^2` bytes of scalar sample capacity
plus vector metadata. Parallel execution writes disjoint slices of one final
result vector. Its safe initialization copies an existing input component,
adding no scalar conversion calls and no second result-sized merge vector.
Scratch scales with active bands and aperture, not image dimensions; extremely
large apertures can still require substantial scratch. Small, empty,
single-row, one-worker and WASM paths do not schedule Rayon work. Small and
single-row calls do not query/initialize the global Rayon pool.

Checked aperture-area/output-size arithmetic precedes allocation. Empty
images retain shape and metadata after parameter validation and avoid scratch
allocation. Allocation failure remains subject to the existing Rust allocator
behavior. No dependency, feature, unsafe boundary or core change is added.

Base: merged PR #192, `5dd1030b01506a7a8d01ed2da7467dfb787c69e9`.
The frozen serial reference is copied from that commit's complete median
function, including full sort, original border fetches and conversions.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/advanced_filter.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median.rs`

Validation and measurement paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_median_smoke.rs`

Four new CPU correctness tests pass. The small-image matrix covers 18,000
conditions and 11,785,000 exact components: channels 1..=4, empty dimensions,
singletons/strips, apertures 1/3/5/7/9, five input profiles, all five borders
and requested worker counts 0/1/2/6. Parallel/threshold/pool/strided-ROI checks
cover another 11,013,219 exact components. Automatic dispatch is checked in
1/2/4-thread pools immediately below/at/above the threshold, including tall
and single-row strips. BGRA/limited-range/straight-alpha metadata and poisoned
row padding are retained/excluded correctly.

Separate u16/f32/f64 tests cover wide values, subnormals, infinities,
signed zeros and signed NaN payloads (including a signaling NaN), with apertures
through 17. Floating comparisons use raw result bits, not float equality or
relaxed tolerances. Independent 3x3 rank and signed-zero cases verify the middle
element. Invalid/overflowing apertures reject identically on empty/nonempty
inputs; worker bounds cover zero, extremes and one-row fallbacks.

The final filter-only suite passes all 32 native tests; all-feature/all-target
vision Clippy rejects warnings, and minimal native/WASM filter/facade builds
pass. The existing `LIGHT_PARALLEL_COMPONENTS` dead-code warning still occurs
in filter-only builds; the dispatch constant is unchanged. Five actual native
GPU median tests pass, including 896 image conditions / 5,248,880 exact bytes,
network/limits/ownership/concurrency/ROI checks; the facade resident-chain smoke
passes. CPU correctness and GPU reference tests also ran before the tiny-sort
adjustment, including the 155-test all-feature vision suite.

An initial all-window selection experiment completed one isolated survey of
14 CPU conditions. At 65x33 gray / 3x3 noise, the automatic caller route regressed:
frozen/current geometric ratio 0.800 (95% bootstrap 0.719-0.859), with medians
0.183/0.218 ms. Its forced-serial ratio was 0.869 (0.830-0.933). This motivated
retaining the tiny full sort. Those measurements are not final performance
results. The superseded second candidate survey was stopped deliberately;
its incomplete raw attempt is retained and excluded. The first candidate
binary/raw/provenance are saved under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/selection-median-candidate-*`.
The parent GPU baseline's frozen binary is also retained locally; final GPU
comparisons below use newly built CPU code, not the historical serial baseline.

## Completed-call measurement boundaries

Final native release surveys use Rust 1.98.1 and 12 available CPU / Rayon pool
workers. CPU cases compare the frozen PR #192 function, forced-one-worker
current code and automatic current code in the same binary. The automatic
CPU benchmark calls the implementation behind the public API's direct
one-line wrapper; the GPU comparison calls the public CPU API itself. The benchmark's
`parallel` record/table route means automatic dispatch; the recorded worker
count shows whether it actually schedules bands. The forced-one-worker helper
is private test code, not a new public execution option.

Each condition has three warmups and 20 paired samples per route. The route
order rotates each round. Fixture/expected construction and returned host
image validation/drop are outside timers. CPU timings include scratch and
result allocation. Fourteen CPU conditions cover gray/RGB noise, 3x3/5x5/9x9,
65x33, 257x193, 1024x1024, 1920x1080, immediately below/at/above the parallel
cutoff, vertical/horizontal strips and RGB all-255 input. Comparisons use the
frozen function compiled into the same final binary, not historical timings
from a differently compiled old binary.

GPU comparisons retain PR #192's 13 conditions and timing/assertion logic;
only the benchmark module description changes from serial to current CPU.
The CPU route now calls the improved public function. Full GPU timings include
explicit upload, median submission, completed readback/unpack and input/output
texture recycling; host validation/drop is excluded as on CPU. Resident GPU
timings include median submission/output checkout and `wait_idle` against an
already uploaded source. Its validation/readback/recycling are outside the
timer, so resident ratios are not full-transfer speedups. Full phase arrays
sum exactly to every full sample; intervals are host submission/wait phases,
not individual device execution durations. GPU algorithms/transfers/receipts
are unchanged by this deliverable.

Controllers require three quiet observations five seconds apart, then monitor
named Cargo/compiler/linker/test/benchmark processes between 500 ms waits. An overlap
discards the whole survey, retaining its raw attempt before retrying. After repeated overlaps the
controller cancels only its own native benchmark as soon as an overlap is
seen, then waits for a new quiet window; foreign work is left running. Own
surveys, build/test jobs and statistical processing are sequential. This
bounded process guard does not prove all applications, OS work or the whole
machine are idle. All 20 samples are retained, with no outlier removal.

Ratios are paired geometric means of old/current or CPU/GPU durations.
Intervals use 10,000 bootstrap resamples of the 20 log ratios and sorted
zero-based endpoints 249/9749. CPU seeds are
`2026101403 + 28*(survey-1) + 2*case_index + stage`, stage 0 serial / 1 automatic.
GPU seeds are `2026101404 + 26*(survey-1) + 2*case_index + stage`, stage 0 full /
1 resident. Tables retain every condition and adverse/inconclusive interval;
values above one favor the current route or GPU. Synthetic fixtures and this
local configuration do not establish universal speedups or a placement policy.
Historical GPU ratios against the serial CPU in
`C:/Users/rsasa/Workspace/SpatialRust/notes/2026-10-14_gpu_median_filter.md`
remain historical evidence; the improved-baseline comparison below supersedes
them for current CPU/GPU placement estimates.

## Results and limits

Two final CPU and two final GPU surveys completed with no detected named-work
overlap and empty benchmark stderr. CPU survey 1 accepted attempt 1 (266.90 s);
survey 2 accepted attempt 3 (259.42 s). Its attempt 1 completed but was discarded
for overlap; attempt 2 overlapped and was stopped deliberately. Both raw attempts
remain excluded. GPU surveys accepted attempt 1 in 62.18/47.66 s. All warm and
timed outputs match their exact references. GPU execution used Vulkan discrete
GPU capabilities and the default 64 MiB image texture pool. Statistical processing
ran after every benchmark completed. Final executable/source hashes still match
those captured before measurement.

For 1920x1080 RGB hash noise / 5x5, old/current automatic CPU paired ratios are
6.716 (6.489-6.955) and 5.882 (5.579-6.202); current CPU medians are
427.069/365.880 ms. Forced-one-worker selection gives 1.522 (1.468-1.588) and
1.486 (1.414-1.558), separating selection from scheduling. At 1024x1024 RGB / 5x5,
automatic ratios are 6.770/6.683; all-255 automatic ratios are 3.802/3.964.
Tiny 65x33 / 3x3 automatic ratios after the sort fallback are 1.005/0.983,
with both intervals crossing one; the initial selection regression is absent
at this precision. There is no claim of universal improvement: the single-row
262145x1 / 3x3 caller path is 0.911 (0.832-1.002) / 0.859 (0.786-0.941).
One-worker vertical-strip and all-255 selection cases also have adverse or
inconclusive intervals. The generic row emitter/selection can add overhead
where scheduling cannot help; those cases remain visible below.

Against the improved public CPU, the 1920x1080 RGB / 5x5 full GPU ratios are
14.811 (13.721-16.102) and 13.568 (12.802-14.520). Full GPU medians are
32.411/31.492 ms. Resident ratios are 29.443/27.128 with their different output
boundary. All twelve non-tiny conditions have full GPU intervals above one
in both surveys. At 65x33 gray / 3x3, full GPU ratios are 0.154 (0.126-0.195) /
0.138 (0.123-0.153), and resident ratios are 0.632/0.608, also below one.
Small-image GPU placement is therefore still unfavorable here. Some absolute
GPU timings differ sharply between surveys (for example 1024x1024 RGB / 5x5
resident medians 9.383/1.247 ms). These repeated local observations support
the stated conditions, not stable cross-machine throughput or an automatic
placement rule.

Raw accepted logs, all rejected/partial attempts, samples, phase arrays,
analysis/controller scripts and provenance are retained under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-cpu-median*`.
Profile 0 is deterministic hash noise, 2 is all-255 and 4 is impulse noise.

## CPU old/current comparisons

| Survey | Dimensions / channels / kernel / profile | Workers | CPU route | Frozen ms | Current ms | Frozen/current paired ratio [95%] |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | 1 | serial | 0.206 | 0.204 | 1.004x (0.965-1.040) |
| 1 | 65x33 / 1 / 3 / 0 | 1 | parallel | 0.206 | 0.207 | 1.005x (0.939-1.084) |
| 1 | 257x193 / 3 / 5 / 0 | 1 | serial | 45.869 | 29.305 | 1.509x (1.427-1.593) |
| 1 | 257x193 / 3 / 5 / 0 | 1 | parallel | 45.869 | 28.764 | 1.583x (1.488-1.673) |
| 1 | 511x513 / 1 / 3 / 0 | 1 | serial | 25.317 | 24.863 | 1.031x (0.966-1.102) |
| 1 | 511x513 / 1 / 3 / 0 | 1 | parallel | 25.317 | 25.886 | 1.000x (0.938-1.068) |
| 1 | 512x512 / 1 / 3 / 0 | 1 | serial | 25.988 | 26.650 | 1.000x (0.927-1.082) |
| 1 | 512x512 / 1 / 3 / 0 | 12 | parallel | 25.988 | 7.388 | 3.776x (3.440-4.132) |
| 1 | 513x512 / 1 / 3 / 0 | 1 | serial | 27.752 | 25.485 | 1.090x (0.985-1.193) |
| 1 | 513x512 / 1 / 3 / 0 | 12 | parallel | 27.752 | 6.972 | 4.088x (3.693-4.558) |
| 1 | 1x262145 / 1 / 3 / 0 | 1 | serial | 25.985 | 31.216 | 0.869x (0.818-0.931) |
| 1 | 1x262145 / 1 / 3 / 0 | 12 | parallel | 25.985 | 6.648 | 3.912x (3.604-4.291) |
| 1 | 262145x1 / 1 / 3 / 0 | 1 | serial | 24.170 | 26.020 | 0.899x (0.802-1.004) |
| 1 | 262145x1 / 1 / 3 / 0 | 1 | parallel | 24.170 | 28.078 | 0.911x (0.832-1.002) |
| 1 | 1024x1024 / 1 / 3 / 0 | 1 | serial | 102.499 | 104.103 | 1.001x (0.924-1.085) |
| 1 | 1024x1024 / 1 / 3 / 0 | 12 | parallel | 102.499 | 27.301 | 4.053x (3.678-4.468) |
| 1 | 1024x1024 / 3 / 3 / 0 | 1 | serial | 241.351 | 230.289 | 1.009x (0.938-1.079) |
| 1 | 1024x1024 / 3 / 3 / 0 | 12 | parallel | 241.351 | 64.872 | 3.761x (3.513-4.029) |
| 1 | 1024x1024 / 1 / 5 / 0 | 1 | serial | 379.609 | 261.323 | 1.424x (1.360-1.495) |
| 1 | 1024x1024 / 1 / 5 / 0 | 12 | parallel | 379.609 | 66.608 | 5.750x (5.399-6.142) |
| 1 | 1024x1024 / 3 / 5 / 0 | 1 | serial | 1334.692 | 835.671 | 1.601x (1.541-1.660) |
| 1 | 1024x1024 / 3 / 5 / 0 | 12 | parallel | 1334.692 | 194.219 | 6.770x (6.498-7.025) |
| 1 | 1024x1024 / 3 / 5 / 2 | 1 | serial | 547.839 | 565.300 | 0.978x (0.942-1.016) |
| 1 | 1024x1024 / 3 / 5 / 2 | 12 | parallel | 547.839 | 142.711 | 3.802x (3.632-3.973) |
| 1 | 1920x1080 / 3 / 5 / 0 | 1 | serial | 2925.173 | 1965.860 | 1.522x (1.468-1.588) |
| 1 | 1920x1080 / 3 / 5 / 0 | 12 | parallel | 2925.173 | 427.069 | 6.716x (6.489-6.955) |
| 1 | 257x193 / 3 / 9 / 0 | 1 | serial | 358.431 | 154.957 | 2.209x (2.119-2.313) |
| 1 | 257x193 / 3 / 9 / 0 | 1 | parallel | 358.431 | 160.436 | 2.082x (1.926-2.229) |
| 2 | 65x33 / 1 / 3 / 0 | 1 | serial | 0.166 | 0.164 | 0.994x (0.933-1.059) |
| 2 | 65x33 / 1 / 3 / 0 | 1 | parallel | 0.166 | 0.164 | 0.983x (0.926-1.023) |
| 2 | 257x193 / 3 / 5 / 0 | 1 | serial | 48.566 | 30.064 | 1.600x (1.486-1.717) |
| 2 | 257x193 / 3 / 5 / 0 | 1 | parallel | 48.566 | 29.663 | 1.642x (1.550-1.733) |
| 2 | 511x513 / 1 / 3 / 0 | 1 | serial | 28.327 | 27.827 | 0.988x (0.897-1.086) |
| 2 | 511x513 / 1 / 3 / 0 | 1 | parallel | 28.327 | 28.320 | 0.968x (0.881-1.061) |
| 2 | 512x512 / 1 / 3 / 0 | 1 | serial | 28.006 | 25.395 | 1.105x (1.030-1.190) |
| 2 | 512x512 / 1 / 3 / 0 | 12 | parallel | 28.006 | 7.534 | 3.225x (2.510-3.879) |
| 2 | 513x512 / 1 / 3 / 0 | 1 | serial | 24.733 | 25.426 | 0.996x (0.906-1.099) |
| 2 | 513x512 / 1 / 3 / 0 | 12 | parallel | 24.733 | 6.444 | 4.250x (3.948-4.593) |
| 2 | 1x262145 / 1 / 3 / 0 | 1 | serial | 27.253 | 29.370 | 0.965x (0.867-1.074) |
| 2 | 1x262145 / 1 / 3 / 0 | 12 | parallel | 27.253 | 6.267 | 4.591x (4.089-5.194) |
| 2 | 262145x1 / 1 / 3 / 0 | 1 | serial | 25.212 | 27.314 | 0.944x (0.888-1.006) |
| 2 | 262145x1 / 1 / 3 / 0 | 1 | parallel | 25.212 | 30.566 | 0.859x (0.786-0.941) |
| 2 | 1024x1024 / 1 / 3 / 0 | 1 | serial | 121.947 | 120.975 | 1.020x (0.960-1.082) |
| 2 | 1024x1024 / 1 / 3 / 0 | 12 | parallel | 121.947 | 31.147 | 3.984x (3.618-4.385) |
| 2 | 1024x1024 / 3 / 3 / 0 | 1 | serial | 288.738 | 288.278 | 0.990x (0.931-1.053) |
| 2 | 1024x1024 / 3 / 3 / 0 | 12 | parallel | 288.738 | 81.221 | 3.599x (3.387-3.797) |
| 2 | 1024x1024 / 1 / 5 / 0 | 1 | serial | 674.388 | 499.856 | 1.344x (1.279-1.412) |
| 2 | 1024x1024 / 1 / 5 / 0 | 12 | parallel | 674.388 | 101.859 | 6.446x (6.015-6.919) |
| 2 | 1024x1024 / 3 / 5 / 0 | 1 | serial | 1373.447 | 872.225 | 1.589x (1.537-1.643) |
| 2 | 1024x1024 / 3 / 5 / 0 | 12 | parallel | 1373.447 | 199.575 | 6.683x (6.293-7.066) |
| 2 | 1024x1024 / 3 / 5 / 2 | 1 | serial | 573.866 | 590.813 | 0.939x (0.900-0.979) |
| 2 | 1024x1024 / 3 / 5 / 2 | 12 | parallel | 573.866 | 140.620 | 3.964x (3.703-4.238) |
| 2 | 1920x1080 / 3 / 5 / 0 | 1 | serial | 2137.779 | 1434.344 | 1.486x (1.414-1.558) |
| 2 | 1920x1080 / 3 / 5 / 0 | 12 | parallel | 2137.779 | 365.880 | 5.882x (5.579-6.202) |
| 2 | 257x193 / 3 / 9 / 0 | 1 | serial | 211.137 | 96.610 | 2.228x (2.125-2.339) |
| 2 | 257x193 / 3 / 9 / 0 | 1 | parallel | 211.137 | 98.738 | 2.106x (1.996-2.225) |

## GPU compared with improved public CPU

| Survey | Dimensions / channels / kernel / profile | Stage | CPU ms | GPU ms | CPU/GPU paired ratio [95%] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | full_gpu | 0.379 | 2.695 | 0.154x (0.126-0.195) |
| 1 | 65x33 / 1 / 3 / 0 | resident | 0.379 | 0.539 | 0.632x (0.493-0.827) |
| 1 | 257x193 / 3 / 5 / 0 | full_gpu | 40.574 | 2.270 | 18.984x (17.582-20.362) |
| 1 | 257x193 / 3 / 5 / 0 | resident | 40.574 | 0.510 | 90.590x (79.751-102.069) |
| 1 | 1024x1024 / 1 / 3 / 0 | full_gpu | 37.695 | 9.040 | 4.183x (3.996-4.415) |
| 1 | 1024x1024 / 1 / 3 / 0 | resident | 37.695 | 0.725 | 51.739x (48.207-55.563) |
| 1 | 1024x1024 / 3 / 3 / 0 | full_gpu | 115.255 | 14.252 | 7.729x (7.174-8.230) |
| 1 | 1024x1024 / 3 / 3 / 0 | resident | 115.255 | 2.464 | 41.834x (37.295-46.028) |
| 1 | 1024x1024 / 4 / 3 / 0 | full_gpu | 161.176 | 14.126 | 10.665x (9.672-11.791) |
| 1 | 1024x1024 / 4 / 3 / 0 | resident | 161.176 | 3.735 | 39.051x (35.412-42.879) |
| 1 | 1024x1024 / 1 / 5 / 0 | full_gpu | 92.221 | 17.462 | 5.405x (5.201-5.609) |
| 1 | 1024x1024 / 1 / 5 / 0 | resident | 92.221 | 9.217 | 10.309x (9.927-10.733) |
| 1 | 1024x1024 / 3 / 5 / 0 | full_gpu | 251.870 | 19.426 | 12.684x (11.108-14.302) |
| 1 | 1024x1024 / 3 / 5 / 0 | resident | 251.870 | 9.383 | 30.301x (26.600-35.008) |
| 1 | 1024x1024 / 4 / 5 / 0 | full_gpu | 342.941 | 19.239 | 17.265x (16.249-18.249) |
| 1 | 1024x1024 / 4 / 5 / 0 | resident | 342.941 | 9.337 | 34.528x (32.217-36.885) |
| 1 | 1920x1080 / 1 / 3 / 0 | full_gpu | 73.493 | 20.437 | 3.443x (3.247-3.634) |
| 1 | 1920x1080 / 1 / 3 / 0 | resident | 73.493 | 5.285 | 13.562x (12.548-14.674) |
| 1 | 1920x1080 / 3 / 5 / 0 | full_gpu | 501.624 | 32.411 | 14.811x (13.721-16.102) |
| 1 | 1920x1080 / 3 / 5 / 0 | resident | 501.624 | 16.731 | 29.443x (27.079-32.063) |
| 1 | 1024x1024 / 1 / 5 / 2 | full_gpu | 95.529 | 18.035 | 5.271x (4.857-5.714) |
| 1 | 1024x1024 / 1 / 5 / 2 | resident | 95.529 | 9.313 | 10.357x (9.548-11.239) |
| 1 | 1024x1024 / 4 / 5 / 2 | full_gpu | 180.368 | 17.507 | 10.612x (10.246-11.028) |
| 1 | 1024x1024 / 4 / 5 / 2 | resident | 180.368 | 9.232 | 20.044x (19.377-20.774) |
| 1 | 1024x1024 / 3 / 5 / 4 | full_gpu | 156.604 | 17.681 | 8.829x (8.453-9.279) |
| 1 | 1024x1024 / 3 / 5 / 4 | resident | 156.604 | 9.290 | 17.007x (16.219-17.936) |
| 2 | 65x33 / 1 / 3 / 0 | full_gpu | 0.307 | 2.118 | 0.138x (0.123-0.153) |
| 2 | 65x33 / 1 / 3 / 0 | resident | 0.307 | 0.420 | 0.608x (0.472-0.751) |
| 2 | 257x193 / 3 / 5 / 0 | full_gpu | 36.769 | 1.909 | 19.200x (17.686-20.805) |
| 2 | 257x193 / 3 / 5 / 0 | resident | 36.769 | 0.421 | 88.336x (78.236-99.656) |
| 2 | 1024x1024 / 1 / 3 / 0 | full_gpu | 34.241 | 7.261 | 4.610x (4.303-4.948) |
| 2 | 1024x1024 / 1 / 3 / 0 | resident | 34.241 | 0.650 | 51.944x (48.012-56.016) |
| 2 | 1024x1024 / 3 / 3 / 0 | full_gpu | 71.604 | 10.080 | 7.067x (6.625-7.545) |
| 2 | 1024x1024 / 3 / 3 / 0 | resident | 71.604 | 2.081 | 36.181x (32.852-40.036) |
| 2 | 1024x1024 / 4 / 3 / 0 | full_gpu | 82.017 | 10.834 | 7.445x (7.136-7.763) |
| 2 | 1024x1024 / 4 / 3 / 0 | resident | 82.017 | 3.292 | 27.413x (25.160-30.048) |
| 2 | 1024x1024 / 1 / 5 / 0 | full_gpu | 88.603 | 7.528 | 9.860x (8.220-11.655) |
| 2 | 1024x1024 / 1 / 5 / 0 | resident | 88.603 | 1.236 | 43.560x (28.731-61.525) |
| 2 | 1024x1024 / 3 / 5 / 0 | full_gpu | 183.277 | 7.713 | 23.886x (22.957-24.820) |
| 2 | 1024x1024 / 3 / 5 / 0 | resident | 183.277 | 1.247 | 142.410x (134.547-150.556) |
| 2 | 1024x1024 / 4 / 5 / 0 | full_gpu | 232.463 | 17.159 | 14.556x (13.751-15.504) |
| 2 | 1024x1024 / 4 / 5 / 0 | resident | 232.463 | 9.235 | 30.477x (27.149-34.723) |
| 2 | 1920x1080 / 1 / 3 / 0 | full_gpu | 59.122 | 19.008 | 3.111x (2.969-3.241) |
| 2 | 1920x1080 / 1 / 3 / 0 | resident | 59.122 | 5.200 | 11.262x (10.820-11.726) |
| 2 | 1920x1080 / 3 / 5 / 0 | full_gpu | 406.637 | 31.492 | 13.568x (12.802-14.520) |
| 2 | 1920x1080 / 3 / 5 / 0 | resident | 406.637 | 16.674 | 27.128x (24.502-30.648) |
| 2 | 1024x1024 / 1 / 5 / 2 | full_gpu | 80.030 | 9.445 | 7.993x (7.344-8.700) |
| 2 | 1024x1024 / 1 / 5 / 2 | resident | 80.030 | 1.391 | 36.850x (28.641-46.867) |
| 2 | 1024x1024 / 4 / 5 / 2 | full_gpu | 194.440 | 10.119 | 15.815x (12.295-19.408) |
| 2 | 1024x1024 / 4 / 5 / 2 | resident | 194.440 | 2.825 | 57.943x (35.681-88.097) |
| 2 | 1024x1024 / 3 / 5 / 4 | full_gpu | 154.748 | 8.610 | 17.539x (16.500-18.529) |
| 2 | 1024x1024 / 3 / 5 / 4 | resident | 154.748 | 1.730 | 80.072x (66.364-94.965) |

## Full GPU host intervals

| Survey | Dimensions / channels / kernel / profile | Upload submit ms | Kernel submit ms | Readback/wait ms | Recycle ms | Full ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | 0.063 | 0.376 | 2.231 | 0.005 | 2.695 |
| 1 | 257x193 / 3 / 5 / 0 | 0.123 | 0.248 | 1.923 | 0.004 | 2.270 |
| 1 | 1024x1024 / 1 / 3 / 0 | 2.683 | 0.360 | 5.880 | 0.005 | 9.040 |
| 1 | 1024x1024 / 3 / 3 / 0 | 3.709 | 0.355 | 10.431 | 0.006 | 14.252 |
| 1 | 1024x1024 / 4 / 3 / 0 | 3.124 | 0.367 | 10.644 | 0.006 | 14.126 |
| 1 | 1024x1024 / 1 / 5 / 0 | 2.417 | 0.308 | 14.901 | 0.004 | 17.462 |
| 1 | 1024x1024 / 3 / 5 / 0 | 3.026 | 0.359 | 16.028 | 0.005 | 19.426 |
| 1 | 1024x1024 / 4 / 5 / 0 | 2.718 | 0.336 | 15.854 | 0.005 | 19.239 |
| 1 | 1920x1080 / 1 / 3 / 0 | 3.877 | 0.302 | 15.790 | 0.005 | 20.437 |
| 1 | 1920x1080 / 3 / 5 / 0 | 4.540 | 0.349 | 27.918 | 0.004 | 32.411 |
| 1 | 1024x1024 / 1 / 5 / 2 | 2.429 | 0.314 | 15.415 | 0.005 | 18.035 |
| 1 | 1024x1024 / 4 / 5 / 2 | 2.162 | 0.278 | 14.904 | 0.004 | 17.507 |
| 1 | 1024x1024 / 3 / 5 / 4 | 2.314 | 0.264 | 15.083 | 0.004 | 17.681 |
| 2 | 65x33 / 1 / 3 / 0 | 0.050 | 0.277 | 1.766 | 0.004 | 2.118 |
| 2 | 257x193 / 3 / 5 / 0 | 0.119 | 0.241 | 1.572 | 0.003 | 1.909 |
| 2 | 1024x1024 / 1 / 3 / 0 | 2.314 | 0.329 | 4.611 | 0.004 | 7.261 |
| 2 | 1024x1024 / 3 / 3 / 0 | 2.117 | 0.267 | 7.737 | 0.004 | 10.080 |
| 2 | 1024x1024 / 4 / 3 / 0 | 2.091 | 0.256 | 8.622 | 0.004 | 10.834 |
| 2 | 1024x1024 / 1 / 5 / 0 | 2.121 | 0.289 | 4.981 | 0.004 | 7.528 |
| 2 | 1024x1024 / 3 / 5 / 0 | 2.414 | 0.293 | 4.969 | 0.005 | 7.713 |
| 2 | 1024x1024 / 4 / 5 / 0 | 1.914 | 0.253 | 14.923 | 0.005 | 17.159 |
| 2 | 1920x1080 / 1 / 3 / 0 | 3.493 | 0.264 | 15.176 | 0.004 | 19.008 |
| 2 | 1920x1080 / 3 / 5 / 0 | 3.929 | 0.269 | 27.246 | 0.005 | 31.492 |
| 2 | 1024x1024 / 1 / 5 / 2 | 2.209 | 0.311 | 6.727 | 0.005 | 9.445 |
| 2 | 1024x1024 / 4 / 5 / 2 | 2.104 | 0.285 | 7.950 | 0.005 | 10.119 |
| 2 | 1024x1024 / 3 / 5 / 4 | 2.249 | 0.259 | 5.956 | 0.004 | 8.610 |

## Frozen execution receipts

| File | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_vision-88eb52e016b36123.exe` | `f793927a9220124be6eea1a4c7b51554f7ce6a4a26e469f9f97a1aa036dff72f` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-d9539d55f12ed728.exe` | `4fdbe9240624c7b11133c12ed60919a7e0b52591ee1961e67320c03fb75b3f9b` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-cpu-median-previous-gpu.exe` | `b40164b11860fac366e118c52f5a948940e3354cf78ee8f8e010a10fbdd0d9ab` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median.rs` | `b080b32152b15c18b7a79f9b502573f8236744f1f935c2599c7142702868c97d` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median_tests.rs` | `ef1e46e88bbd15c880b6861ff1e3565c540dae5d4ab99d451b5044f8103f6d86` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/median_benchmark.rs` | `c4e6c9e195eae49d6c4ad07ae2f2e63b25fb8899f3c2df0df3a04f2efc87d053` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_benchmark.rs` | `6bf27779593d684510eb163ce41079dfc36a0cfbd0bab4594f9c28aa9872fad6` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/advanced_filter.rs` | `d7f834bb8c9a18a3b9c396144f80f23d94a6cb7a4c12469b0abe90d1d2e33a57` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/dispatch.rs` | `434c6ca04b713f90cc0c39c94d0f9ef38dfe52a458b19b9b0a69d9c7825e62d3` |

Frozen parent median body SHA-256: `3bd6872745368107b7689c366b5a1c6bd57c36a5aa7d435c3440b4992dd7053a`. Its
whitespace-normalized body was checked against the exact merged parent source.

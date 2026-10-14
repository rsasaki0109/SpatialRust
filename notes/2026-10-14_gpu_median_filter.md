# Resident GPU median filtering

`median_blur_gpu(runtime, source, kernel_size, border)` adds caller-selected
GPU median filtering under `gpu-image`. It accepts 3x3/5x5 square neighborhoods
and the existing `GpuImageBorder::{Replicate, ConstantZero}`. All 1..=4 logical
u8 components, including alpha, are filtered independently. Metadata and
logical channel count are retained. Generic CPU median placement, dtypes,
larger kernels and other border modes are unchanged.

The source and owned output use the existing `rgba8uint` representation.
One ordered compute pass selects exact byte medians using static vector
min/max comparisons. A host-generated bitonic network is pruned backwards
from the required median output. Nine/25 input vectors are padded to 16/32
with 255, the largest byte; the selected index is four/12. The emitted
networks contain 63/191 comparator pairs before shader compiler optimization.
Private values use named static variables rather than dynamically indexed
arrays. This is a full square median, not successive horizontal/vertical
medians, which would be a different operator.

No image-sized scratch buffer/texture is allocated. Fixed per-invocation
register/private state remains driver-managed, so this is not a claim about
driver register spills or total process/device memory. Each pass records
four physical destination bytes per pixel in the existing receipt; this is
a destination footprint, not total sampling bandwidth. A 16-byte uniform
holds dimensions and border mode; control data is excluded from image payload
receipts. Explicit upload and readback each retain four physical bytes per
pixel, excluding staging-row padding.

Runtime ownership, supported kernels, positive dimensions, texture/signed
coordinate bounds, both dispatch axes, workgroup shape/invocations, sampled
and storage texture bindings, uniform size/buffer size and group/binding limits
are checked before output allocation or lazy cache initialization. This
texture-only kernel imposes no storage-buffer binding or image-sized buffer
limit. Kernel pipelines initialize independently in one runtime-owned family.
Production filtering submits work without a host completion wait or payload
upload/readback. Owned results can outlive inputs and compose with existing
gray, Sobel, Gaussian and AI tensor stages. Pool return remains caller-explicit.

Base: merged PR #191, `d20c5e4a7cd5de3dfe6c5761542acd7a890123dc`.
No new dependency, feature, unsafe boundary or core change is introduced.

Production and API paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_params.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/mod.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/mod.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/lib.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/runtime.rs`

Validation and measurement paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_median_smoke.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/advanced_filter.rs`

Five new native tests pass, including 896 CPU byte-exact image conditions and
5,248,880 logical component comparisons. Eight shapes cover singleton/strips,
small neighborhoods, workgroup tails, non-aligned rows and 129x127 images.
Seven profiles cover hash noise, zeros, 255, checkerboard, impulses, ramps and
constant per-channel colors. Both kernels/borders and all four logical channel
counts are tested. Full physical RGBA readback verifies unused components
remain zero. Receipts match upload, one output footprint and explicit readback.

Independent scalar network tests cover all 512 binary 3x3 windows, 200,000
random byte windows against a separate byte sort and all 256 constant bytes
for each kernel. Limit tests cover rejected kernels and dimensions, twelve
individual deficient capabilities, exact dispatch boundaries, the largest
signed-safe coordinates, 16-byte uniform bounds and zero storage-buffer limits.
Invalid public calls and foreign runtimes leave receipts, texture pools and
pipeline initialization unchanged. Four cold concurrent calls initialize both
kernels while keeping outputs owned and independent.

Strided ROI tests preserve BGRA, limited-range and straight-alpha metadata;
queued outputs remain correct after the input is recycled. A median -> gray ->
Sobel -> resident CHW tensor chain has no intermediate host readback.
The facade smoke test preserves constant RGB color and runs the same resident
chain. All 99 native image tests pass, including existing Canny/Gaussian/gray/
Sobel/morphology/CLAHE/histogram/transfer regressions. GPU all-feature/all-target
Clippy, CPU-only GPU check, native facade and WASM facade build, rustfmt and
diff checks pass. Correctness uses exact bytes, without widened tolerances.

Two isolated native release surveys use Vulkan discrete GPU execution, 12
available CPU workers, Rust 1.98.1 and the default 64 MiB texture pool budget.
The unchanged `spatialrust_vision::median_blur` is a generic serial f64-sorting
CPU baseline. These ratios do not compare against an optimized/parallel CPU
median or external implementation. CPU placement remains caller-selected.
Large explicit transfer packing/unpacking uses the existing bounded workers.

Each survey covers 13 conditions: hash-noise gray/RGB/RGBA, 3x3/5x5,
65x33, 257x193, 1024x1024 and 1920x1080, plus all-255 and impulse profiles.
Fixtures and one CPU expected result are outside timers. Three warmups precede
20 paired samples for each of three routes, rotated each round:

- CPU: the current public generic median, including result/scratch allocation.
  Returned host result validation/drop is outside the timer.
- Full GPU: actual upload, median submission, readback/completion wait and
  logical-channel unpack, then caller-explicit input/output texture recycling.
  Returned host result validation/drop is outside the timer, as on CPU.
- Resident GPU: preuploaded source, median submission/output checkout and
  `wait_idle`. Explicit readback/validation and texture recycling are outside
  this timer. This measures completed work, not submission time, but has a
  different transfer/output boundary from the full CPU/full GPU calls.

Both GPU routes match the CPU expected bytes for every warm/timed call.
Full GPU receipts retain 4*pixels H2D, one 4*pixels destination footprint and
4*pixels D2H. Host phase samples include upload submission, kernel submission,
readback/wait/unpack and recycling; their sum equals each full sample. Pending
GPU upload/kernel execution may complete during the wait, so these are host
intervals rather than isolated device pass/transfer durations.

The first survey attempt is discarded in full for observed other-process
overlap, with its raw log retained. Accepted survey one is attempt two
(193.28 seconds), and survey two is attempt one (216.38 seconds), each without
observed overlap. The controller requires three quiet observations spaced
five seconds apart and monitors named Cargo/compiler/linker/test/benchmark
process classes while running. Any overlap discards the whole survey. This
does not prove all machine activity is idle or remove frequency/scheduler
variation; absolute times vary appreciably between surveys. No foreign work
was stopped, and statistical processing began only after all measurements.

All twelve conditions above 65x33 have full-GPU intervals above one in both
surveys. Million-pixel-or-larger complete ratios range from 13.256 to 157.814x
relative to the current serial CPU baseline. The 1920x1080 RGB 5x5 noise case
has CPU medians 2028.068/2207.995 ms versus full-GPU 12.452/13.912 ms, with
paired geometric ratios 157.814x [145.735, 170.051] and
148.898x [134.887, 162.721]. Its resident medians are 2.719/3.001 ms;
resident ratios must not be presented as transfer-inclusive speedups.

The adverse 65x33 gray 3x3 full ratios are 0.188x [0.175, 0.204] and
0.098x [0.065, 0.148]. Its resident interval includes one in survey one and
is below one in survey two. GPU startup/transfer/completion overhead can
dominate such small workloads; this API adds no automatic placement policy.
Flat and impulse profiles also improve at the tested large size, but all
condition tables and variability are retained rather than asserting one
universal crossover or speedup.

Intervals bootstrap 20 paired log CPU/GPU ratios with 10,000 resamples and
sorted endpoints 249/9749. They are unadjusted across multiple conditions.
Seed is 2026101402 + 26*(survey-1) + 2*case_index + stage, where stage is
full GPU 0 or resident 1. Ratios use geometric means and times use separate
medians; median phase sums need not equal the median full time.

Identities:

- Native test/benchmark executable SHA-256: `B40164B11860FAC366E118C52F5A948940E3354CF78EE8F8E010A10FBDD0D9AB`
- Benchmark source SHA-256: `ABDB7CC88074E30CCF8C013FD1FDC5C9B698C822C9E251275502F40B8C6DAB65`
- Unchanged CPU source SHA-256: `26955FAA400B51774B00D1DB3E443EA04B796ECC82C2B2B7A11EC0706D2BB4D6`
- Facade executable SHA-256: `6475257F5DAC20FCAC3D41B5C35FD2F0C62695212273AE36FD50873E363E2ACE`

Raw samples, rejected/accepted logs, phase arrays, overlap controller and
analysis remain local under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:

- `gpu-median-filter-provenance.json`
- `run-gpu-median-filter-survey.ps1`
- `analyze-gpu-median-filter.py`
- `gpu-median-filter-survey-1-attempt-1.stdout.log` / `.stderr.log` (discarded)
- `gpu-median-filter-benchmark-1.log` / `-2.log` (accepted)
- `gpu-median-filter-measurements.json`
- `gpu-median-filter-table.md` / `gpu-median-filter-phase-table.md`
- `gpu-median-filter-image-regressions.log` and named facade/feature check logs

Exact ignored benchmark:
`image::kernels::median::tests::benchmark::completed_median_calls`.
GPU-native tests and benchmark require a suitable native adapter; ordinary
tests do not run the benchmark. Reports describe backend and measurement
conditions without naming the local adapter. No shader/production change
occurred between correctness checks and accepted measurements.

## Completed call comparisons

| Survey | Dimensions / channels / kernel / profile | Stage | CPU ms | GPU ms | CPU/GPU paired ratio [95%] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | full_gpu | 0.261 | 1.549 | 0.188x (0.175-0.204) |
| 1 | 65x33 / 1 / 3 / 0 | resident | 0.261 | 0.291 | 0.968x (0.899-1.048) |
| 1 | 257x193 / 3 / 5 / 0 | full_gpu | 39.526 | 1.477 | 26.856x (24.235-29.538) |
| 1 | 257x193 / 3 / 5 / 0 | resident | 39.526 | 0.349 | 111.240x (98.631-124.624) |
| 1 | 1024x1024 / 1 / 3 / 0 | full_gpu | 87.917 | 6.517 | 13.256x (12.334-14.134) |
| 1 | 1024x1024 / 1 / 3 / 0 | resident | 87.917 | 0.741 | 115.608x (102.552-128.861) |
| 1 | 1024x1024 / 3 / 3 / 0 | full_gpu | 196.964 | 6.842 | 27.021x (24.232-29.647) |
| 1 | 1024x1024 / 3 / 3 / 0 | resident | 196.964 | 0.904 | 193.292x (150.181-236.832) |
| 1 | 1024x1024 / 4 / 3 / 0 | full_gpu | 243.711 | 6.180 | 38.374x (34.472-41.796) |
| 1 | 1024x1024 / 4 / 3 / 0 | resident | 243.711 | 0.686 | 300.189x (243.393-358.137) |
| 1 | 1024x1024 / 1 / 5 / 0 | full_gpu | 367.111 | 6.813 | 53.943x (50.696-57.127) |
| 1 | 1024x1024 / 1 / 5 / 0 | resident | 367.111 | 1.262 | 259.704x (229.438-290.502) |
| 1 | 1024x1024 / 3 / 5 / 0 | full_gpu | 941.099 | 6.732 | 131.177x (115.344-144.227) |
| 1 | 1024x1024 / 3 / 5 / 0 | resident | 941.099 | 1.350 | 621.290x (548.795-698.135) |
| 1 | 1024x1024 / 4 / 5 / 0 | full_gpu | 1151.254 | 7.780 | 148.909x (136.279-161.858) |
| 1 | 1024x1024 / 4 / 5 / 0 | resident | 1151.254 | 1.854 | 615.765x (519.933-725.335) |
| 1 | 1920x1080 / 1 / 3 / 0 | full_gpu | 184.449 | 10.950 | 16.323x (14.928-17.756) |
| 1 | 1920x1080 / 1 / 3 / 0 | resident | 184.449 | 1.062 | 149.579x (110.219-190.303) |
| 1 | 1920x1080 / 3 / 5 / 0 | full_gpu | 2028.068 | 12.452 | 157.814x (145.735-170.051) |
| 1 | 1920x1080 / 3 / 5 / 0 | resident | 2028.068 | 2.719 | 757.695x (634.927-905.864) |
| 1 | 1024x1024 / 1 / 5 / 2 | full_gpu | 310.619 | 10.008 | 31.013x (28.653-33.551) |
| 1 | 1024x1024 / 1 / 5 / 2 | resident | 310.619 | 1.288 | 223.507x (200.181-248.329) |
| 1 | 1024x1024 / 4 / 5 / 2 | full_gpu | 726.605 | 9.516 | 71.742x (61.755-82.595) |
| 1 | 1024x1024 / 4 / 5 / 2 | resident | 726.605 | 2.081 | 332.489x (265.898-406.861) |
| 1 | 1024x1024 / 3 / 5 / 4 | full_gpu | 1072.894 | 10.787 | 89.881x (70.640-109.773) |
| 1 | 1024x1024 / 3 / 5 / 4 | resident | 1072.894 | 1.739 | 717.811x (604.897-838.883) |
| 2 | 65x33 / 1 / 3 / 0 | full_gpu | 0.569 | 7.007 | 0.098x (0.065-0.148) |
| 2 | 65x33 / 1 / 3 / 0 | resident | 0.569 | 1.006 | 0.496x (0.308-0.779) |
| 2 | 257x193 / 3 / 5 / 0 | full_gpu | 92.507 | 2.541 | 34.516x (31.651-37.583) |
| 2 | 257x193 / 3 / 5 / 0 | resident | 92.507 | 0.724 | 125.539x (104.021-149.475) |
| 2 | 1024x1024 / 1 / 3 / 0 | full_gpu | 152.428 | 9.037 | 17.376x (16.457-18.410) |
| 2 | 1024x1024 / 1 / 3 / 0 | resident | 152.428 | 0.925 | 169.675x (153.841-186.998) |
| 2 | 1024x1024 / 3 / 3 / 0 | full_gpu | 299.972 | 8.828 | 36.734x (33.523-40.598) |
| 2 | 1024x1024 / 3 / 3 / 0 | resident | 299.972 | 0.821 | 362.131x (315.513-414.462) |
| 2 | 1024x1024 / 4 / 3 / 0 | full_gpu | 413.961 | 10.062 | 44.299x (40.594-48.642) |
| 2 | 1024x1024 / 4 / 3 / 0 | resident | 413.961 | 0.816 | 473.694x (407.893-552.753) |
| 2 | 1024x1024 / 1 / 5 / 0 | full_gpu | 567.987 | 8.243 | 67.702x (62.317-74.275) |
| 2 | 1024x1024 / 1 / 5 / 0 | resident | 567.987 | 1.748 | 343.379x (304.855-382.535) |
| 2 | 1024x1024 / 3 / 5 / 0 | full_gpu | 1172.581 | 8.679 | 145.756x (136.096-158.302) |
| 2 | 1024x1024 / 3 / 5 / 0 | resident | 1172.581 | 1.904 | 750.010x (651.870-872.743) |
| 2 | 1024x1024 / 4 / 5 / 0 | full_gpu | 1656.181 | 15.543 | 129.603x (113.817-148.803) |
| 2 | 1024x1024 / 4 / 5 / 0 | resident | 1656.181 | 7.279 | 358.264x (259.046-501.112) |
| 2 | 1920x1080 / 1 / 3 / 0 | full_gpu | 238.303 | 12.080 | 19.151x (17.113-21.166) |
| 2 | 1920x1080 / 1 / 3 / 0 | resident | 238.303 | 1.141 | 178.792x (137.480-221.112) |
| 2 | 1920x1080 / 3 / 5 / 0 | full_gpu | 2207.995 | 13.912 | 148.898x (134.887-162.721) |
| 2 | 1920x1080 / 3 / 5 / 0 | resident | 2207.995 | 3.001 | 561.383x (447.431-691.823) |
| 2 | 1024x1024 / 1 / 5 / 2 | full_gpu | 288.455 | 8.888 | 33.678x (29.807-37.743) |
| 2 | 1024x1024 / 1 / 5 / 2 | resident | 288.455 | 1.644 | 142.055x (114.903-173.074) |
| 2 | 1024x1024 / 4 / 5 / 2 | full_gpu | 631.963 | 7.225 | 86.031x (81.499-90.995) |
| 2 | 1024x1024 / 4 / 5 / 2 | resident | 631.963 | 1.437 | 401.871x (350.867-456.635) |
| 2 | 1024x1024 / 3 / 5 / 4 | full_gpu | 816.120 | 7.070 | 110.546x (103.888-117.159) |
| 2 | 1024x1024 / 3 / 5 / 4 | resident | 816.120 | 1.807 | 518.461x (446.676-596.421) |

## Full GPU host intervals

| Survey | Dimensions / channels / kernel / profile | Upload submit ms | Kernel submit ms | Readback/wait ms | Recycle ms | Full ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | 0.023 | 0.143 | 1.385 | 0.002 | 1.549 |
| 1 | 257x193 / 3 / 5 / 0 | 0.086 | 0.158 | 1.226 | 0.002 | 1.477 |
| 1 | 1024x1024 / 1 / 3 / 0 | 1.817 | 0.263 | 4.428 | 0.003 | 6.517 |
| 1 | 1024x1024 / 3 / 3 / 0 | 1.967 | 0.254 | 4.670 | 0.003 | 6.842 |
| 1 | 1024x1024 / 4 / 3 / 0 | 1.698 | 0.232 | 4.179 | 0.002 | 6.180 |
| 1 | 1024x1024 / 1 / 5 / 0 | 1.726 | 0.264 | 4.543 | 0.003 | 6.813 |
| 1 | 1024x1024 / 3 / 5 / 0 | 1.868 | 0.266 | 4.644 | 0.003 | 6.732 |
| 1 | 1024x1024 / 4 / 5 / 0 | 1.778 | 0.303 | 5.833 | 0.003 | 7.780 |
| 1 | 1920x1080 / 1 / 3 / 0 | 3.006 | 0.233 | 7.570 | 0.003 | 10.950 |
| 1 | 1920x1080 / 3 / 5 / 0 | 3.204 | 0.288 | 9.121 | 0.004 | 12.452 |
| 1 | 1024x1024 / 1 / 5 / 2 | 2.427 | 0.303 | 6.966 | 0.004 | 10.008 |
| 1 | 1024x1024 / 4 / 5 / 2 | 2.143 | 0.329 | 6.708 | 0.005 | 9.516 |
| 1 | 1024x1024 / 3 / 5 / 4 | 2.522 | 0.329 | 7.751 | 0.004 | 10.787 |
| 2 | 65x33 / 1 / 3 / 0 | 0.067 | 0.417 | 6.472 | 0.006 | 7.007 |
| 2 | 257x193 / 3 / 5 / 0 | 0.151 | 0.315 | 2.061 | 0.004 | 2.541 |
| 2 | 1024x1024 / 1 / 3 / 0 | 2.463 | 0.314 | 5.962 | 0.004 | 9.037 |
| 2 | 1024x1024 / 3 / 3 / 0 | 2.482 | 0.311 | 5.642 | 0.005 | 8.828 |
| 2 | 1024x1024 / 4 / 3 / 0 | 2.487 | 0.354 | 6.804 | 0.005 | 10.062 |
| 2 | 1024x1024 / 1 / 5 / 0 | 2.071 | 0.310 | 5.814 | 0.004 | 8.243 |
| 2 | 1024x1024 / 3 / 5 / 0 | 1.993 | 0.281 | 6.370 | 0.004 | 8.679 |
| 2 | 1024x1024 / 4 / 5 / 0 | 1.785 | 0.277 | 13.391 | 0.004 | 15.543 |
| 2 | 1920x1080 / 1 / 3 / 0 | 3.213 | 0.256 | 8.474 | 0.004 | 12.080 |
| 2 | 1920x1080 / 3 / 5 / 0 | 3.335 | 0.261 | 10.125 | 0.004 | 13.912 |
| 2 | 1024x1024 / 1 / 5 / 2 | 1.892 | 0.302 | 6.330 | 0.004 | 8.888 |
| 2 | 1024x1024 / 4 / 5 / 2 | 1.721 | 0.250 | 5.130 | 0.004 | 7.225 |
| 2 | 1024x1024 / 3 / 5 / 4 | 1.813 | 0.237 | 5.032 | 0.003 | 7.070 |

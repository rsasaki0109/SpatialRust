# Resident GPU Gaussian filtering

Date: 2026-10-13. Named slice: weighted image filtering on device, preserving
the existing specialized CPU u8 Gaussian arithmetic and explicit image transfers.

## API and execution

`gaussian_blur_gpu(runtime, source, GpuGaussianBlurOptions)` consumes a borrowed
resident `GpuImage` and returns an independently owned image. It is exported by
`spatialrust_gpu` and available through the facade's `spatialrust::gpu` module
under `gpu-image`. Default options use 5x5 kernels, sigmas 1.2, and Reflect101.
Widths/heights independently accept 3, 5 or 7; sigma values must be finite and
positive. `GpuGaussianBorder` provides Replicate, Reflect, Reflect101, Wrap and
Constant([u8;4]). Constant components beyond the logical channels are ignored.
The output retains dimensions, logical 1..=4 channels and image metadata.
It can feed subsequent resident gray/Sobel/morphology/AI operations directly.

The host prepares only two tiny coefficient vectors in binary64, using the
same normalization, rounding and center correction as
`spatialrust_vision::gaussian_blur_u8`. The weight sum is 256 for a 3/5 call;
when either axis uses 7, BOTH axes use sum 32768. Horizontal weighted sums
are computed on GPU and remain unrounded in private u32 scratch. Vertical
weighted sums execute in a second GPU pass within the same submission. Q8
totals fit u32. Q15 products/totals require up to 38 bits: two unsigned u32
limbs preserve them, then a single half-up rounding produces the final u8.
GPU computation uses integer loads, weights and accumulations throughout;
shader floating-point rounding or native GPU f64 support are unnecessary.

Bounded flat workgroups use at most 4096 groups with grid-stride loops. Tail
advancement tests the remaining pixel count before adding a stride. Reflection
handles one/two-pixel strips with repeated mapping. Wrapping maps negative
coordinates with unsigned nonnegative remainders, covered by CPU strip tests.
Output components outside the logical channel count are zero.

For N pixels and C logical channels, private scratch is 4*N*C bytes and physical
output writes are 4*N bytes. The input receipt is cloned and gains the named
`gaussian_horizontal_gpu` and `gaussian_blur_gpu` device stages. There is no image
payload readback, repeated image upload, intermediate CPU arithmetic or host
wait inside the filter. The host-to-device upload of 128 uniform control bytes
(including the coefficient vectors) is excluded from payload counters under
the existing image accounting convention. An explicit final texture readback
uses 256-byte-padded RGBA rows. Scratch is exclusively leased until submission;
pool budgets, live results and runtime identities retain existing ownership.

Runtime mismatch, unsupported kernel/sigma, empty or unsupported dimensions,
signed reflection arithmetic, u32 pixel/component indices, uniform/storage
buffer sizes, texture/binding/workgroup limits and optional profiler capacity
reject before output/scratch allocation or cache initialization. Large color
images can exceed the runtime's storage binding capacity; the filter returns
an error. There is no CPU fallback or automatic placement selection.

`gaussian_blur_gpu_profiled` reserves two optional timestamp stages before
recording. Timestamp collection remains an explicit native readback with a
separate receipt; failed recordings cannot be collected. Default runtimes do
not enable timestamps. Pipelines are cached lazily per runtime in a separate
family; existing box/gray/Sobel/contrast/Vision chain defaults are unchanged.

The CPU Gaussian algorithm and its dispatch remain unchanged. A private packed
resize helper's cfg now follows its actual resize/test consumers, so enabling
only CPU filter/analysis references does not generate a dead-code lint.
GPU development dependencies enable existing `imgproc-filter` reference tests;
no production dependency, core model or unsafe boundary is added.

## Validation

Six native Gaussian tests passed. The primary exact comparison covers 3,240
conditions and 3,176,550 logical bytes: six image geometries (1x1, 1x7, 7x1,
2x3, 17x11 and 65x33), all 1/2/3/4 channel counts, nine width/height kernel
combinations, three asymmetric sigma pairs and all five borders. The reference
is the existing CPU specialized Gaussian implementation, not another device
shader. All resulting logical bytes agree exactly, without a tolerance.

Additional comparisons cover minimum positive subnormal sigma, 1e-300,
1e-100, 0.05, 0.5, 1e100 and f64::MAX, mixed and constant-255 inputs, constant
border mass and complete physical RGBA output with zero unused alpha. A
1025x1025 image crosses the 4096-workgroup grid-stride boundary and matches
CPU for both mixed and maximum-mass Q15 data; its high-limb sums remain exact.

Host limit tests cover exact scratch binding capacities, 128-byte uniforms,
14 reduced device limits, invalid kernels/sigmas/channels/dimensions and large
index/reflection ranges. Invalid runtime/options/profiler calls leave scratch
and texture pool stats unchanged and do not initialize Gaussian pipelines.
Timestamp assertions run when supported; ordinary execution remains tested
on runtimes without timestamps.

Strided RGB ROI tests preserve metadata and expected CPU bytes. Four callers
submit filters against one source while keeping every result live; results
survive source recycling and moving the runtime. Repeated dirty-scratch reuse
does not increase allocations after warmup. A gray/Gaussian/Sobel/AI chain
retains one original upload and no intermediate image readback. A facade smoke
test also verifies constant RGB Q15 output and the explicit resident chain.

The complete native GPU image library regression passed: 79 tests, with 13
intentional ignored measurement fixtures. GPU Clippy with all features/targets,
the facade Gaussian smoke, feature-free GPU builds, the GPU-image WASM build
and formatting checks passed. The final host guard additionally verifies
uniform buffer capacity and all three workgroup dimensions.

## Completed-call performance

Two sequential isolated native release surveys completed successfully on Vulkan
with a discrete device. The adapter model is not recorded. Twelve cases cover
256x192, 1920x1080 and 3840x2160; 1/3/4 channels; 3/5/7 square kernels; mixed
patterns and a constant-255 RGBA case. All measured options use sigmas 1.2
and Reflect101. The existing CPU specialized implementation keeps its normal
SIMD and automatically bounded parallel dispatch; no worker count is forced.
Pipeline compilation, runtime construction and input generation are excluded.

Each case has three completed warmups per stage and 20 paired rounds, rotating
four stages each round:

- CPU owned API: specialized Gaussian, including output/workspace allocation,
  coefficient preparation, the existing parallel/SIMD compute and returned image.
- CPU into: the same specialized algorithm with caller-owned packed output and
  a reused GaussianBlurU8Workspace, including any per-call worker allocations.
- GPU full: explicit image upload/packing, coefficient/control preparation,
  two GPU passes, completed explicit readback/unpacking, receipt copy and
  input/output recycling. It returns the same complete logical host image.
- GPU resident: an already uploaded source, host coefficients/controls, resource
  acquisition, submission and wait_idle completion. It excludes the original
  upload, output readback and image recycling. The output is downloaded and
  verified afterward, before the next timed call.

Every warm/timed result exactly matches the CPU bytes and metadata. The full
receipt asserts physical RGBA upload, scratch/output writes and padded readback.
Resident calls validate the original upload with zero intermediate D2H. No
local builds, tests, other benchmarks or statistical analysis overlap either
survey. Warm pipelines are used; CPU/GPU resource costs inside their respective
API boundaries remain timed.

Times below are medians in milliseconds. Ratios use the geometric mean of
20 paired baseline/GPU time ratios. CPU API/full GPU compares two completed
host-image APIs. CPU into/resident GPU compares reusable output/workspace with
an uploaded device source, and therefore describes compute placement after
inputs are prepared. Values above one favor the GPU route. Parenthesized 95%
intervals use 10,000 paired log-ratio bootstrap draws (sorted indices 249/9749),
seeded by 20261013 + 2*record_index + comparison_index. These intervals describe
within-survey variation; the two synthetic surveys do not establish universal
speedups. Absolute times and scheduling vary between rounds and surveys.

| Survey | Image/channels/kernel | Pattern | CPU API ms | CPU into ms | GPU full ms | GPU resident ms | CPU API/GPU full (95%) | CPU into/GPU resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 256x192/C1/K3 | mixed | 0.054 | 0.047 | 1.070 | 0.200 | 0.05x (0.05-0.05) | 0.24x (0.23-0.25) |
| 1 | 256x192/C3/K7 | mixed | 1.701 | 1.681 | 1.276 | 0.377 | 1.33x (1.29-1.36) | 4.48x (4.34-4.64) |
| 1 | 1920x1080/C1/K3 | mixed | 2.653 | 0.747 | 8.724 | 0.791 | 0.30x (0.29-0.30) | 0.89x (0.84-0.95) |
| 1 | 1920x1080/C3/K3 | mixed | 7.117 | 2.086 | 9.647 | 1.380 | 0.76x (0.73-0.78) | 1.53x (1.46-1.60) |
| 1 | 1920x1080/C1/K5 | mixed | 2.895 | 0.893 | 9.045 | 0.966 | 0.32x (0.31-0.32) | 0.91x (0.86-0.96) |
| 1 | 1920x1080/C3/K5 | mixed | 7.582 | 2.337 | 10.144 | 1.707 | 0.75x (0.74-0.77) | 1.35x (1.32-1.39) |
| 1 | 1920x1080/C1/K7 | mixed | 7.721 | 6.453 | 11.743 | 2.054 | 0.72x (0.64-0.80) | 3.21x (2.94-3.49) |
| 1 | 1920x1080/C3/K7 | mixed | 25.401 | 13.164 | 26.351 | 8.189 | 1.01x (0.88-1.18) | 1.93x (1.66-2.28) |
| 1 | 1920x1080/C4/K7 | mixed | 33.025 | 22.729 | 38.226 | 15.139 | 0.94x (0.85-1.04) | 1.58x (1.47-1.71) |
| 1 | 3840x2160/C1/K5 | mixed | 10.125 | 2.944 | 50.163 | 8.117 | 0.21x (0.19-0.23) | 0.35x (0.30-0.38) |
| 1 | 3840x2160/C3/K7 | mixed | 81.695 | 57.534 | 76.851 | 28.149 | 1.13x (1.02-1.25) | 2.03x (1.85-2.22) |
| 1 | 3840x2160/C4/K7 | hot | 121.313 | 96.935 | 108.217 | 43.857 | 1.16x (1.07-1.28) | 2.50x (2.17-2.88) |
| 2 | 256x192/C1/K3 | mixed | 0.069 | 0.058 | 1.502 | 0.327 | 0.05x (0.05-0.05) | 0.20x (0.18-0.23) |
| 2 | 256x192/C3/K7 | mixed | 1.927 | 1.886 | 1.422 | 0.364 | 1.34x (1.30-1.37) | 5.38x (4.91-5.92) |
| 2 | 1920x1080/C1/K3 | mixed | 2.823 | 0.751 | 8.928 | 0.714 | 0.31x (0.30-0.32) | 1.04x (0.98-1.10) |
| 2 | 1920x1080/C3/K3 | mixed | 7.789 | 2.332 | 9.667 | 1.208 | 0.83x (0.81-0.85) | 1.89x (1.83-1.95) |
| 2 | 1920x1080/C1/K5 | mixed | 2.910 | 0.923 | 8.984 | 0.827 | 0.33x (0.32-0.34) | 1.08x (1.02-1.15) |
| 2 | 1920x1080/C3/K5 | mixed | 8.607 | 2.671 | 10.080 | 1.555 | 0.85x (0.83-0.88) | 1.71x (1.67-1.77) |
| 2 | 1920x1080/C1/K7 | mixed | 10.486 | 8.678 | 12.338 | 2.008 | 0.85x (0.82-0.88) | 4.23x (4.05-4.42) |
| 2 | 1920x1080/C3/K7 | mixed | 24.503 | 18.859 | 21.220 | 7.866 | 1.15x (1.08-1.24) | 2.54x (2.37-2.77) |
| 2 | 1920x1080/C4/K7 | mixed | 34.477 | 29.621 | 29.092 | 14.828 | 1.11x (1.02-1.20) | 1.94x (1.81-2.07) |
| 2 | 3840x2160/C1/K5 | mixed | 10.437 | 2.971 | 47.090 | 8.156 | 0.22x (0.20-0.23) | 0.38x (0.33-0.42) |
| 2 | 3840x2160/C3/K7 | mixed | 81.188 | 59.971 | 77.784 | 25.300 | 1.12x (0.98-1.28) | 2.26x (1.79-2.77) |
| 2 | 3840x2160/C4/K7 | hot | 120.096 | 97.200 | 112.879 | 48.919 | 1.08x (1.05-1.13) | 2.02x (1.91-2.14) |

For the 7x7 1920x1080 grayscale case, resident GPU completion is 3.21-4.23x
faster than the reused CPU into route. For 7x7 mixed RGB at 3840x2160 the
resident gain is 2.03-2.26x; completed host-image API ratios are only
1.12-1.13x, and the second survey's interval includes one. The hot RGBA 4K
case gives 2.02-2.50x resident gains and 1.08-1.16x completed API gains.
The small 256x192 RGB 7x7 case has 4.48-5.38x resident gains and 1.33-1.34x
completed API gains.

The GPU is slower for several lightweight grayscale cases. At 256x192/3x3,
CPU owned API is about 20x faster than a GPU call with upload/readback.
At 4K grayscale/5x5, full GPU time is 4.5-4.7x CPU API time; even resident
GPU completion is 2.6-2.9x reused CPU into time. RGB 1080p 3x3/5x5 gains
1.35-1.89x in the resident comparison, while full GPU remains slower than
CPU. The 1080p RGB/RGBA 7x7 completed-call gains vary between surveys, with
some intervals including one. All results are retained; no automatic GPU
selection or blanket completed-call speed claim is introduced.

The resident filter is useful when image data already stays on device and its
weighted compute is substantial. Callers with host images or light kernels
can keep the existing CPU API. At 4K RGBA, horizontal scratch is 132,710,400
bytes, requiring a storage binding at least that large. Current capacity and
bounded pool retention remain part of the measured API costs.

Raw records and derived artifacts remain private and outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-gaussian-blur-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-gaussian-blur-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-gaussian-blur-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-gaussian-blur.py`

Reproduce from `C:/Users/rsasa/Workspace/SpatialRust`: compile with `--no-run`
first, then execute this exact measurement alone twice, sequentially, with no
overlapping local tests/builds/benchmarks:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib image::kernels::gaussian::tests::benchmark::completed_gaussian_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Source references:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_params.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_horizontal.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_vertical.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/filter.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_gaussian_smoke.rs`

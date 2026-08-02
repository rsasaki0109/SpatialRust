# Specialized Gaussian acceleration

Date: 2026-07-16 (Asia/Tokyo)

## Scope

This slice adds an interleaved `u8` Gaussian surface with 3×3, 5×5, and 7×7
contracts. The accelerated 3×3/5×5 path uses symmetric normalized Q8 kernels,
a `u16` horizontal intermediate, branch-free unrolled interiors, runtime SIMD
dispatch, and bounded Rayon row stages. `GaussianBlurU8Workspace` owns scratch
and cached kernels explicitly; `gaussian_blur_u8_into` writes caller-owned
packed output. The 7×7 contract retains the high-precision generic path until
its wider accumulator specialization is complete.

Python `gaussian_blur_image(..., out=)` now avoids copying contiguous input and
validates output shape, contiguity, and aliasing.

## Native before/after

Criterion on the same Windows 11, Intel 6-core/12-thread host, RGB `u8`, 5×5,
sigma 1.2, Reflect101:

| Profile | Previous generic allocate | Specialized allocate | Improvement | Specialized workspace reuse |
| --- | ---: | ---: | ---: | ---: |
| 640×480 | 20.235 ms | 1.630 ms | 12.4× | 1.330 ms |
| 1080p | 120.460 ms | 5.812 ms | 20.7× | 2.488 ms |
| 4K | 666.160 ms | 24.946 ms | 26.7× | 10.115 ms |

The reuse comparison versus the old allocating path is 48.4× at 1080p and
65.9× at 4K. This closes Epic 116D's 10× Gaussian improvement gate.

## OpenCV boundary

Focused Python timings used CPython 3.12.10, OpenCV 4.13.0, 12 threads,
OpenCL disabled, seeded interleaved samples, and minimum 20 ms batches:

| Profile | Mode | OpenCV | SpatialRust | Outcome |
| --- | --- | ---: | ---: | --- |
| 1080p | allocate | 1.995 ms | 6.188 ms | OpenCV 3.10× |
| 4K | allocate | 7.179 ms | 21.031 ms | OpenCV 2.93× |
| 8K | allocate | 24.361 ms | 87.220 ms | OpenCV 3.58× |
| 1080p | caller output | 1.415 ms | 5.469 ms | OpenCV 3.86× |
| 4K | caller output | 5.191 ms | 20.586 ms | OpenCV 3.97× |
| 8K | caller output | 22.025 ms | 88.229 ms | OpenCV 4.01× |

The canonical 5×5 profiles were bit-exact. Three hundred randomized 3×3,
5×5, and 7×7 RGB cases, including non-contiguous inputs, stayed within a
maximum absolute error of 2/255. The receipt deliberately does not claim a
standalone OpenCV win. It reduces the gap by over an order of magnitude and
identifies high-precision 7×7 and fused Gaussian consumers as the next work.

Reproduce with `bench/opencv_gaussian_comparison/performance.py`; the local
JSON receipt is `target/opencv-gaussian-performance-final.json`.

## Validation

- five border modes and strided Rust views against the generic implementation
- 3×3, 5×5, and 7×7 sizes with metadata preservation
- workspace capacity reuse and invalid output rejection
- Python allocated/output identity, shape, contiguity, and overlap checks
- 300 randomized OpenCV cases and 1080p/4K/8K paired timings
- focused Rust tests, warnings-denied Clippy, and Criterion compilation

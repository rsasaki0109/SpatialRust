# Epic 116 paired Sobel and fused L1 receipt (2026-07-16)

## Outcome

SpatialRust now exposes OpenCV-compatible paired 3×3 Sobel gradients and an
exact fused L1 magnitude operation for grayscale `u8`. The paired primitive is
used by the aperture-3 Canny path. The fused operation avoids materializing X,
Y, absolute-X, and absolute-Y images when only `abs(Gx) + abs(Gy)` is needed.

On the recorded host, allocated fused L1 Python calls are **1.86× faster at
1080p**, **2.19× faster at 4K**, and **2.42× faster at 8K** than the equivalent
OpenCV public pipeline. This is a fusion/allocation result, not a claim that
SpatialRust's standalone paired-gradient primitive beats OpenCV
`spatialGradient`.

## Public API

Rust:

- `spatial_gradient_u8` / `spatial_gradient_u8_into`
- `sobel_l1_magnitude_u8` / `sobel_l1_magnitude_u8_into`

Python:

- `spatial_gradient_image(image, out_dx=None, out_dy=None)`
- `sobel_l1_magnitude_image(image, out=None)`

The paired result is signed `i16`. Fused L1 is non-negative signed `i16` in
[0, 2040]. Replicate and Reflect101 borders are supported in Rust; Python uses
Reflect101. Wrong shapes, non-contiguous outputs, partial paired outputs, and
overlap are rejected.

## OpenCV comparison

Environment:

- Windows 11 `10.0.26300`, Intel Family 6 Model 158, 6 cores / 12 logical CPUs
- CPython 3.12.10, OpenCV 4.13.0, OpenCL disabled, 12 OpenCV threads
- seeded packed random `uint8`, paired/interleaved calls
- 300 additional randomized cases, including non-contiguous Python inputs
- exact `int16` equality required before timing

OpenCV stages are `spatialGradient` → `absdiff(Gx, 0)` → `absdiff(Gy, 0)` →
`add`. SpatialRust uses one fused operation.

| Profile | Mode | OpenCV | SpatialRust | Result |
| --- | --- | ---: | ---: | ---: |
| 1080p | allocate | 5.508 ms | 2.954 ms | **SpatialRust 1.86×** |
| 1080p | reuse | 2.377 ms | 2.369 ms | effectively tied (SpatialRust 1.004×) |
| 4K | allocate | 22.502 ms | 10.262 ms | **SpatialRust 2.19×** |
| 4K | reuse | 10.839 ms | 12.064 ms | OpenCV 1.11× |
| 8K | allocate | 97.258 ms | 40.238 ms | **SpatialRust 2.42×** |
| 8K | reuse | 39.609 ms | 40.313 ms | OpenCV 1.02× |

The authoritative JSON was generated as
`target/opencv-sobel-l1-performance.json` by
`bench/opencv_sobel_l1_comparison/performance.py`.

## Validation

- exact paired gradients against independent Sobel for strided Rust views
- Replicate and Reflect101 border parity, metadata preservation, width-one and
  output-length handling
- fused L1 identity against paired gradients
- Python allocated/caller-owned identity and invalid-output tests
- 300 OpenCV randomized fused-output cases
- `spatialrust-vision` feature tests and Clippy with warnings denied
- release Python extension build and focused binding tests

## Remaining Epic 116 boundary

OpenCV remains faster for standalone paired gradients and the existing generic
Gaussian path. Cached Gaussian kernels, reusable separable intermediates, and
specialized 5×5/7×7 passes remain planned; this receipt does not close those
parts of Epic 116.

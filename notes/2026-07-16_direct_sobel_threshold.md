# Epic 116E direct and fused Sobel receipt (2026-07-16)

## Outcome

The standard grayscale `u8`, first-order 3×3 Sobel path no longer enters the
generic separable engine or allocates a full `f64` intermediate. It uses
bounded parallel stripes, three reusable `i16` horizontal rows per worker, and
direct `f32` output. Additive fused APIs produce saturated absolute `u8`
responses or binary edge masks without public signed/absolute intermediates.

## Public APIs

Rust:

- `sobel_3x3_u8` / `sobel_3x3_u8_into`
- `sobel_abs_3x3_u8` / `sobel_abs_3x3_u8_into`
- `sobel_threshold_3x3_u8` / `sobel_threshold_3x3_u8_into`

Python:

- `sobel_image(..., out=None)` dispatches first-order 3×3 calls to the direct path
- `sobel_abs_image(..., out=None)`
- `sobel_threshold_image(..., threshold, out=None)`

Rust supports Replicate and Reflect101 borders. Python uses Reflect101. Generic
Sobel remains available for other sizes, orders, components, and channels.

## OpenCV comparison

Environment: Windows 11, Intel Family 6 Model 158, 6 cores / 12 logical CPUs,
CPython 3.12.10, OpenCV 4.13.0, OpenCL disabled, 12 OpenCV threads. Inputs are
seeded packed random grayscale `u8`; calls are paired/interleaved and adaptively
batched.

Standalone `CV_32F` Sobel X allocation:

| Profile | OpenCV | SpatialRust | Result |
| --- | ---: | ---: | ---: |
| VGA | 0.372 ms | 0.398 ms | OpenCV 1.07× |
| 1080p | 2.137 ms | 1.134 ms | **SpatialRust 1.88×** |
| 4K | 7.582 ms | 3.737 ms | **SpatialRust 2.03×** |

This reverses historical 20.31× and 23.30× deficits at 1080p and 4K. Packed
NumPy input is borrowed; non-contiguous input is explicitly packed. All
canonical outputs have maximum absolute error zero.

Fused `abs(Sobel X) > 96` binary mask:

| Profile | OpenCV allocate | SpatialRust allocate | SpatialRust caller output | Caller output vs OpenCV allocate |
| --- | ---: | ---: | ---: | ---: |
| VGA | 0.304 ms | 0.080 ms | 0.069 ms | **SpatialRust 4.38×** |
| 1080p | 2.545 ms | 0.522 ms | 0.187 ms | **SpatialRust 13.58×** |
| 4K | 9.453 ms | 1.423 ms | 0.635 ms | **SpatialRust 14.90×** |

OpenCV's allocated pipeline is `Sobel(CV_16S)` → `convertScaleAbs` → binary
`threshold`. Same-mode allocated SpatialRust wins 3.81×, 4.87×, and 6.64×;
same-mode caller-output SpatialRust wins 2.95×, 6.63×, and 8.68×. The larger
cross-mode ratios in the table explain the effect of one fused output versus
three allocated OpenCV stages; they are not substituted for the same-mode
claims.

## Correctness and validation

- direct path equals the generic SpatialRust Sobel for X/Y, scale/delta, both
  supported borders, metadata, and strided Rust inputs
- Python `out=` identity plus shape/contiguity rejection
- 300 randomized OpenCV cases alternating X/Y, including non-contiguous inputs
- exact binary pixels for all timed profiles
- focused Rust tests, feature Clippy with warnings denied, Python crate check
- reproducible JSON:
  `C:\Users\rsasa\Workspace\SpatialRust\target\opencv-sobel-threshold-performance.json`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\opencv-comparison\vision-performance-sobel-direct.json`

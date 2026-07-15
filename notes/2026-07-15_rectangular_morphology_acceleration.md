# Rectangular morphology acceleration receipt — 2026-07-15

## Outcome

SpatialRust now dispatches packed grayscale `u8` rectangular morphology to a
safe separable sliding min/max engine. The result is bit-exact with OpenCV and
reduces the recorded 4K 5×5 opening from 4599.72 ms to 105.05 ms, a **43.8×
improvement** over the previous SpatialRust Python path.

OpenCV remains decisively faster for small rectangles. For a named large-scale
background-estimation/document workload, SpatialRust crosses over: 511×511
opening is **2.46× faster at 1080p** and **2.06× faster at 4K** on this host.

## Algorithm

Rectangles are separable, so a two-dimensional minimum or maximum is computed
as horizontal and vertical one-dimensional windows. Each padded line is split
into kernel-sized blocks with forward prefix and backward suffix extrema; a
window result combines one suffix and one prefix value. Work therefore does
not multiply by the rectangular kernel area. Large images run bounded parallel
row/column stages with explicit transpose buffers. No `unsafe` code or hidden
device transfer is used.

The existing generic implementation remains the exact fallback for other
component types, channels, strides, and Cross/Ellipse/Diamond/custom masks.

References:

- OpenCV morphology API and iteration semantics: <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html>
- OpenCV row/column morphology implementation: <https://codebrowser.dev/opencv/opencv/modules/imgproc/src/morph.simd.hpp.html>
- Kimmel and Gil, *Efficient Implementation of Min/Max Filters*: <https://www.cs.technion.ac.il/wp-content/ron-kimmel/papers/EfficientMorphology_KimmelGil_PAMI_2002.pdf>
- Lemire, *Streaming Maximum-Minimum Filter Using No More than Three Comparisons per Element*: <https://arxiv.org/abs/cs/0610046>

## Reproduction environment

- Windows 11 `10.0.26300`, AMD64
- Intel Family 6 Model 158, 6 cores / 12 logical CPUs
- CPython 3.12.10
- OpenCV 4.13.0, 12 reported threads, OpenCL disabled
- SpatialRust 1.0.0 release wheel
- seeded packed random grayscale `uint8`; `BORDER_REPLICATE`
- six warmups; paired/interleaved order; calls batched to at least 20 ms
- 30 VGA, 20 1080p, and 12 4K samples per kernel

Run:

```powershell
python bench/opencv_morphology_comparison/performance.py `
  --output target/opencv-morphology-performance.json
```

## Python API medians

| Profile | Kernel | OpenCV | SpatialRust | Result |
| --- | ---: | ---: | ---: | ---: |
| VGA | 5×5 | 0.135 ms | 8.858 ms | OpenCV 65.6× |
| VGA | 511×511 | 8.100 ms | 19.433 ms | OpenCV 2.40× |
| 1080p | 5×5 | 1.570 ms | 20.598 ms | OpenCV 13.1× |
| 1080p | 511×511 | 55.543 ms | 22.559 ms | **SpatialRust 2.46×** |
| 4K | 5×5 | 5.741 ms | 105.051 ms | OpenCV 18.3× |
| 4K | 511×511 | 207.788 ms | 100.889 ms | **SpatialRust 2.06×** |

The timing scope includes the Python call and allocated output. OpenCV's
kernel array is prepared outside its call; SpatialRust's public binding still
constructs and validates its rectangular element inside the timed call.

## Correctness gates

- exact output for every timed profile and kernel;
- 980 seeded randomized comparisons covering all seven operations, dimensions
  from 1 pixel upward, odd/even rectangles, zero/two iterations, and replicated
  borders;
- Rust parity against the generic implementation for explicit asymmetric
  anchors, all five border modes, oversized kernels, and multiple iterations;
- contiguous and non-contiguous NumPy input parity.

## Scope boundary

This is not a general claim that SpatialRust morphology is faster than OpenCV.
The measured win is limited to large rectangular windows on sufficiently large
images. OpenCV's tuned SIMD small-window engine remains substantially faster;
Cross/Ellipse/Diamond/custom masks use SpatialRust's generic correctness path.

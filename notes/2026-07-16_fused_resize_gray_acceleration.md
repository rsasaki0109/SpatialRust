# Fused bilinear resize-to-gray acceleration

This Epic 115C slice adds safe allocating and caller-owned APIs that combine a
reusable Q11 `u8` bilinear resize plan with Q14 BT.601 RGB-to-gray conversion.
The implementation writes gray output directly and never materializes the
resized RGB image. Packed and strided inputs/outputs retain explicit ownership
and metadata behavior.

## Correctness

- Fused output is bit-exact with `BilinearResizeU8Plan::resize` followed by
  `rgb_to_gray` for arbitrary dimensions and canonical half reductions.
- Three hundred seeded randomized dimensions include non-contiguous input.
- OpenCV 4.13 disagreement is at most 1/255; canonical exact fractions are
  99.8655%–99.8700%.

## Native reuse medians

| Input → output | Unfused | Fused | Improvement |
| --- | ---: | ---: | ---: |
| 1920×1080 → 960×540 | 0.760 ms | 0.695 ms | 1.09× |
| 3840×2160 → 1920×1080 | 2.775 ms | 2.410 ms | 1.15× |
| 7680×4320 → 3840×2160 | 11.087 ms | 9.453 ms | 1.17× |

## OpenCV 4.13 Python medians

OpenCL was disabled and both runtimes used their default 12-thread policies.
Allocated OpenCV timings include `resize` plus `cvtColor`; reuse supplies both
the intermediate RGB image and final gray output.

| Input → output | OpenCV allocate | SpatialRust allocate | Result | OpenCV reuse | SpatialRust reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920×1080 → 960×540 | 0.755 ms | 0.677 ms | **SpatialRust 1.12×** | 0.347 ms | 0.658 ms |
| 3840×2160 → 1920×1080 | 2.665 ms | 2.687 ms | OpenCV 1.01× | 1.517 ms | 2.392 ms |
| 7680×4320 → 3840×2160 | 8.863 ms | 10.062 ms | OpenCV 1.14× | 6.525 ms | 9.474 ms |

The claim is intentionally limited to the allocated 1080p→540p camera-pyramid
pipeline. The focused harness at
`C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_fused_resize_gray_comparison`
emits the complete environment, dispersion, and raw samples.

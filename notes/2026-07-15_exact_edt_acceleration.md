# Exact EDT acceleration receipt — 2026-07-15

## Outcome

The exact unit-spacing Euclidean distance transform now uses a binary-row
nearest-background pass, compact `u16` horizontal distances, cache-tiled
transposes, a parallel Felzenszwalb–Huttenlocher column envelope, and final
`f32` square roots. Anisotropic spacing keeps the general exact `f64` path.

`DistanceTransformWorkspace`, `distance_transform_edt_into`, and
`distance_transform_edt_u8_into` make scratch/output reuse explicit. The Python
binding exposes the same policy through `DistanceTransformWorkspace` and
`out=`; common packed `0/255` masks no longer require a normalization copy.

## Correctness

- canonical VGA, 1080p, and 4K masks: exact fraction `1.0`, max error `0.0`
  against OpenCV `DIST_L2/DIST_MASK_PRECISE`;
- irregular unit mask: exact brute-force equality;
- anisotropic property coverage retains the general-spacing implementation;
- non-empty all-foreground masks remain an explicit error.

## Performance

Host: Windows 11, 6-core/12-thread Intel CPU, OpenCV 4.10, OpenCL disabled,
CPython 3.12. Timings are machine-specific medians.

| Measurement | Before | After |
| --- | ---: | ---: |
| Native Criterion 4K allocate | 451.63 ms | about 75 ms |
| Native Criterion 4K reusable output/workspace | unavailable | about 43 ms |
| Python/OpenCV 4K allocate ratio | OpenCV 12.35× | OpenCV 1.63× |
| Python/OpenCV 4K reuse ratio | unavailable | OpenCV 1.07× |

The native reusable kernel crosses the earlier OpenCV allocate baseline, while
the fully interleaved Python reuse comparison remains a narrow OpenCV win. No
blanket faster-than-OpenCV claim is made. Re-run
`bench/opencv_vision_comparison/performance.py` before quoting host-specific
numbers.

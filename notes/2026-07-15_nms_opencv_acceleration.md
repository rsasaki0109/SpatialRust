# NMS OpenCV acceleration receipt — 2026-07-15

## Outcome

SpatialRust greedy NMS now precomputes each valid box area once and reuses it
for all overlap comparisons. The same cached geometry is used by class-aware
batched NMS and Soft-NMS. The Python binding borrows contiguous float32 scores
directly and only packs non-contiguous inputs.

This preserves the public safe API, score-descending deterministic ordering,
half-open `xyxy` box semantics, generic non-contiguous NumPy fallback, and the
existing score/IoU validation contract.

## Correctness contract

The comparison uses OpenCV [`dnn.NMSBoxes`](https://docs.opencv.org/master/df/d57/namespacecv_1_1dnn.html)
as the oracle. Both implementations receive the same deterministic float32
coordinates and scores (represented as corresponding `xyxy` and `xywh` boxes), score
threshold `0.25`, and IoU threshold `0.5`.

All three profiles returned exactly the same ordered kept indices:

| Profile | Candidates | Kept | Exact indices |
| --- | ---: | ---: | ---: |
| Small post-process | 100 | 71 | yes |
| Medium post-process | 1,000 | 654 | yes |
| YOLO-style output | 8,400 | 3,675 | yes |

Rust coverage also checks cached-area IoU bit-for-bit against the public IoU
implementation for overlapping, disjoint, identical, and zero-area boxes.

## Performance

Host: Windows 11, Intel 6-core/12-thread CPU, CPython 3.12.10, OpenCV 4.10,
OpenCL disabled. Timings are randomized, interleaved Python API medians after
three warmups and include the returned index array.

| Candidates | Repeats | OpenCV | SpatialRust | SpatialRust speedup |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 50 | 0.2978 ms | 0.0333 ms | **8.95×** |
| 1,000 | 20 | 8.7205 ms | 2.2856 ms | **3.82×** |
| 8,400 | 8 | 407.0856 ms | 126.5618 ms | **3.22×** |

Native Criterion medians for a separately seeded workload were approximately
17.1 microseconds, 1.17 milliseconds, and 50.3 milliseconds respectively.
These results are host- and workload-specific, not universal performance
claims.

Reproduce the machine-readable report with:

```powershell
python bench/opencv_nms_comparison/performance.py `
  --output target/opencv-nms-performance.json
```

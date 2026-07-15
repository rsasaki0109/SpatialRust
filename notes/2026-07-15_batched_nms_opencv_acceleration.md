# Class-aware batched NMS OpenCV acceleration receipt — 2026-07-15

## Outcome

SpatialRust class-aware NMS now stores accepted indices in per-class buckets.
Each candidate only performs IoU comparisons with previously accepted boxes
from its own class, while a separate output vector preserves global descending
score order. The public Rust API remains safe and deterministic.

Python now exposes `batched_nms(boxes, scores, class_ids, ...)`. Contiguous
float32 scores and int64 class IDs are borrowed; non-contiguous arrays use an
explicit packed fallback. The call returns original indices as int64 NumPy.

## Correctness contract

The comparison uses OpenCV
[`dnn.NMSBoxesBatched`](https://docs.opencv.org/master/df/d57/namespacecv_1_1dnn.html)
as the oracle. Both implementations receive the same deterministic float32
boxes and scores, integer class IDs, score threshold `0.25`, and IoU threshold
`0.5`. OpenCV receives `xywh`; SpatialRust receives the corresponding `xyxy`.

| Profile | Candidates | Classes | Kept | Exact ordered indices |
| --- | ---: | ---: | ---: | ---: |
| Multi-class | 1,000 | 20 | 733 | yes |
| YOLO-style | 8,400 | 80 | 6,234 | yes |

Rust coverage also compares the bucketed implementation with the previous
global-scan semantics on 257 deterministic boxes across positive and negative
class IDs. Python coverage checks same-class suppression, different-class
retention, non-contiguous scores/class IDs, and length validation.

## Performance

Host: Windows 11, Intel 6-core/12-thread CPU, CPython 3.12.10, OpenCV 4.10,
OpenCL disabled. Timings are seeded, randomized, interleaved Python API medians
after three warmups and include the returned index array.

| Candidates / classes | Repeats | OpenCV | SpatialRust | SpatialRust speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 / 20 | 30 | 3.5377 ms | 0.1341 ms | **26.38×** |
| 8,400 / 80 | 10 | 211.7618 ms | 2.1776 ms | **97.25×** |

The machine-readable report retains raw samples and dispersion; the 8,400
profile had visible host-load variation, but every paired sample set retained
a wide SpatialRust lead. These are scoped workload results, not universal
claims about all inputs or OpenCV builds.

Native Criterion medians for a separately seeded 80-class workload were about
98.3 microseconds for 1,000 candidates and 2.42 milliseconds for 8,400.

Reproduce the report with:

```powershell
python bench/opencv_batched_nms_comparison/performance.py `
  --output target/opencv-batched-nms-performance.json
```

# Soft-NMS OpenCV acceleration receipt — 2026-07-15

## Outcome

SpatialRust linear and Gaussian Soft-NMS now maintain only active candidates.
One traversal decays scores, compacts survivors, and selects the next maximum;
the chosen item is removed in constant time. Cached box areas are reused,
disjoint boxes return zero IoU before division, and no-op score updates are
skipped. The Python binding borrows contiguous float32 scores and only packs
non-contiguous views.

The public safe Rust API and Python signature remain compatible. Hard,
linear, and Gaussian methods preserve deterministic descending-score order,
including original-index tie breaking.

Soft-NMS is useful for crowded detection scenes because it reduces scores as
a function of overlap rather than deleting every overlapping detection. The
original paper reports improvements without retraining and the same quadratic
complexity class as greedy NMS.

## Correctness contract

The oracle is OpenCV
[`dnn.softNMSBoxes`](https://docs.opencv.org/master/df/d57/namespacecv_1_1dnn.html).
Both implementations receive the same deterministic integer-coordinate boxes,
float32 scores, score threshold `0.25`, IoU threshold `0.5`, and Gaussian sigma
`0.5`. OpenCV receives `xywh`; SpatialRust receives the corresponding `xyxy`.

Across 100, 1,000, and 8,400 candidates for both linear and Gaussian decay:

- kept indices and ordering matched exactly;
- maximum updated-score error was `1.7881393e-7`;
- all results passed the declared `4e-7` absolute score tolerance.

An additional 48 randomized linear/Gaussian cases retained exact indices and
the same observed maximum score error. Rust tests compare the optimized active
set against the previous full-sort semantics for Hard, Linear, and Gaussian,
including tied scores. Python tests cover contiguous and non-contiguous score
arrays.

## Performance

Host: Windows 11, Intel 6-core/12-thread CPU, CPython 3.12.10, OpenCV 4.10,
OpenCL disabled. Timings are seeded, randomized, interleaved Python API medians
after three warmups, batch short calls to at least 5 ms, and include both
returned score and index collections.

| Candidates | Method | Repeats | OpenCV | SpatialRust | SpatialRust speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Linear | 50 | 0.0922 ms | 0.0146 ms | **6.33×** |
| 100 | Gaussian | 50 | 0.1084 ms | 0.0146 ms | **7.40×** |
| 1,000 | Linear | 20 | 5.6359 ms | 1.6494 ms | **3.42×** |
| 1,000 | Gaussian | 20 | 6.0473 ms | 1.2928 ms | **4.68×** |
| 8,400 | Linear | 8 | 310.7088 ms | 76.6595 ms | **4.05×** |
| 8,400 | Gaussian | 8 | 213.6960 ms | 39.8155 ms | **5.37×** |

Native Criterion compares the one-pass active set with a retained full-sort
baseline on the same separately seeded linear workload:

| Candidates | One-pass active set | Sorting baseline | Native improvement |
| ---: | ---: | ---: | ---: |
| 100 | 22.1 µs | 27.3 µs | **1.24×** |
| 1,000 | 2.14 ms | 2.18 ms | **1.02×** |
| 8,400 | 88.7 ms | 108.2 ms | **1.22×** |

These are scoped host/workload measurements, not universal claims.

Reproduce the machine-readable report with:

```powershell
python bench/opencv_soft_nms_comparison/performance.py `
  --output target/opencv-soft-nms-performance.json
```

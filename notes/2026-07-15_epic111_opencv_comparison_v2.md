# Epic 111: OpenCV comparison methodology v2

Date: 2026-07-15 (Asia/Tokyo)

## Receipt

- Platform: Windows 11, AMD64, 12 logical CPUs
- Processor: Intel64 Family 6 Model 158 Stepping 10
- Python: CPython 3.12.10
- OpenCV: 4.10.0, 12 threads, OpenCL disabled
- SpatialRust Python package: 1.0.0
- Input: deterministic seed 103; VGA, 1080p, and 4K RGB u8
- Timing: three warmups; 20/8/3 samples by profile; seeded interleaved pairs;
  garbage collection disabled during samples; calls batched to at least 5 ms
  where practical and normalized per call

Generated JSON remains under `target/opencv-comparison/epic111/` and is not
committed as a portable baseline.

## Accuracy result

All three profiles passed. Bilinear resize, Sobel X, morphology open, and Canny
were exact for this deterministic input. RGB-to-gray and Gaussian blur differed
by at most 1 u8 level. Canny precision, recall, F1, and IoU were all 1.0.
The report also retains MAE, RMSE, relative-L2, exact fraction, PSNR, and binary
disagreement rather than reducing accuracy to one maximum-error number.

## Median latency result

| Profile | Workload | OpenCV | SpatialRust | Outcome |
| --- | --- | ---: | ---: | --- |
| VGA | AI preprocess allocate | 4.280 ms | 0.946 ms | SpatialRust 4.53x |
| VGA | AI preprocess reuse vs OpenCV allocate | 4.547 ms | 0.533 ms | SpatialRust 8.54x |
| 1080p | AI preprocess allocate | 32.640 ms | 4.542 ms | SpatialRust 7.19x |
| 1080p | AI preprocess reuse vs OpenCV allocate | 29.537 ms | 2.576 ms | SpatialRust 11.46x |
| 4K | AI preprocess allocate | 146.931 ms | 13.974 ms | SpatialRust 10.52x |
| 4K | AI preprocess reuse vs OpenCV allocate | 146.381 ms | 8.543 ms | SpatialRust 17.13x |

OpenCV was faster for resize, RGB-to-gray, Gaussian blur, Sobel X, morphology
open, and Canny at every measured profile. Those gaps are recorded in the JSON
instead of omitted. In particular, SpatialRust's current scalar morphology and
Gaussian implementations are optimization targets; this receipt makes no
general CPU-kernel superiority claim.

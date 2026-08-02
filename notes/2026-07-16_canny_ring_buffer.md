# Epic 118B/118D: Canny magnitude ring and high-contrast fast path

Date: 2026-07-16 (Asia/Tokyo)

## Outcome

The allocation-light 3x3 Canny implementation now computes comparison
magnitudes through three-row rings instead of retaining one `i32` value per
pixel. Large images partition rows across Rayon workers; each worker owns three
rows, so directional non-maximum suppression retains exact adjacent-row access
without sharing mutable scratch. Gradients, classification state, and the
hysteresis stack remain caller-owned in `CannyWorkspace`.

Classification also records whether any weak edge exists. If none exists, all
retained edges are already strong and the graph traversal is skipped. This is
especially useful for high-contrast document, map, and industrial line imagery.
Dense/noisy images continue through exact hysteresis.

## Correctness and memory

- 300 seeded randomized images are bit-exact with OpenCV 4.13.0.
- Rust L1/L2 fast paths are bit-exact with `canny_with_intermediates`.
- A 1000x1000 parallel-path test covers worker stripe boundaries and verifies
  reusable storage stays below six bytes per pixel on a high-contrast profile.
- The focused 4K document-line run reserves 43,936,192 workspace bytes. The
  legacy full magnitude image alone required 33,177,600 bytes, while 12
  three-row rings require 552,960 bytes; the legacy gradients+magnitude+state
  lower bound was 74,649,600 bytes.

## Native improvement

Criterion, 4K document lines, 3x3 aperture, thresholds 80/160, L2 gradient:

| Path | Median |
| --- | ---: |
| Inspectable intermediates | 96.914 ms |
| Ring workspace reuse | 8.134 ms |
| Improvement | **11.92x** |

## Focused OpenCV timing

Windows 11, Intel64 Family 6 Model 158, 12 logical CPUs, CPython 3.12.10,
OpenCV 4.13.0 with 12 threads and OpenCL disabled. Timings are seeded,
warm-started, batched to at least 20 ms, and randomized/interleaved.

| Profile | Pattern | OpenCV reuse | SpatialRust reuse | Result |
| --- | --- | ---: | ---: | ---: |
| VGA | document lines | 0.491 ms | 0.686 ms | OpenCV 1.40x |
| 1080p | document lines | 2.900 ms | 2.138 ms | **SpatialRust 1.36x** |
| 4K | document lines | 11.480 ms | 8.103 ms | **SpatialRust 1.42x** |
| VGA | sensor noise | 2.509 ms | 8.979 ms | OpenCV 3.58x |
| 1080p | sensor noise | 17.441 ms | 29.381 ms | OpenCV 1.68x |
| 4K | sensor noise | 84.511 ms | 132.968 ms | OpenCV 1.57x |

All timed outputs are bit-exact. The OpenCV-beating claim is intentionally
limited to reusable high-contrast document-line workloads at 1080p and 4K.

## Reproduction

```powershell
.venv\Scripts\python.exe bench\opencv_canny_comparison\performance.py `
  --warmup 10 --output target\opencv-canny-ring-performance-final.json

cargo bench -p spatialrust-vision --bench canny --features imgproc-canny -- `
  'canny_document_lines/.*/4k' --warm-up-time 1 --measurement-time 3
```

Relevant absolute paths on the receipt host:

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\canny.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_canny_comparison\performance.py`
- `C:\Users\rsasa\Workspace\SpatialRust\target\opencv-canny-ring-performance-final.json`

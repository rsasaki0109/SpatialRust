# Epic 118A/118C: allocation-light Canny

Date: 2026-07-16 (Asia/Tokyo)

## Outcome

The ordinary 3x3 Canny path no longer builds four public intermediate images
before discarding them. `canny_with_intermediates` remains available for
inspection. `canny_into` accepts packed or strided caller-owned output and a
`CannyWorkspace` that retains paired `i16` gradients, comparison magnitude,
classification state, and the hysteresis stack. Large magnitude, directional
suppression, and packed-output stages use the existing Rayon CPU policy.

The Python binding exposes the same contract through `out=` and
`CannyWorkspace`. There are no hidden device transfers.

## Correctness

- Rust fast-path output is bit-exact with the inspectable path for L1 and L2.
- Strided output tests verify row padding remains untouched.
- The focused Python harness passed 300 seeded randomized images bit-exact
  against OpenCV 4.13.0.
- VGA, 1080p, and 4K document-line and sensor-noise timing inputs are also
  bit-exact.

## Focused OpenCV timing

Windows 11, Intel64 Family 6 Model 158, 12 logical CPUs, CPython 3.12.10,
OpenCV 4.13.0 with 12 threads and OpenCL disabled. Calls were warmed up, seeded,
batched to at least 20 ms, and measured in randomized interleaved order.

| Profile | Pattern | OpenCV allocate | SpatialRust allocate | OpenCV reuse | SpatialRust reuse | Reuse result |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VGA | document lines | 0.491 ms | 1.153 ms | 0.430 ms | 0.762 ms | OpenCV 1.77x |
| 1080p | document lines | 3.134 ms | 8.417 ms | 2.631 ms | 4.454 ms | OpenCV 1.69x |
| 4K | document lines | 10.605 ms | 30.408 ms | 10.300 ms | 16.994 ms | OpenCV 1.65x |
| VGA | sensor noise | 2.123 ms | 8.163 ms | 2.039 ms | 7.306 ms | OpenCV 3.66x |
| 1080p | sensor noise | 15.068 ms | 32.668 ms | 15.286 ms | 29.361 ms | OpenCV 1.92x |
| 4K | sensor noise | 64.216 ms | 122.607 ms | 63.170 ms | 113.000 ms | OpenCV 1.79x |

OpenCV still wins these standalone workloads, so Epic 118D remains open. The
focused row replaces the old README measurement where `canny()` always created
inspectable intermediates and OpenCV led by 10.66x–12.65x.

## Reproduction

```powershell
.venv\Scripts\python.exe bench\opencv_canny_comparison\performance.py `
  --output target\opencv-canny-performance.json
```

Relevant absolute paths on the receipt host:

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\canny.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_canny_comparison\performance.py`
- `C:\Users\rsasa\Workspace\SpatialRust\target\opencv-canny-performance.json`

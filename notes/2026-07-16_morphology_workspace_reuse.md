# Morphology workspace and OpenCV reuse receipt — 2026-07-16

## Outcome

Epic 117C adds explicit caller-owned output and reusable host scratch to the
bit-exact rectangular morphology engine. On the recorded host, 511×511 opening
with both libraries writing caller-owned arrays is **3.25× faster than OpenCV
at 1080p** and **2.77× faster at 4K**.

The public Rust surface is additive:

- `RectMorphologyWorkspace`
- `erode_rect_u8_into`
- `dilate_rect_u8_into`
- `morphology_rect_u8_into`

Python exposes `MorphologyWorkspace` and optional `out=` / `workspace=`
arguments on `morphology_image`. The binding caches the validated rectangular
element in the Python workspace, borrows packed NumPy input, returns the exact
caller-owned output object, and rejects overlapping input/output storage.

## Reproduction environment

- Windows 11 `10.0.26300`, AMD64
- Intel Family 6 Model 158, 6 cores / 12 logical CPUs
- CPython 3.12.10
- OpenCV 4.13.0, 12 reported threads, OpenCL disabled
- SpatialRust 1.0.0 release wheel
- seeded packed random grayscale `uint8`; `BORDER_REPLICATE`
- six warmups; paired/interleaved order; calls batched to at least 20 ms
- 30 VGA, 20 1080p, and 12 4K samples per kernel and allocation mode

Run:

```powershell
python bench/opencv_morphology_comparison/performance.py `
  --output target/opencv-morphology-workspace-performance.json
```

## Python API medians

| Profile | Kernel | Mode | OpenCV | SpatialRust | Result |
| --- | ---: | --- | ---: | ---: | ---: |
| VGA | 5×5 | allocate | 0.131 ms | 8.010 ms | OpenCV 60.96× |
| VGA | 5×5 | reuse | 0.173 ms | 10.435 ms | OpenCV 60.32× |
| VGA | 511×511 | allocate | 8.582 ms | 18.034 ms | OpenCV 2.10× |
| VGA | 511×511 | reuse | 8.105 ms | 19.939 ms | OpenCV 2.46× |
| 1080p | 5×5 | allocate | 1.497 ms | 19.968 ms | OpenCV 13.34× |
| 1080p | 5×5 | reuse | 0.933 ms | 15.160 ms | OpenCV 16.25× |
| 1080p | 511×511 | allocate | 60.004 ms | 22.959 ms | **SpatialRust 2.61×** |
| 1080p | 511×511 | reuse | 60.287 ms | 18.562 ms | **SpatialRust 3.25×** |
| 4K | 5×5 | allocate | 5.473 ms | 83.590 ms | OpenCV 15.27× |
| 4K | 5×5 | reuse | 4.054 ms | 72.066 ms | OpenCV 17.78× |
| 4K | 511×511 | allocate | 211.419 ms | 88.032 ms | **SpatialRust 2.40×** |
| 4K | 511×511 | reuse | 221.212 ms | 80.001 ms | **SpatialRust 2.77×** |

The comparison includes the Python API call. In reuse mode, OpenCV receives
`dst=` and SpatialRust receives both `out=` and `workspace=`. Workspace image
capacity stabilizes at exactly 2,073,600 pixels for 1080p and 8,294,400 pixels
for 4K, with 12 persistent worker line-buffer sets.

## Correctness and ownership gates

- bit-exact allocated and reused output for every timed profile;
- 980 seeded randomized OpenCV comparisons across all seven operations;
- Rust `*_into` parity for odd/even kernels, asymmetric anchors, oversized
  rectangles, all border modes, multiple iterations, and strided input;
- full-image, worker-count, and line-buffer capacities remain unchanged on a
  second same-size call;
- Python returns `out` by object identity;
- wrong shape, non-contiguous output, sparse-shape workspace use, and aliased
  input/output are rejected.

## Scope boundary

Scratch reuse improves SpatialRust's steady-state large-window path but does
not erase OpenCV's SIMD advantage for small rectangles. The OpenCV-beating
claim remains limited to large rectangular windows on sufficiently large
images. Generic Cross/Ellipse/Diamond/custom masks do not use this workspace.

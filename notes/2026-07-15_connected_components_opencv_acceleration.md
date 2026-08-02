# Connected-components OpenCV acceleration receipt — 2026-07-15

## Outcome

SpatialRust's connected-component labeling now uses row runs and union-find
instead of queue-based per-pixel flood fill. On the recorded structured
segmentation and document masks it is **2.17×–3.61× faster** than OpenCV's
explicit SAUF implementation while returning exactly the same row-major labels,
foreground areas, and bounding boxes.

Connected components are a foundational post-segmentation operation for object
counting, instance extraction, OCR/document regions, defect inspection, and
binary-mask cleanup. The optimized path remains pure safe Rust.

## Algorithm choice

OpenCV exposes SAUF, BBDT, and Spaghetti labeling. Its documentation states that
SAUF is the variant that forces row-major label ordering, so SAUF is the
compatibility oracle for SpatialRust's existing row-major contract. OpenCV's
default 8-connected path uses Spaghetti and does not promise the same ordering.

The SpatialRust implementation scans each row into maximal foreground runs,
unions only overlapping runs in the preceding row, path-compresses equivalence
classes, then emits consecutive labels and statistics in first-run order. Long
structured regions therefore avoid queue traffic and repeated neighbor checks.
This follows the same union-find optimization direction described by Wu, Otoo,
and Suzuki while exploiting run structure rather than inspecting every
foreground pixel's neighborhood.

References:

- OpenCV connected-components API and ordering contract: <https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html>
- OpenCV implementation dispatch: <https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/connectedcomponents.cpp>
- Wu, Otoo, Suzuki, *Optimizing Two-Pass Connected-Component Labeling Algorithms*: <https://sdm.lbl.gov/~kewu/ps/paa-final.pdf>
- Bolelli et al., *Spaghetti Labeling*: <https://federicobolelli.it/pub_files/2019tip.pdf>

## Reproduction environment

- Windows 11 `10.0.26300`, AMD64
- Intel Family 6 Model 158, 6 cores / 12 logical CPUs
- CPython 3.12.10
- OpenCV 4.13.0, 12 reported threads, OpenCL disabled
- SpatialRust 1.0.0 release wheel
- 8-connectivity structured masks; `uint8` values 0/255
- paired/interleaved order, eight warmups, calls batched to at least 20 ms
- 40 VGA, 30 1080p, and 20 4K samples per pattern

Run:

```powershell
python bench/opencv_connected_components_comparison/performance.py `
  --output target/opencv-connected-components-performance.json
```

## Python API medians

| Profile | Pattern | OpenCV SAUF | SpatialRust | Speedup |
| --- | --- | ---: | ---: | ---: |
| VGA | Segmentation blobs | 1.284 ms | 0.413 ms | 3.11× |
| VGA | Document lines | 1.271 ms | 0.352 ms | 3.61× |
| 1080p | Segmentation blobs | 6.763 ms | 2.815 ms | 2.40× |
| 1080p | Document lines | 6.649 ms | 2.407 ms | 2.76× |
| 4K | Segmentation blobs | 21.356 ms | 9.838 ms | 2.17× |
| 4K | Document lines | 21.075 ms | 8.606 ms | 2.45× |

The timing scope includes each Python API call, label image, and foreground
statistics. Packed NumPy input is borrowed; generated label storage is moved
into NumPy rather than copied.

## Correctness gates

- exact label image against OpenCV `CCL_SAUF`
- exact foreground component count, areas, and half-open bounding boxes
- all canonical profile/pattern pairs passed
- 320 seeded randomized rectangle/noise masks passed for both 4- and
  8-connectivity, including arbitrary non-zero foreground byte values
- 192 deterministic Rust cases match an independent pixel flood-fill reference,
  including areas, boxes, and SpatialRust pixel-center centroids

## Native Criterion medians

| Profile | Segmentation blobs | Document lines |
| --- | ---: | ---: |
| VGA | 376 µs | 269 µs |
| 1080p | 2.319 ms | 1.703 ms |
| 4K | 8.682 ms | 7.806 ms |

The corresponding median throughputs range from 816.7 million to 1.143 billion
pixels/s. Reproduce with:

```powershell
cargo bench -p spatialrust-vision --bench dense --features dense -- `
  connected_components_8_structured
```

## Scope boundary

The speed claim is intentionally limited to masks with useful horizontal run
structure, such as region/instance segmentation and document lines. Preliminary
random-noise tests improved substantially over the former flood fill but still
favored OpenCV for dense or highly fragmented masks; no general random-mask win
is claimed.

# Exact Euclidean distance transform

This slice adds a safe exact Euclidean distance transform (EDT) to
`spatialrust-vision`'s `dense` feature. For every nonzero binary-mask pixel, it
returns the physical L2 distance to the nearest zero pixel. Unit spacing and
positive finite anisotropic `(x, y)` spacing are supported.

## Selection and algorithm

EDT was selected after reviewing the existing catalog because it was absent and
is reusable for mask interior thickness, contour distance, watershed seeds,
feather blending, and signed-distance pipelines. The implementation follows
Felzenszwalb and Huttenlocher's separable lower-envelope transform: a 2D EDT is
computed as two 1D squared-distance transforms and one final square root. Its
time complexity is `O(width * height)` and its working memory is linear in the
image plus the longest axis.

Primary references:

- Pedro F. Felzenszwalb and Daniel P. Huttenlocher, "Distance Transforms of
  Sampled Functions," Theory of Computing 8 (2012),
  <https://doi.org/10.4086/toc.2012.v008a019>.
- OpenCV `distanceTransform` reference semantics and `DIST_MASK_PRECISE`,
  <https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html>.

The public contract deliberately rejects a non-empty all-foreground mask,
because no finite nearest-background distance exists. Empty masks produce an
empty image. Python accepts conventional `0/255` masks by treating every
nonzero value as foreground; the Rust entry point uses validated `BinaryMask`.

## Evidence

- known-grid and anisotropic-spacing unit tests;
- generated-mask property comparison against exhaustive nearest-zero search;
- Python dtype/shape/value coverage;
- OpenCV exact-L2 comparison with maximum error gate `1e-5`;
- VGA/1080p/4K native Criterion and paired Python/OpenCV performance cases.

### Local Windows receipt

The release build ran on Windows 11 with CPython 3.12.10, OpenCV 4.10.0,
12 OpenCV threads, and OpenCL disabled. The canonical threshold-derived masks
matched OpenCV exactly at every profile. A separate irregular-mask correctness
case had maximum absolute error `9.536743e-7`.

| Profile | Native Criterion estimate | Python SpatialRust median | OpenCV median | Outcome |
| --- | ---: | ---: | ---: | --- |
| VGA | 12.585 ms / 24.41 MPix/s | 19.795 ms | 1.867 ms | OpenCV 10.60x |
| 1080p | 100.98 ms / 20.53 MPix/s | 157.172 ms | 12.443 ms | OpenCV 12.63x |
| 4K | 451.63 ms / 18.37 MPix/s | 640.121 ms | 51.813 ms | OpenCV 12.35x |

These results establish an honest scalar baseline, not a superiority claim.
OpenCV's tuned parallel implementation is faster on this host. Future work can
add caller-owned scratch and bounded row/column parallelism without changing
the exact public contract.

Run from `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$HOME\.cargo\bin\cargo.exe" test -p spatialrust-vision --features full
& "$HOME\.cargo\bin\cargo.exe" bench -p spatialrust-vision --features dense --bench dense
.venv\Scripts\python.exe bench\opencv_vision_comparison\run.py
.venv\Scripts\python.exe bench\opencv_vision_comparison\performance.py
```

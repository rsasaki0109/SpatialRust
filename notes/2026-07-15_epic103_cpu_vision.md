# Epic 103: reusable CPU vision paths

Epic 103 adds explicit caller-owned output paths for resize, RGB-to-gray,
interleaved normalization, and planar CHW packing. Packed and strided image
views remain safe, metadata is propagated deliberately, and no CPU API performs
an implicit device transfer.

Large CHW buffers dispatch one safe scoped worker per channel; smaller inputs
stay scalar to avoid thread overhead. Python exposes the same ownership choice
through optional NumPy `out=` arrays and borrows contiguous RGB input without a
packing copy.

The canonical OpenCV performance receipt was recorded on Windows 11, CPython
3.12.10, OpenCV 4.10.0, and a 12-logical-CPU Intel host. Correctness stayed
within one `u8` value for resize/gray and `5.97e-8` for normalized CHW. OpenCV
remained substantially faster for bilinear resize and RGB-to-gray. SpatialRust
was 3.97x/5.76x/7.62x faster for allocating RGB-to-CHW and
8.54x/13.07x/16.11x faster when reusing output at VGA/1080p/4K respectively.
Raw samples, medians, p95, min/max, and environment details are emitted by
`bench/opencv_vision_comparison/performance.py` rather than checked into source.

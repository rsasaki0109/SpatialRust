# OpenCV comparison contract (Epic 101)

This directory defines the machine-readable contract used by every SpatialRust
comparison with OpenCV. It separates two claims:

- **correctness** reports compare numerical behavior against a documented
  tolerance;
- **performance** reports contain raw samples, robust dispersion statistics,
  and per-workload throughput. Paired implementations run in seeded,
  interleaved order; calls shorter than 5 ms are automatically batched and
  normalized per call. A publication must retain the environment receipt.

The stable v1 report envelope contains `schema_version`, `suite`, `kind`,
`status`, `generated_at`, `environment`, and `results`. The implementation is
Python-standard-library only so CI can validate the contract without adding
OpenCV to production dependencies.

## Canonical workload matrix

[`manifest.json`](manifest.json) is the authoritative registry. It reserves
VGA, 1080p, and 4K profiles and the initial competitive workload set:

1. bilinear resize
2. RGB to grayscale
3. Gaussian blur
4. Sobel
5. Canny
6. morphology open
7. ORB
8. StereoBM
9. dense depth to XYZ (allocate and reuse)
10. colored RGB-D to point cloud
11. AI preprocessing
12. RGB-D to voxel end-to-end
13. detection NMS and class-aware batched NMS post-processing

Exact matches use a JSON `null` PSNR (mathematically infinite) so reports remain
strict RFC-compatible JSON. Numerical comparisons retain max/mean/RMS and
relative-L2 error, exact-pixel fraction, and PSNR; binary edge comparisons use
precision, recall, F1, IoU, and disagreement fraction.

Not every workload is a speed gate. A speed claim becomes publishable only
after its harness emits this report contract at all applicable profiles and the
reproduction receipt names the machine and library versions.

## Run

Build the Python extension and install `numpy` plus `opencv-contrib-python`,
then run both current suites:

```powershell
python bench\opencv_comparison\run.py
python bench\opencv_nms_comparison\performance.py
python bench\opencv_batched_nms_comparison\performance.py
```

Reports are written under `target/opencv-comparison/`. Run one suite with
`--suite vision` or `--suite rgbd`. Validate the dependency-free contract with:

```powershell
python bench\opencv_comparison\test_report.py
```

Do not treat one machine's result as a universal performance claim. Store dated
receipts under `notes/` only when the hardware, OS, OpenCV version, SpatialRust
commit, thread settings, and OpenCL/GPU settings are documented.

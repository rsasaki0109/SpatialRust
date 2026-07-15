# Canny comparison

This focused harness compares OpenCV Canny with SpatialRust's ordinary allocated
API and its caller-owned output plus reusable `CannyWorkspace` API. Both use a
3x3 aperture, thresholds 80/160, and L2 gradient magnitude. It checks 300 seeded
random images for bit-exact parity before timing document-line and sensor-noise
profiles at VGA, 1080p, and 4K.

```powershell
.venv\Scripts\python.exe bench\opencv_canny_comparison\performance.py `
  --output target\opencv-canny-performance.json
```

Results are workload- and machine-specific. The report records raw interleaved
samples, versions, thread count, OpenCL state, and caller-owned output timings.

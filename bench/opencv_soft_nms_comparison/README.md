# OpenCV Soft-NMS comparison

This harness compares SpatialRust `soft_nms` with OpenCV `dnn.softNMSBoxes`
for linear and Gaussian score decay. Both receive the same deterministic
integer-coordinate boxes, float32 scores, score/IoU thresholds, and sigma.
Kept indices must match exactly and updated scores must remain within `4e-7`
absolute error before timings are published.

```powershell
python bench/opencv_soft_nms_comparison/performance.py `
  --output target/opencv-soft-nms-performance.json
```

The report follows `spatialrust.opencv-comparison.v1` and records raw samples,
dispersion, library versions, thread policy, and the host environment. Results
are machine-specific and must not be generalized beyond the named workload.

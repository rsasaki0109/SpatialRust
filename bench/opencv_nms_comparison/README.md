# OpenCV NMS comparison

This harness compares SpatialRust `nms` with OpenCV `dnn.NMSBoxes` using the
same deterministic float32 boxes, scores, score threshold, and IoU threshold.
It covers small post-processing, 1,000-candidate, and YOLO-style 8,400-candidate
profiles. Returned indices must match exactly before timings are published.

```powershell
python bench/opencv_nms_comparison/performance.py `
  --output target/opencv-nms-performance.json
```

The report follows `spatialrust.opencv-comparison.v1` and records raw samples,
dispersion, library versions, thread policy, and the host environment. Results
are machine-specific and must not be generalized beyond the named workload.

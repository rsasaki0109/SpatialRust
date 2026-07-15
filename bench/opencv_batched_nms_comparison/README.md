# OpenCV class-aware batched NMS comparison

This harness compares SpatialRust `batched_nms` with OpenCV
`dnn.NMSBoxesBatched` using the same deterministic float32 boxes, scores,
integer class IDs, score threshold, and IoU threshold. It covers 1,000
candidates across 20 classes and a YOLO-style 8,400 candidates across 80
classes. Returned indices must match exactly before timings are published.

```powershell
python bench/opencv_batched_nms_comparison/performance.py `
  --output target/opencv-batched-nms-performance.json
```

The report follows `spatialrust.opencv-comparison.v1` and records raw samples,
dispersion, library versions, thread policy, and the host environment. Results
are machine-specific and must not be generalized beyond the named workload.

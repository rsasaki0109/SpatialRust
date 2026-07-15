# OpenCV connected-components comparison

This harness compares SpatialRust connected-component labeling with OpenCV's
explicit SAUF implementation on structured segmentation-blob and document-line
masks. SAUF is selected because OpenCV documents it as the algorithm that
guarantees row-major label ordering, matching SpatialRust's public contract.

```powershell
python bench/opencv_connected_components_comparison/performance.py `
  --output target/opencv-connected-components-performance.json
```

Labels, component areas, and half-open bounding boxes must match exactly before
timings are published. The report follows `spatialrust.opencv-comparison.v1`
and records raw samples, dispersion, library versions, OpenCV thread policy,
and the host environment. Calls are batched to at least 20 ms to stabilize
allocator-heavy short profiles. Results are scoped to the named structured masks.

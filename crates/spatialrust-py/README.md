# spatialrust (Python)

Python bindings for [SpatialRust](https://github.com/rsasaki0109/SpatialRust) —
**PyTorch for Spatial Computing**. Native-Rust point cloud processing (IO, voxel
downsampling, RANSAC plane segmentation, Euclidean clustering) with NumPy interop
and no C++ binding layer.

## Build & install

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release        # builds the Rust extension into the venv
```

## Quickstart

```python
import numpy as np
import spatialrust as sr

# (N, 3) float32 XYZ -> native point cloud
pts = np.random.default_rng(0).uniform(0, 5, (10_000, 3)).astype(np.float32)
cloud = sr.PointCloud.from_xyz(pts)

# Voxel downsample (policy: "auto" | "cpu" | "cpu-single")
small = sr.voxel_downsample(cloud, leaf_size=0.1, policy="auto")

# Full MVP pipeline: downsample -> normals -> RANSAC plane -> clustering
result = sr.run_pipeline(cloud, leaf_size=0.1, cluster_tolerance=0.3)
print(result)                      # PipelineResult(points=..., clusters=..., ...)
print(result.plane_normal)         # (nx, ny, nz) of the dominant plane
labels = result.labels()           # (N,) int32 cluster ids
xyz = result.output.xyz()          # (N, 3) float32

# Read/write LAS/PCD/PLY/COPC by extension
sr.write("labeled.las", result.output)
reloaded = sr.read("labeled.las")
```

## API

| Symbol | Description |
| --- | --- |
| `PointCloud.from_xyz(arr)` | Build a cloud from an `(N, 3)` float32 array |
| `PointCloud.xyz()` | XYZ as an `(N, 3)` float32 array |
| `PointCloud.labels()` | Cluster labels as `(N,)` int32, or `None` |
| `PointCloud.field_names()` / `len(cloud)` | Schema fields / point count |
| `read(path)` / `write(path, cloud)` | IO by file extension |
| `voxel_downsample(cloud, leaf_size, policy="auto")` | Voxel-grid downsample |
| `run_pipeline(cloud, leaf_size=0.05, cluster_tolerance=None, min_cluster_size=None, plane_distance=None, policy="auto")` | Full MVP pipeline |
| `region_growing(cloud, k_neighbors=30, smoothness_deg=3.0, min_region_size=10)` | Estimate normals, then grow smooth regions |

## Example

```bash
python examples/segment_room.py --png room.png
```

Synthesizes a scan-like room, runs the pipeline, and writes a labeled cloud plus
a top-down preview.

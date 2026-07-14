# SpatialRust Master Architecture (v0.1)

Design version: **v0.1 Master Architecture Draft**

North star: **Rust-native spatial computing**

## Core decisions

| Area | Decision |
| --- | --- |
| Core model | `SpatialTensor` + `SpatialIndex` + `SpatialAlgorithm` + `SpatialRuntime` |
| Storage | Hybrid Schema-SoA + chunked AoSoA views |
| GPU | wgpu/WebGPU-first + CUDA specialized backend |
| Robotics | ROS2 first-class, zero-copy oriented |
| AI | DLPack / ONNX / embedding-native point cloud |
| Repository | Mono repo / Cargo workspace |

## Initial workspace (Epic 0)

- `spatialrust` — meta crate
- `spatialrust-core` — schema, metadata, traits
- `spatialrust-math` — Vec/Mat/Pose math
- `spatialrust-io` — readers/writers (feature-gated formats)
- `spatialrust-gpu` — device buffers and GPU runtime
- `spatialrust-image` — typed CPU image buffers and strided zero-copy views
- `spatialrust-camera` — camera models, distortion, and RGB-D/point-cloud bridge
- `spatialrust-vision` — feature-gated CPU preprocessing, warps, detection, masks, and dense maps

## MVP scope

1. Load PCD/PLY/LAS/LAZ
2. Voxel downsample
3. Normal estimation
4. RANSAC plane segmentation
5. Euclidean clustering
6. ICP registration
7. Save output
8. Partial wgpu execution (voxel key assignment via `filter-voxel-gpu`)

Post-MVP additions:

- Unified file IO via `read_point_cloud_file` / `write_point_cloud_file`
- `MvpPipelineConfig::voxel_policy` for GPU voxel downsampling (`pipeline-mvp-gpu`)

## Dependency direction

```
math -> core -> search/geometry/io/gpu -> algorithms -> integration
```

Forbidden: `core -> io`, `core -> gpu impl`, `core -> ros2`, `core -> ai`.

Image and camera dependency direction:

```
math -> image
math -> image -> vision
math + image + core -> camera -> vision::spatial/rgbd/odometry
```

`spatialrust-image` remains independent of `spatialrust-core`. GPU image storage
must use a dedicated backend and explicit upload/readback APIs.
`spatialrust-vision` keeps preprocessing, warp, detection, dense-map, and spatial
bridges in separate additive features. CPU APIs never perform implicit device
copies; future GPU/CUDA implementations belong behind explicit backend features.

## Roadmap epics

| Year | Focus |
| --- | --- |
| 1 | Foundation / MVP |
| 2 | v1.0 stable geometry runtime |
| 3 | Robotics adoption (ROS2/Autoware/Nav2) |
| 4 | AI integration |
| 5 | Spatial computing platform |

See the full master architecture document in project planning materials for trait-level design, ADRs, and Codex execution tasks (Epics 0–13).

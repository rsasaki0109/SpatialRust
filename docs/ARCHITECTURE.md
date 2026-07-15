# SpatialRust Master Architecture (v0.1)

Design version: **v0.1 Master Architecture Draft**

North star: **Rust-native spatial intelligence: capture, understand, reconstruct, and act**

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
- `spatialrust-image-io` — bounded, feature-gated image codecs and source metadata
- `spatialrust-tensor` — runtime-independent tensor metadata, CPU ownership, and DLPack boundary
- `spatialrust-camera` — camera models, distortion, and RGB-D/point-cloud bridge
- `spatialrust-vision` — feature-gated CPU preprocessing, Feature2D, geometry/multiview, warps, detection, masks, and dense maps

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
math -> image -> image-io
math -> image -> vision
math -> image -> tensor -> ai
math + image + core -> camera -> vision::spatial/rgbd/odometry
```

`spatialrust-image` remains independent of `spatialrust-core`. GPU image storage
must use a dedicated backend and explicit upload/readback APIs; Epic 89 provides
`GpuImage` and image compute kernels in `spatialrust-gpu` behind `gpu-image`.
`spatialrust-image-io` depends on storage, never the reverse; standard codecs
are additive, while TIFF and OpenEXR remain independently gated.
`spatialrust-vision` keeps preprocessing, Feature2D, geometry/multiview (H/F/E,
PnP, sparse LK, stereo BM), warp, detection, dense-map, and spatial bridges in
separate additive features. Geometry depends on `spatialrust-camera` only and does
not pull Feature2D or dense-map types. CPU APIs never perform implicit device
copies; future GPU/CUDA implementations belong behind explicit backend features.
Its `imgproc-*` features share one border extrapolation contract; `filter2d`
means correlation, while true convolution is an explicitly named operation.
`spatialrust-tensor` is distinct from the point-cloud chunk iterator named
`spatialrust-core::SpatialTensor`; it owns generic dtype/shape/stride/device
contracts and never performs implicit host/device transfers.
`spatialrust-ai` depends on `spatialrust-tensor`, never the reverse. Its default
build defines only backend/session and explicit-copy contracts, plus a
deterministic `MockInferenceBackend` for demos and tests. ONNX Runtime and
each hardware execution provider are additive features. Runtime-owned CPU
outputs cross back through the runtime-independent `HostTensorStorage` trait,
so their allocator lifetime can be retained without copying or adding an ONNX
dependency to the tensor crate.
`spatialrust-vision` `ai-adapters` bridge CPU images to contiguous host tensors
and decode depth/mask/detection tensors back into dense vision types without
depending on `spatialrust-ai`.

## Roadmap epics

| Year | Focus |
| --- | --- |
| 1 | Foundation / MVP |
| 2 | v1.0 stable geometry runtime |
| 3 | Robotics adoption (ROS2/Autoware/Nav2) |
| 4 | AI integration |
| 5 | Spatial computing platform |

The canonical post-foundation horizon is reserved as Epic 91–100 in
`docs/ROADMAP.md`. It extends the existing tensor, image, geometry, GPU, and AI
contracts into synchronized sensor replay, mapping, semantic spatial data,
embodied-AI evaluation, robotics execution, scene interchange, and explicit
edge/distributed execution. These capabilities remain outside
`spatialrust-core`; the core supplies stable schemas and capability traits while
dedicated crates own Arrow, MCAP, ROS 2, OpenUSD, glTF, and runtime dependencies.

See the full master architecture document in project planning materials for trait-level design, ADRs, and Codex execution tasks (Epics 0–13).

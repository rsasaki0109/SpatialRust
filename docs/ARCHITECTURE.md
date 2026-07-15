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
The v2 representation is a pooled `rgba8uint` texture with a retained logical
channel count. Kernels return device-resident images and append named receipt
stages; only `upload_u8` and `readback_u8` cross the host/device boundary.
Runtime adapter identity and synchronization are explicit (`adapter_info`,
`wait_idle`), and `recycle` returns textures to the runtime pool.
`spatialrust-image-io` depends on storage, never the reverse; standard codecs
are additive, while TIFF and OpenEXR remain independently gated.
Calibration datasets, robust solver controls, and residual receipts live in
`spatialrust-camera`; they depend only on `spatialrust-math` small dense solves.
External nonlinear optimizers may be added only behind dedicated features.
`spatialrust-vision` keeps preprocessing, Feature2D, geometry/multiview (H/F/E,
PnP, sparse LK, stereo BM), warp, detection, dense-map, and spatial bridges in
separate additive features. Geometry depends on `spatialrust-camera` only and does
not pull Feature2D or dense-map types. CPU APIs never perform implicit device
copies; future GPU/CUDA implementations belong behind explicit backend features.
Video algorithms depend on dense/detection contracts, while timestamped pull
sources are isolated behind `video-adapters`; native codec/camera runtimes stay
in future dedicated adapter crates/features.
Visual and RGB-D odometry kernels remain in the additive `odometry` vision
feature. Their conversion into stamped trajectory motion is a one-way optional
bridge in `spatialrust-mapping`; monocular scale and invalid depth remain
explicit at that boundary.
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

The canonical post-foundation horizon is Epics 91–100 in `docs/ROADMAP.md`.
Epic 91 adds `spatialrust-records` and `spatialrust-arrow` while keeping Arrow
out of `spatialrust-core`. Epic 92 adds `spatialrust-sync` for clocked stamps,
frame graphs, and deterministic multimodal episode replay. Epic 93 adds
`spatialrust-mapping` for trajectories, relative motion traits, and pose graphs.
Epics 94–100 extend into scene reconstruction (`spatialrust-scene`), semantic
spatial data (`spatialrust-semantic`), embodied-AI episodes
(`spatialrust-episode`), robotics runtime contracts (`spatialrust-runtime`),
glTF/OpenUSD interchange (`spatialrust-interchange`), explicit distributed
execution (`spatialrust-distribute`), and platform stability
(`spatialrust-platform`). Heavy native bindings stay optional.

See the full master architecture document in project planning materials for trait-level design, ADRs, and Codex execution tasks (Epics 0–13).

# Feature matrix

This is the authoritative map of optional capabilities in SpatialRust. The
workspace keeps heavy dependencies behind crate-local features; applications
should normally depend on `spatialrust` and select only the profiles they need.

## User-facing profiles

| Profile | Enables | Intended use |
| --- | --- | --- |
| `mvp` | PCD, PLY, LAS/LAZ, COPC; voxel, normals, plane, Euclidean clustering, ICP, MVP pipeline | CPU-first end-to-end pipeline |
| `mvp-http` | `mvp` + HTTP byte-range COPC | Remote COPC input |
| `gpu-wgpu` | wgpu runtime and GPU kernels | Explicit GPU operation APIs |
| `pipeline-mvp-gpu` | `mvp` GPU stages plus `gpu-wgpu` | GPU-enabled MVP pipeline |
| `gpu-aoso-staging` | AoSoA core packing plus GPU-resident frame APIs | Chained GPU execution with explicit readback |
| `serde` | serde derives for core/math metadata and schemas | Configuration and metadata serialization |
| `io-manifest` | explicit storage roots plus file manifests | SHA-256/size receipts for local IO |
| `rosbag2-sqlite` | read-only rosbag2 SQLite PointCloud2 CDR source | Bounded XYZ/XYZI records in `spatialrust-ros2`; native ROS 2 executors remain separate |

The Python extension selects its supported meta-crate features in
`crates/spatialrust-py/Cargo.toml`. It is intentionally outside the Rust
workspace because its build requires a Python toolchain.

## Crate capability map

| Crate | CPU/default capability | Optional capability | Heavy dependency boundary |
| --- | --- | --- | --- |
| `spatialrust-core` | schema, metadata, `PointCloud`, tensors, execution contracts | serde, AoSoA packing | none |
| `spatialrust-math` | vector/matrix/pose math | serde | none |
| `spatialrust-io` | no format enabled by default | PCD, PLY, LAS/LAZ, E57, COPC, HTTP COPC, explicit roots/manifests, per-node `CopcNodeReader` hierarchy walk | format and checksum crates are optional |
| `spatialrust-ros2` | ROS 2 type contracts through `spatialrust-runtime` | read-only rosbag2 SQLite PointCloud2 CDR streaming with optional float32 intensity, plus source-bound TFMessage inventory | `rusqlite` is isolated behind `rosbag2-sqlite` |
| `spatialrust-search` | KD-tree | graph, parallel queries | none |
| `spatialrust-filtering` | voxel | GPU voxel, outlier, crop, FPS, MLS | wgpu/search optional |
| `spatialrust-features` | normals | ISS, orientation, boundary, GPU normals | wgpu/search optional |
| `spatialrust-segmentation` | plane and Euclidean clustering | GPU stages, DBSCAN, ground, primitives, region growing | wgpu/search optional |
| `spatialrust-registration` | ICP | point-to-plane, GICP, GPU covariance, NDT, FPFH | wgpu/search optional |
| `spatialrust-gpu` | device markers only | wgpu runtime, AoSoA staging | wgpu/bytemuck/pollster optional |
| `spatialrust-pipeline` | MVP pipeline | GPU MVP stages | algorithm crates only |
| `spatialrust-interchange` | `interchange-gltf`, `interchange-openusd` | `tiles3d`: deterministic OGC 3D Tiles 1.1 `tileset.json` + `pnts` octree export; `tiles3d-copc`: bounded COPC hierarchy → tileset | `tiles3d-copc` pulls `spatialrust-io` + `spatialrust-core` for COPC node reads |
| `spatialrust-py` | Python binding surface | selected meta-crate features, including `export_tiles3d` / `export_copc_tiles3d` | PyO3/NumPy |

## Execution contract

- `ExecutionPolicy::CpuSingle` and `CpuParallel` request CPU execution.
- `ExecutionPolicy::Gpu(DeviceKind)` explicitly names the requested GPU
  backend. A backend that cannot satisfy the request must return an error; it
  must not silently turn an explicit request into CPU work.
- `ExecutionPolicy::Auto` is the only policy that permits algorithm-specific
  heuristics and CPU fallback, including fallback when no compatible wgpu
  adapter is available.
- `ExecutionReceipt` and `ExecutionOutput<T>` preserve the requested policy,
  resolved backend, logical stages, and explicit transfer byte accounting.
- `MvpPipelineResult.receipt` exposes the receipt for every MVP stage and an
  aggregate transfer view; a GPU stage may still report a documented host-side
  refinement stage.
- `GpuSpatialFrame` is the explicit GPU-resident path. Host positions are
  uploaded at the frame boundary, GPU stages remain resident, and readback is
  performed only through an explicit readback method. The frame receipt records
  transfer accounting.
- `SpatialRuntime` contains only backend identity and policy compatibility;
  backend-specific queues, buffers, and transfers stay in the implementing
  crate.
- `spatialrust-core` contains no wgpu, CUDA, ROS2, ONNX, or IO implementation.

Host-staged GPU algorithms are explicit hybrid implementations: voxel, normal,
and plane APIs expose their GPU kernel boundary, while Euclidean clustering
builds, sorts, and compacts the sparse grid on wgpu and performs deterministic
component union on the host. This split is visible in receipts and is not an
implicit CPU fallback.

## Verification commands

```bash
cargo test --workspace
cargo test --workspace --all-features --no-run
cargo test -p spatialrust --features mvp
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

GPU runtime tests require a compatible adapter. On GPU-less CI, compile the
GPU test targets and execute them on a runner with a wgpu adapter.

## Deliberately separate future crates

The following are not core features and must remain separate integration
crates: `spatialrust-ros2`, `spatialrust-ai`/DLPack/ONNX, and the CUDA backend.

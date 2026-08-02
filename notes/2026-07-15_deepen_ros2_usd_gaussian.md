# Deepen ROS 2 CDR, USDA ASCII, and Gaussian CPU renderer

Date: 2026-07-15 (Asia/Tokyo)

## ROS 2 (`runtime-ros2`)

Path: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-runtime\src\ros2.rs`

- Negotiation catalog for `sensor_msgs/msg/PointCloud2`
- CDR little-endian XYZ encode/decode
- `LoopbackRos2Node` publish/take without linking `rclrs`

```text
cargo test -p spatialrust-runtime --features ros2 --lib
```

## OpenUSD (`interchange-openusd`)

Path: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-interchange\src\usd.rs`

- `MemoryUsdStageAdapter::export_usda` / `import_mesh_from_usda`
- Portable `#usda 1.0` Mesh prims (no libusd)

```text
cargo test -p spatialrust-interchange --features openusd --lib
```

## Gaussian (`scene-gaussian`)

Path: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-scene\src\gaussian.rs`

- Validated `GaussianScene` + `render_gaussians_cpu` soft-splat RGBA8 path

```text
cargo test -p spatialrust-scene --features gaussian --lib
```

## Still install-time only

- Native `rclrs` executor linking
- Native libusd / Hydra
- GPU 3DGS rasterizer

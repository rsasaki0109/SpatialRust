# SpatialRust

**PyTorch for Spatial Computing** — a Rust-native framework for point clouds, geometry, GPU compute, robotics, and spatial AI.

## Status

MVP pipeline stages are implemented (PCD IO, voxel, normals, segmentation, ICP). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the master design.

## Workspace crates

| Crate | Role |
| --- | --- |
| `spatialrust` | Meta crate / stable re-exports |
| `spatialrust-core` | Point schema, metadata, execution traits |
| `spatialrust-math` | Vec/Mat/Pose math primitives |
| `spatialrust-io` | Point cloud readers/writers |
| `spatialrust-search` | KD-tree spatial search |
| `spatialrust-filtering` | Voxel downsample and filters |
| `spatialrust-features` | Normal estimation |
| `spatialrust-segmentation` | RANSAC plane and Euclidean clustering |
| `spatialrust-registration` | ICP registration |
| `spatialrust-pipeline` | Composable MVP pipelines |
| `spatialrust-gpu` | Device buffers and GPU runtime |

## Quick start

```bash
cargo test --workspace
cargo test -p spatialrust --features mvp
cargo doc --workspace --open
```

Load or save by file extension:

```rust
use spatialrust::{read_point_cloud_file, write_point_cloud_file};

let cloud = read_point_cloud_file("scan.las")?;
write_point_cloud_file("output.ply", &cloud)?;
```

## MVP target pipeline

```
PCD/PLY/LAS -> voxel downsample -> normals -> plane RANSAC -> clustering -> ICP -> save
```

Optional partial GPU execution is available for voxel key assignment:

```bash
cargo test -p spatialrust-gpu --features gpu-wgpu
cargo test -p spatialrust --features filter-voxel-gpu
cargo test -p spatialrust-io --features io-las
cargo test -p spatialrust-io --features io-laz
```

## License

Licensed under MIT OR Apache-2.0 at your option.

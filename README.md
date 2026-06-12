# SpatialRust

**PyTorch for Spatial Computing** — a Rust-native framework for point clouds, geometry, GPU compute, robotics, and spatial AI.

## Status

MVP pipeline is implemented end-to-end: PCD/PLY/LAS/COPC IO, voxel downsampling (CPU + optional wgpu), normals, RANSAC plane segmentation, Euclidean clustering, and ICP. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the master design.

## Workspace crates

| Crate | Role |
| --- | --- |
| `spatialrust` | Meta crate / stable re-exports |
| `spatialrust-core` | Point schema, metadata, execution traits |
| `spatialrust-math` | Vec/Mat/Pose math primitives |
| `spatialrust-io` | Point cloud readers/writers (PCD, PLY, LAS, COPC) |
| `spatialrust-search` | KD-tree spatial search |
| `spatialrust-filtering` | Voxel downsample and filters |
| `spatialrust-features` | Normal estimation |
| `spatialrust-segmentation` | RANSAC plane and Euclidean clustering |
| `spatialrust-registration` | ICP registration |
| `spatialrust-pipeline` | Composable MVP pipelines |
| `spatialrust-gpu` | wgpu runtime and voxel kernels |

## Quick start

```bash
cargo test --workspace
cargo test -p spatialrust --features mvp
cargo doc --workspace --open
```

### CLI (MVP pipeline)

```bash
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- input.las output.las
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --leaf-size 0.2 --voxel-policy auto scan.copc.laz out.copc.laz
```

### Library

Load or save by file extension:

```rust
use spatialrust::{read_point_cloud_file, write_point_cloud_file};

let cloud = read_point_cloud_file("scan.las")?;
write_point_cloud_file("output.ply", &cloud)?;
```

COPC partial read:

```rust
use spatialrust::{read_copc_file_with_query, CopcBounds, CopcQuery};

let bounds = CopcBounds::from_ranges((0.0, 100.0), (0.0, 100.0), (-1.0, 1.0));
let cloud = read_copc_file_with_query("scan.copc.laz", &CopcQuery::bounds(bounds))?;
```

## MVP target pipeline

```
PCD/PLY/LAS/COPC -> voxel downsample -> normals -> plane RANSAC -> clustering -> ICP -> save
```

GPU voxel downsampling (wgpu) is available behind features. `ExecutionPolicy::Auto` keeps CPU for clouds below ~500k points (centroid mode).

```bash
cargo test -p spatialrust-gpu --features gpu-wgpu
cargo test -p spatialrust --features filter-voxel-gpu
cargo test -p spatialrust --features mvp mvp_copc_pipeline_roundtrip
cargo test -p spatialrust --features mvp mvp_copc_query_pipeline
```

## License

Licensed under MIT OR Apache-2.0 at your option.

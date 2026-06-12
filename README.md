# SpatialRust

<p align="center">
  <img src="docs/assets/readme_mvp_pipeline.gif" alt="SpatialRust MVP pipeline: input, voxel downsample, plane RANSAC, and Euclidean cluster labels from a real pipeline run" width="720">
</p>

<p align="center">
  <strong>PyTorch for Spatial Computing</strong><br>
  Point clouds · wgpu · COPC · RANSAC · ICP — native Rust, no C++ binding layer.
</p>

<p align="center">
  <a href="https://github.com/rsasaki0109/SpatialRust/actions/workflows/ci.yml"><img src="https://github.com/rsasaki0109/SpatialRust/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.75+-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/GPU-wgpu-38bdf8.svg" alt="wgpu">
</p>

The GIF above is **real MVP pipeline output** (not a mockup). Regenerate all README assets:

```bash
cargo run -p spatialrust --features mvp --example readme_mvp_preview
```

## Why SpatialRust?

| | Typical C++ stack (PCL / Open3D bindings) | SpatialRust |
| --- | --- | --- |
| Core language | C++ + FFI glue | **Native Rust** |
| GPU path | varies by wrapper | **wgpu voxel filter** with CPU fallback |
| COPC | bolt-on scripts | **bounds + LOD queries** in library & CLI |
| Pipeline | glue code | **composable MVP crate** |

**One command** from LAS/COPC to labeled clusters:

```bash
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- scan.las labeled.las
```

Partial COPC read + pipeline:

```bash
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --bounds 0,0,-1,100,100,1 --resolution 0.5 scan.copc.laz roi.copc.laz
```

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
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --bounds 0,0,-1,100,100,1 scan.copc.laz roi.copc.laz
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --bounds 0,0,-1,100,100,1 --resolution 0.5 scan.copc.laz roi.copc.laz
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --resolution 0.5 scan.copc.laz coarse.copc.laz
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

## Social preview

Upload `docs/assets/social_preview.svg` (or export to PNG) as the GitHub repository social image under **Settings → General → Social preview**.

## License

Licensed under MIT OR Apache-2.0 at your option.

# SpatialRust

<p align="center">
  <img src="docs/assets/readme_hero.gif" alt="SpatialRust hero: isometric LiDAR room scan sweep, voxel downsample, plane RANSAC, and neon Euclidean cluster labels from a real MVP pipeline run" width="960">
</p>

<p align="center">
  <strong>PyTorch for Spatial Computing</strong><br>
  Point clouds · wgpu · COPC · RANSAC · ICP — native Rust, no C++ binding layer.
</p>

<p align="center">
  <a href="https://github.com/rsasaki0109/SpatialRust/actions/workflows/ci.yml"><img src="https://github.com/rsasaki0109/SpatialRust/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://rsasaki0109.github.io/SpatialRust/spatialrust/"><img src="https://img.shields.io/badge/docs-rustdoc-blue.svg" alt="Docs"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-md-green.svg" alt="Changelog"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.75+-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/GPU-wgpu-38bdf8.svg" alt="wgpu">
</p>

The hero GIF above is **real MVP pipeline output** (not a mockup): an isometric room scan sweeps in, voxel-downsamples, RANSAC peels off the dominant floor plane, and Euclidean clustering lights up each object in its own neon color — every frame rendered straight from a live pipeline run.

<p align="center">
  <img src="docs/assets/readme_mvp_preview.svg" alt="SpatialRust MVP pipeline preview: RANSAC plane inliers, Euclidean cluster labels, and the pipeline stages" width="960">
</p>

| ⚡ GPU-accelerated | 🗂️ COPC-native | 🦀 Pure Rust | 🧩 Composable |
| --- | --- | --- | --- |
| wgpu voxel filter, **~3.9× at 2M points**, automatic CPU fallback | **bounds + LOD** partial reads straight off disk — no full-tile load | no C++ / FFI binding layer to fight | one MVP crate: **IO → filter → segment → register** |

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

Partial COPC read + pipeline — stream only the region of interest straight off disk, no full-tile load:

```bash
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --bounds 0,0,-1,100,100,1 --resolution 0.5 scan.copc.laz roi.copc.laz
```

<p align="center">
  <img src="docs/assets/copc_query.gif" alt="COPC partial read: a bounds box selects a region of interest from the full tile, then the recentered subset is read out to roi.copc.laz" width="720">
</p>

## Performance

The voxel downsampler runs on CPU or GPU (wgpu). `ExecutionPolicy::Auto` keeps small clouds on the CPU — where it's fastest — and switches to the GPU as point counts grow, so you get the best of both without tuning.

<p align="center">
  <img src="docs/assets/benchmark_voxel.svg" alt="Voxel downsample latency: CPU vs GPU across point counts, showing the GPU crossover above ~200k points and ~3.9x speedup at 2M" width="960">
</p>

End-to-end centroid filter latency (leaf=4.0), measured via `cargo bench -p spatialrust-filtering`:

| Points | CPU | GPU | Winner |
| ---: | ---: | ---: | :--- |
| 100k | **~7 ms** | ~17 ms | CPU |
| 200k | **~24 ms** | ~26 ms | ~even |
| 500k | ~94 ms | **~51 ms** | GPU |
| 1M | ~155 ms | **~56 ms** | GPU (~2.8x) |
| 2M | ~389 ms | **~101 ms** | GPU (~3.9x) |

Reproduce: `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample`.

Normal estimation has an optional wgpu path (`GpuNormalEstimator`, `feature-normal-gpu`). In **radius mode** the neighbor search runs entirely on the GPU via a uniform grid (covariance + Jacobi eigensolver included), which is **up to ~50× faster** than the CPU KD-tree estimator:

| Points | CPU (KD-tree) | GPU grid | Speedup |
| ---: | ---: | ---: | :--- |
| 100k | ~220 ms | **~8.6 ms** | ~26× |
| 200k | ~442 ms | **~15 ms** | ~29× |
| 500k | ~1.47 s | **~29 ms** | ~50× |

(A k-nearest mode that keeps neighbor search on the CPU is also available but only ~1.1× — see [notes](notes/2026-06-15_gpu_normals_bench.md).) Reproduce: `cargo bench -p spatialrust-features --features feature-normal-gpu --bench normals`.

### Registration methods

Four registration backends, compared on a synthetic box corner (7500 points, small misalignment):

| Method | Recovery error | Time | Notes |
| --- | ---: | ---: | --- |
| ICP (point-to-point) | 0.0196 m | ~147 ms | slow to converge on planar surfaces |
| **Point-to-plane ICP** | 0.0007 m | **~6.5 ms** | best speed/accuracy balance |
| GICP | **0.0006 m** | ~26 ms | most accurate; per-point covariance (optional GPU covariance ~1.7×, `register-gicp-gpu`) |
| NDT | 0.0008 m | ~8.7 ms | voxel distributions + Levenberg–Marquardt |

See [notes](notes/2026-06-15_registration_bench.md). Reproduce: `cargo bench -p spatialrust-registration --features register-icp,register-icp-point-to-plane,register-gicp,register-ndt --bench registration`.

## Status

MVP pipeline is implemented end-to-end: PCD/PLY/LAS/COPC IO, voxel downsampling (CPU + optional wgpu), normals, RANSAC plane segmentation, Euclidean clustering, region growing, and registration (ICP point-to-point/point-to-plane, GICP, NDT). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the master design.

## Workspace crates

One dataflow, eleven crates — each pipeline stage maps to the crate that implements it, all sitting on a small math/core/search foundation:

<p align="center">
  <img src="docs/assets/architecture.svg" alt="SpatialRust architecture: Load → Voxel → Normals → Plane → Cluster → Register → Save dataflow with implementing crates, wgpu voxel acceleration, and the core/math/search foundation" width="960">
</p>

| Crate | Role |
| --- | --- |
| `spatialrust` | Meta crate / stable re-exports |
| `spatialrust-core` | Point schema, metadata, execution traits |
| `spatialrust-math` | Vec/Mat/Pose math primitives |
| `spatialrust-io` | Point cloud readers/writers (PCD, PLY, LAS, COPC) |
| `spatialrust-search` | KD-tree spatial search |
| `spatialrust-filtering` | Voxel downsample and filters |
| `spatialrust-features` | Normal estimation (CPU + optional wgpu, incl. GPU grid neighbor search) |
| `spatialrust-segmentation` | RANSAC plane, Euclidean clustering, region growing |
| `spatialrust-registration` | Registration: ICP (point-to-point, point-to-plane), GICP, NDT |
| `spatialrust-pipeline` | Composable MVP pipelines |
| `spatialrust-gpu` | wgpu runtime and voxel kernels |

## Python

True to the **"PyTorch for Spatial Computing"** tagline, the whole pipeline is callable from Python with NumPy interop — no C++ binding layer:

```python
import numpy as np
import spatialrust as sr

cloud = sr.PointCloud.from_xyz(points)            # (N, 3) float32 -> native cloud
result = sr.run_pipeline(cloud, leaf_size=0.1, cluster_tolerance=0.3)

print(result.plane_normal)                        # dominant plane normal (nx, ny, nz)
labels = result.labels()                          # (N,) int32 cluster ids
sr.write("labeled.las", result.output)            # LAS/PCD/PLY/COPC by extension
```

<p align="center">
  <img src="docs/assets/python_segmentation.png" alt="Top-down view of five neon clusters segmented from a synthetic room scan via a single Python run_pipeline() call" width="540">
</p>

Registration is callable too — align two scans with ICP / point-to-plane / GICP / NDT:

```python
result = sr.register_gicp(source, target)   # also: register_icp / _point_to_plane / _ndt
T = result.transform()                       # 4x4 matrix mapping source -> target
```

<p align="center">
  <img src="docs/assets/python_registration.png" alt="Before/after of two scans aligned by SpatialRust: a misaligned orange source scan snaps onto the blue target after registration" width="760">
</p>

Build the extension with [maturin](https://www.maturin.rs/) and reproduce the image above:

```bash
pip install maturin numpy matplotlib
cd crates/spatialrust-py && maturin develop --release
python examples/segment_room.py --png ../../docs/assets/python_segmentation.png
```

Prebuilt `abi3` wheels (CPython 3.8+) are produced by CI and published to PyPI on tagged releases (`pip install spatialrust`). See [crates/spatialrust-py/README.md](crates/spatialrust-py/README.md) for the full Python API.

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

<p align="center">
  <img src="docs/assets/readme_mvp_pipeline.gif" alt="Top-down 2D view of the MVP pipeline stages: input scan, voxel grid, plane RANSAC, and neon Euclidean clusters" width="640">
</p>

GPU voxel downsampling (wgpu) is available behind features. `ExecutionPolicy::Auto` keeps CPU for clouds below ~500k points (centroid mode).

```bash
cargo test -p spatialrust-gpu --features gpu-wgpu
cargo test -p spatialrust --features filter-voxel-gpu
cargo test -p spatialrust --features mvp mvp_copc_pipeline_roundtrip
cargo test -p spatialrust --features mvp mvp_copc_query_pipeline
```

## README visuals

Every image in this README — the hero GIF, the COPC query GIF, the architecture diagram, the performance chart, and the social card — is generated from a single live pipeline run. Regenerate them all with:

```bash
cargo run -p spatialrust --features mvp --example readme_mvp_preview
```

Outputs: `readme_hero.gif` (header), `readme_mvp_preview.svg` (pipeline panel), `copc_query.gif` (COPC partial read), `benchmark_voxel.svg` (Performance chart), `architecture.svg` (crates diagram), `readme_mvp_pipeline.gif` (compact 2D view), and `social_preview.svg`.

## Social preview

Upload `docs/assets/social_preview.svg` (or export to PNG) as the GitHub repository social image under **Settings → General → Social preview**.

## License

Licensed under MIT OR Apache-2.0 at your option.

# GPU Euclidean clustering (Epic 65 / 2026-07-03)

## Goal

Accelerate MVP Euclidean clustering on plane outliers with wgpu, using the same
`ExecutionPolicy` surface as voxel, plane, and normal stages.

## Implementation

- WGSL label propagation over a uniform grid (cell size = `cluster_tolerance`)
- CPU counting-sort grid build (reuses `normals_grid` helpers)
- Up to 128 propagation iterations on GPU
- CPU `finalize_euclidean_clusters` for min/max size filtering and label remap

## API

| Item | Location |
| --- | --- |
| `euclidean_cluster_roots_gpu` | `spatialrust-gpu` |
| `GpuEuclideanClusterExtractor` | `spatialrust-segmentation` |
| `extract_with_policy` | `EuclideanClusterExtractor` |
| `MvpPipelineConfig::cluster_policy` | `spatialrust-pipeline` |
| `--cluster-policy` | `spatialrust-mvp` CLI |
| Feature | `segment-euclidean-gpu` / `pipeline-mvp-gpu` |

## Auto threshold

`DEFAULT_GPU_MIN_POINTS_EUCLIDEAN = 2_000` — matches MVP plane outlier scale on
the reference `table_scene_lms400` path.

## Verify

```bash
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu gpu_finds_three_separated_clusters
cargo test -p spatialrust-pipeline --features pipeline-mvp-gpu runs_with_gpu_cluster_policy
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu
```

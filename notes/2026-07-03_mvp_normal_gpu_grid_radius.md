# MVP GPU normal radius from voxel leaf (2026-07-03)

## Goal

When MVP selects the wgpu normal backend under `ExecutionPolicy::Auto` or `Gpu`,
and `NormalEstimationConfig::search_radius` is unset, derive a radius from the
voxel leaf so the **GPU uniform-grid path** runs instead of k-NN (up to ~50× on
dense scans — see `notes/2026-06-15_gpu_normals_bench.md`).

## Changes

- `MvpPipelineConfig::normal_gpu_radius_scale` (default `2.0`)
- In `MvpPipeline::run`, when GPU normal is selected and `search_radius` is
  `None`, set `search_radius = leaf_size * normal_gpu_radius_scale` (min `1e-4`)
- `NormalEstimator::selects_gpu_backend` for policy resolution without duplicating
  threshold logic

## Rationale

| Mode | Neighbor search | Typical speedup |
| --- | --- | --- |
| k-NN (CPU tree + GPU cov) | CPU KD-tree | ~1.1× |
| Radius + GPU grid | wgpu uniform grid | **~50×** |

MVP default voxel leaf `0.05` → radius `0.10` at scale `2.0`, matching grid cell
scale for downsampled scenes.

Auto still keeps CPU for clouds below `DEFAULT_GPU_MIN_POINTS_NORMAL = 10_000`.
Use `--normal-policy gpu` on smaller dense crops, or set `search_radius` explicitly.

## Verify

```bash
cargo test -p spatialrust-pipeline --features pipeline-mvp-gpu
cargo test -p spatialrust-core tensor::
```

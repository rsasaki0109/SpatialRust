# MVP GPU normal integration (2026-07-03)

## Goal

Wire existing `GpuNormalEstimator` (`feature-normal-gpu`) into the MVP pipeline
with the same `ExecutionPolicy` surface as voxel and plane stages.

## Changes

- `NormalEstimator::estimate_with_policy` + `NormalEstimationConfig::gpu_min_points`
- `MvpPipelineConfig::normal_policy` (default `Auto`)
- `pipeline-mvp-gpu` enables `feature-normal-gpu`
- CLI `--normal-policy auto|cpu|gpu`

## Auto threshold

`DEFAULT_GPU_MIN_POINTS_NORMAL = 10_000`.

Rationale: MVP default k-NN config sees modest GPU wins until ~10k+ points
(`notes/2026-06-15_gpu_normals_bench.md`). MVP downsampled scenes (~2k on
`table_scene_lms400`) stay on CPU under `Auto`. Use `--normal-policy gpu` or
set `search_radius` + GPU grid path for larger speedups on dense scans.

## Verify

```bash
cargo test -p spatialrust-pipeline --features pipeline-mvp-gpu runs_with_gpu_normal_policy
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu
```

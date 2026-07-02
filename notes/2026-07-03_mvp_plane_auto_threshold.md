# MVP plane Auto threshold tuning (2026-07-03)

## Goal

Pick `DEFAULT_GPU_MIN_POINTS_PLANE` so `ExecutionPolicy::Auto` selects GPU RANSAC
plane scoring on both full clouds and the MVP downsampled path.

## Bench setup

Public PCL `table_scene_lms400.pcd` (460,400 points), 1000 RANSAC iterations,
release build via `bench/ransac_plane/run.py`.

| Path | Points | CPU (s) | GPU (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Full cloud | 460,400 | ~1.94 | ~0.18 | ~11× |
| MVP preprocess (`--mvp-leaf 0.05`, voxel + normals) | 2,194 | ~0.008 | ~0.003 | ~2.7× |

Both paths show clear GPU wins. The previous threshold (100,000) never selected
GPU on the MVP default pipeline (~2k points after downsampling).

## Decision

Set `DEFAULT_GPU_MIN_POINTS_PLANE = 2_000` in
`crates/spatialrust-segmentation/src/plane.rs`.

Rationale: MVP-downsampled scenes on the reference sample sit around 2k points
and GPU remains ~2.7× faster; full clouds still benefit strongly. Clouds below
2k fall back to CPU under `Auto` (GPU upload overhead may dominate).

Override per call via `RansacPlaneConfig::gpu_min_points` or force backend with
`ExecutionPolicy::Cpu` / `Gpu` or MVP `--plane-policy`.

## Reproduce

```bash
python bench/ransac_plane/run.py --repeat 3
cargo run --release -p spatialrust --example bench_ransac_plane \
  --features segment-ransac-plane,segment-ransac-plane-gpu,io-pcd,gpu-wgpu -- \
  target/bench-data/table_scene_lms400.pcd --mvp-leaf 0.05 --repeat 3
```

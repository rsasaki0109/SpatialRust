# Epic 69 — GPU Euclidean cluster speed on dense scans

## Problem

On `table_scene_lms400.pcd` (460k pts, tolerance 0.05), grid union-find took ~18–22s
while CPU KD-tree BFS was ~11s. MVP (~1.4k pts) was already faster on grid (~0.7ms).

## Changes

1. **`GpuEuclideanClusterExtractor` adaptive backend** — inputs with
   `len >= DEFAULT_GPU_KDTREE_MIN_POINTS` (50_000) use KD-tree BFS via
   `extract_cpu_roots`; smaller clouds keep parallel grid UF.
2. **Grid UF micro-optimizations** — union uses read-only root lookup during the
   neighbor scan; path compression runs once at the end.
3. **Parallel grid UF** (`spatialrust-search/parallel`, enabled by
   `segment-euclidean-gpu`) — threaded atomic min-root union for clouds ≥ 4096 pts.

## Measured (release, 2026-07-03)

| Path | CPU | GPU | Clusters |
|------|-----|-----|----------|
| Full 460k | 11.04s | 11.05s | 60 / 60 |
| MVP ~1.4k | 0.0009s | 0.0007s | 125 / 125 |

Full-cloud GPU dropped from ~18–22s (grid UF only) to CPU parity via KD-tree routing.
MVP path remains ~1.2× faster on grid UF.

## Tests

```bash
cargo test -p spatialrust-search uniform_grid
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu gpu_matches
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline
python bench/euclidean_cluster/run.py --full-cloud --repeat 1 --warmup 0
```

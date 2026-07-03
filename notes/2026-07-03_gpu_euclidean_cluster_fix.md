# GPU Euclidean cluster correctness fix (Epic 66 / 2026-07-03)

## Problem

Full-cloud bench on `table_scene_lms400.pcd` (460,400 pts, tolerance 0.05):

| Backend | Clusters | Notes |
| --- | ---: | --- |
| CPU | 60 | reference |
| GPU (before) | **1** | label propagation capped at **128** iterations |

## Root cause

Jacobi min-label propagation needs ~component-diameter iterations. The kernel
used `min(128, point_count)`, far too low for large spatial graphs (thin chains
need up to `point_count - 1` hops).

## Fix

- `label_propagation_iterations(dims, point_count)` — `max(grid_span, path_bound)`
  capped at 16_384
- Early exit when labels stabilize (readback every 64 iters)
- CPU root extraction fallback if GPU propagation does not converge

## Verify

```bash
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu gpu_matches
python bench/euclidean_cluster/run.py --full-cloud --repeat 1 --warmup 0
```

## After (Windows release / wgpu)

| Backend | Latency | Clusters | Points |
| --- | ---: | ---: | ---: |
| CPU | 12.29 s | 60 | 460,400 |
| GPU | 75.36 s | **60** | 460,400 |

Correctness restored; GPU latency still needs algorithm work (union-find / fewer
passes) before Auto should prefer GPU at this scale.

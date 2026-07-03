# CPU grid union-find for Euclidean clustering (Epic 68 / 2026-07-03)

## Fable の助言

> 同じ grid グラフを **一箇所**（`spatialrust-search`）に置け。旅人は道を共有し、
> 荷物（実装の二重持ち）だけを捨てよ。

## 変更

- `spatialrust-search/src/uniform_grid.rs` — 共有: `grid_bounds`, `build_grid`, `uniform_grid_fits`, `euclidean_cluster_roots`
- `GpuEuclideanClusterExtractor` + GPU normals/covariance — search モジュールを利用
- **CPU `extract_cpu_roots`** — 460k 計測で KD-tree (~10 s) の方が grid UF (~21 s) より速いため **KD-tree 維持**（Auto CPU 経路）

## ベンチ（460k / tolerance 0.05）

| Backend | 時間 | Clusters | アルゴリズム |
| --- | ---: | ---: | --- |
| CPU | ~10 s | 60 | KD-tree BFS |
| GPU | ~18–21 s | 60 | grid UF |

→ コード統合は完了。CPU を grid に切り替えるのは Epic 69（並列 UF / 真 wgpu）まで保留。

## Verify

```bash
cargo test -p spatialrust-search uniform_grid
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu
python bench/euclidean_cluster/run.py --full-cloud --repeat 1 --warmup 0
```

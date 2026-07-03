# GPU Euclidean cluster speedup — grid union-find (Epic 67 / 2026-07-03)

## Fable の助言

> Jacobi ラベル伝播は「直径の長さだけ歩く旅人」。460k 点の森では 16k 反復も足りず、
> 足りたときは GPU 往復で遅い。**同じグリッドグラフなら Union-Find は一晩で結末がわかる。**

## 変更

WGSL 反復ループ（最大 16_384 pass + readback）を **CPU グリッド Union-Find** に置換。

- `euclidean_cluster_roots_grid` — `build_grid` + path-compression UF（WGSL と同じ 3×3 セル近傍 + 半径チェック）
- `GpuEuclideanClusterExtractor` — `WgpuRuntime` 不要（wgpu 初期化コスト除去）
- `euclidean_cluster_roots_gpu` — 互換ラッパ（`_runtime` 未使用）

## ベンチ（`table_scene_lms400` 460k / tolerance 0.05）

| 段階 | GPU 時間 | Clusters | vs CPU |
| --- | ---: | ---: | ---: |
| Epic 65 Jacobi (128 iter) | ~60 s | 1 ✗ | — |
| Epic 66 Jacobi (16k iter) | ~75 s | 60 ✓ | 0.16× |
| **Epic 67 grid UF** | **~18 s** | **60 ✓** | **~0.56×** |
| CPU KD-tree | ~10 s | 60 | 1.0× |

MVP ~1.4k 点パスも wgpu 起動なしで **数 ms 級**に。

## 次

- CPU `extract_cpu_roots` も grid UF へ統一すれば ~10 s → 数 s の可能性
- 真の wgpu UF / atomicMin は Epic 68+（Auto 閾値見直し前に）

## Verify

```bash
cargo test -p spatialrust-segmentation --features segment-euclidean-gpu gpu_matches
python bench/euclidean_cluster/run.py --full-cloud --repeat 1 --warmup 0
```

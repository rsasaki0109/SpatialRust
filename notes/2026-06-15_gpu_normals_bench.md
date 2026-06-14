# GPU 法線推定 ベンチマーク（2026-06-15、GPU グリッド近傍探索を追記）

## 結論

法線推定には2つの GPU 経路がある（`GpuNormalEstimator`、`feature-normal-gpu`）:

1. **k 近傍モード（`search_radius=None`）**: CPU KD-tree で k 近傍 → GPU で共分散＋固有分解。近傍探索が CPU 支配的で **~1.1x の控えめな優位**にとどまる。
2. **半径モード（`search_radius=Some(r)`）= GPU グリッド近傍探索**: 均一グリッド近傍探索ごと GPU 化。**最大 ~50x**（500k 点で CPU 1.47s → 29 ms）。近傍探索がボトルネックだったため、これを GPU に移して劇的に改善。

→ **半径モード（GPU グリッド）が本命**。グリッド構築（counting sort, O(n)）のみ CPU、近傍ギャザー＋共分散＋Jacobi 固有分解は全て GPU。

## 確認済み事実（中央値）

| 点数 | cpu | gpu (k近傍) | **gpu_grid (半径)** | grid 比 |
|------|-----|------------|--------------------|---------|
| 10k | ~26.5 ms | ~27.5 ms | **~3.5 ms** | **~7.6x** |
| 50k | ~103 ms | ~98 ms | **~5.9 ms** | **~17x** |
| 100k | ~220 ms | ~194 ms | **~8.6 ms** | **~26x** |
| 200k | ~442 ms | ~410 ms | **~15.2 ms** | **~29x** |
| 500k | ~1.47 s | ~1.26 s | **~29.3 ms** | **~50x** |

| 項目 | 内容 |
|------|------|
| ベンチコマンド | `cargo bench -p spatialrust-features --features feature-normal-gpu --bench normals` |
| 計測設定 | criterion `--warm-up-time 0.5 --measurement-time 1.2 --sample-size 10`（簡易計測） |
| 入力 | 合成うねり面、k 近傍モード k=20 / 半径モード r=0.12 |
| GPU グリッド実装 | CPU で bbox＋counting sort（O(n)）→ GPU カーネルで27近接セルの半径ギャザー＋共分散＋Jacobi 固有分解（WGSL）|
| 注意 | グリッドは密配列（`MAX_CELLS=64M`）。点群が広く疎な場合は半径を大きくするか CPU 経路を使用 |

## 次の最適化候補

- グリッド buffer の GPU 常駐化（voxel パイプラインの upload cache と同様）でアップロード往復削減
- 半径ギャザーの k 近傍化（上位 k のみ保持）で近傍定義を CPU 経路と一致
- GPU グリッド近傍探索を登録（GICP/NDT の共分散推定）にも再利用

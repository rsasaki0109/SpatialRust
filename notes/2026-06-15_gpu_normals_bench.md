# GPU 法線推定 ベンチマーク（2026-06-15）

## 結論

GPU 法線推定（`GpuNormalEstimator`、`feature-normal-gpu`）は CPU 版（`NormalEstimator`）に対し **100k〜200k 点で ~1.1〜1.17x 程度の控えめな優位**にとどまる。理由は **近傍探索（CPU KD-tree）が支配的**で、GPU が担うのは per-point の共分散計算＋3x3 Jacobi 固有分解のみだから。さらにバッファのアップロード／リードバック往復のオーバーヘッドがあり、小規模（〜50k）では CPU と同等かわずかに不利。

真の高速化には **近傍探索自体の GPU 化（GPU KD-tree / グリッドハッシュ）** が必要。現状はその前段として、固有分解パートを GPU にオフロードした実装。

## 確認済み事実（中央値）

| 点数 | cpu | gpu | 比 |
|------|-----|-----|----|
| 10k | **~26.5 ms** | ~27.0 ms | ≈同 |
| 50k | **~106 ms** | ~113 ms | ≈同 |
| 100k | ~215 ms | **~193 ms** | GPU ~1.1x |
| 200k | ~432 ms | **~368 ms** | GPU ~1.17x |
| 500k | ~1.32 s | **~1.24 s** | GPU ~1.06x |

| 項目 | 内容 |
|------|------|
| ベンチコマンド | `cargo bench -p spatialrust-features --features feature-normal-gpu --bench normals` |
| 計測設定 | criterion `--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`（簡易計測） |
| 入力 | 合成うねり面（`(x*0.7).sin()*0.1 + (y*0.5).cos()*0.1`）、k=20 近傍 |
| 実装 | CPU で KD-tree k 近傍 → GPU で共分散＋Jacobi 固有分解（WGSL）→ CPU で視点向き付け |
| ボトルネック | CPU 側 k 近傍探索（両者共通コスト）。GPU 化対象は固有分解のみ |

## 次の最適化候補

- GPU KD-tree / 空間ハッシュで近傍探索ごと GPU 化（最大の伸びしろ）
- 近傍インデックスの GPU 常駐化（voxel パイプラインの upload cache と同様）でアップロード往復削減
- 法線推定を voxel ダウンサンプルと同一 GPU パスに統合し往復回数を削減

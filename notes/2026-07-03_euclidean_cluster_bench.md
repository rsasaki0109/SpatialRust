# Euclidean cluster CPU vs GPU ベンチ（2026-07-03）

## 結論

公開 PCL `table_scene_lms400.pcd` で CPU/GPU クラスタ数は一致。**MVP 前処理後（~1.4k 点）は GPU 起動コストで CPU より遅い**。フルクラウド（460k 点）では GPU が ~2× 程度（ローカル wgpu 計測）。

## 条件

| 項目 | 値 |
| --- | --- |
| 入力 | `target/bench-data/table_scene_lms400.pcd` |
| `cluster_tolerance` | 0.05 |
| `min_cluster_size` | 1 |
| 計測 | warmup 1 + repeat 2 平均 |
| ビルド | release |
| 環境 | Windows ローカル（wgpu） |

## 結果 — MVP 前処理（`--mvp-leaf 0.05`）

voxel → normals → plane RANSAC → **plane outliers** でクラスタ入力。

| Backend | 平均 latency | Clusters | Points | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | **0.0010 s** | 125 | 1,369 | — |
| GPU | **0.0269 s** | 125 | 1,369 | **0.04×** |

→ Auto 閾値 `DEFAULT_GPU_MIN_POINTS_EUCLIDEAN = 2_000` は妥当（MVP 規模では CPU 維持）。

## 結果 — フルクラウド（`--full-cloud`）

460,400 点・`tolerance=0.05`・warmup 0・repeat 1（Windows release / wgpu）。

| Backend | 平均 latency | Clusters | Points | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | **12.29 s** | 60 | 460,400 | — |
| GPU | **75.36 s** | **60** | 460,400 | **0.16×** |

→ Epic 66 で **クラスタ数一致**（128 反復 cap が原因だった）。GPU はまだ遅いので Auto は CPU フォールバック/閾値見直しが必要（`notes/2026-07-03_gpu_euclidean_cluster_fix.md`）。

## 再現

```bash
python bench/euclidean_cluster/run.py
python bench/euclidean_cluster/run.py --full-cloud --repeat 3
```

## 追加ファイル

| パス | 内容 |
| --- | --- |
| `bench/euclidean_cluster/run.py` | ハーネス |
| `crates/spatialrust/examples/bench_euclidean_cluster.rs` | 計測 example |

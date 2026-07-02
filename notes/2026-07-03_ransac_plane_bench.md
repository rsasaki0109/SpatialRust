# RANSAC plane CPU vs GPU ベンチ（2026-07-03）

## 結論

公開 PCL `table_scene_lms400.pcd`（460,400 点）で **GPU RANSAC plane が CPU 比 ~11× 高速**。inlier 数も一致（278,059）。MVP への `ExecutionPolicy::Auto` 統合候補。

## 条件

| 項目 | 値 |
| --- | --- |
| 入力 | `target/bench-data/table_scene_lms400.pcd` |
| 点数 | 460,400 |
| `max_iterations` | 1,000 |
| `distance_threshold` | 0.025 |
| `seed` | 42 |
| 計測 | warmup 1 + repeat 3 平均 |
| ビルド | release |
| 環境 | Windows ローカル（wgpu） |

## 結果

| Backend | 平均 latency | Inliers | Speedup |
| --- | ---: | ---: | ---: |
| CPU | **1.9385 s** | 278,059 | — |
| GPU | **0.1764 s** | 278,059 | **~10.99×** |

## 再現

```bash
python bench/ransac_plane/run.py
```

## 判断

- **≥2× 目標を大きく上回る** → MVP パイプラインで GPU plane path を Auto 統合する価値あり
- 次: plane segmentation に `ExecutionPolicy` を追加し、閾値（例: 100k 点以上で GPU）をベンチで確定

## MVP 統合（2026-07-03 追記）

- `MvpPipelineConfig::plane_policy`（default `Auto`）を追加
- `RansacPlaneSegmenter::segment_with_policy` が wgpu 経路を選択（feature: `pipeline-mvp-gpu`）
- Auto 閾値: `DEFAULT_GPU_MIN_POINTS_PLANE = 100_000`
- CLI: `--plane-policy auto|cpu|gpu`

## 追加ファイル

| パス | 内容 |
| --- | --- |
| `bench/ransac_plane/run.py` | ハーネス |
| `crates/spatialrust/examples/bench_ransac_plane.rs` | 計測 example |

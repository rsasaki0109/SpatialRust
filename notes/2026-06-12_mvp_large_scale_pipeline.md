# 実スキャン規模 MVP end-to-end 計測（Epic 39 / 2026-06-12）

## 結論

合成 `point_xyzi` スキャン（平面格子 + 遠方バンプクラスタ）で **500k–2M 点の MVP フルパイプライン**を計測。**GPU voxel（centroid, leaf=4.0）込みで 2M 点 ~102 ms**、CPU ~243 ms（**~2.4×**）。voxel 後点数は ~1.2k（平面）+ 外れ点クラスタで、normals/RANSAC/cluster が支配的だが **数百万点入力でも実用的なレイテンシ**。

## 確認済み事実

### Integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_large_scale_xyzi_pipeline_smoke` | 100k xyzi 格子 → voxel → normals → plane → cluster、label 付き出力 |

### MVP end-to-end ベンチ（leaf=4.0, centroid, `without_gpu_min_points()`）

| 点数 | cpu_full_pipeline | gpu_full_pipeline | GPU/CPU |
|------|-------------------|-------------------|---------|
| 500k | ~58 ms | ~39 ms | **~1.5×** |
| 1M | ~133 ms | ~62 ms | **~2.1×** |
| 2M | ~243 ms | ~102 ms | **~2.4×** |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-pipeline --features pipeline-mvp-gpu --bench mvp_large_scale` |
| 入力 | √(N)×√(N) 平面格子（spacing 0.1 m）+ 100 点バンプ（z=0.5, 格子外） |
| 推論 | xyzi は xyzinormal MVP より軽量。2M でも GPU フルパイプライン <110 ms |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-pipeline/benches/mvp_large_scale.rs` | 500k/1M/2M CPU+GPU ベンチ |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | 100k smoke テスト |

## 未確認/要確認項目

- 実 LAS/COPC ファイル（非合成）での同規模計測
- xyzinormal / 複合スキーマ MVP @2M
- COPC `--resolution` + MVP @数百万点

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC で bounds + resolution 曲線の再現

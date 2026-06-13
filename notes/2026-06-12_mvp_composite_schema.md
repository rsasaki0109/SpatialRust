# 複合スキーマ MVP 検証（Epic 40 / 2026-06-12）

## 結論

`StandardSchemas::point_xyzirgb()`（xyzi + U8 RGB）を追加し、MVP パイプラインが **PCD/LAS roundtrip**・**GPU/CPU voxel 一致**・**100k/500k end-to-end ベンチ**で動作することを確認。LAS 書き込みは export スキーマ（`red/green/blue` U16）と source（`r/g/b` U8）の **semantic フォールバック読み取り**を修正して xyzirgb 入力に対応。

## 確認済み事実

### Integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_composite_xyzirgb_pcd_pipeline_roundtrip` | PCD 入出力、intensity/RGB/normals/label 保持 |
| `mvp_composite_xyzirgb_las_pipeline_roundtrip` | LAS PDRF2 入出力 → MVP → classification 保持 |
| `mvp_composite_xyzirgb_gpu_voxel_matches_cpu` | GPU/CPU voxel 後 xyz・intensity・RGB 一致 |

### MVP end-to-end ベンチ（leaf=4.0, centroid, `without_gpu_min_points()`）

| 点数 | cpu_full_pipeline | gpu_full_pipeline | GPU/CPU |
|------|-------------------|-------------------|---------|
| 100k | ~22 ms | ~30 ms | CPU 優位（小規模オーバーヘッド） |
| 500k | ~98 ms | ~48 ms | **~2.0×** |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-pipeline --features pipeline-mvp-gpu --bench mvp_composite_xyzirgb` |
| スキーマ | `point_xyzirgb`（7 fields: xyz + intensity + rgb U8） |
| LAS 修正 | `cloud_field_name_for_export` + buffer 型ベース readback |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-core/src/schema.rs` | `point_xyzirgb()` |
| `SpatialRust/crates/spatialrust-io/src/las/writer.rs` | semantic/color dtype フォールバック |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | 複合 MVP テスト 3 件 |
| `SpatialRust/crates/spatialrust-pipeline/benches/mvp_composite_xyzirgb.rs` | 100k/500k ベンチ |

## 未確認/要確認項目

- xyzinormal + rgb 7+ フィールド超の LAS 実ファイル
- COPC 入力 + 複合スキーマ MVP
- U16 RGB（LAS 読み込み後）の GPU voxel 最適化

## 次アクション

1. 実 LAS/COPC ファイル end-to-end
2. xyzinormal approximate-first GPU 改善 or スキーマ別 Auto 閾値

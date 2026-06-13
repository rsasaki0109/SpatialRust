# MVP xyzinormal end-to-end 計測（Epic 37 / 2026-06-12）

## 結論

`point_xyzinormal`（xyz + intensity + normal）入力で MVP パイプライン（voxel → normals → RANSAC → cluster）が **PCD roundtrip** および **GPU voxel = CPU voxel** で動作することを確認した。end-to-end ベンチでは **100k 点以下では CPU フルパイプラインが GPU より速い**（GPU ~36 ms vs CPU ~20 ms）。**500k 点では GPU フルパイプライン ~67 ms** で、voxel-only ベンチ（~49 ms @500k）に近いオーバーヘッド。

## 確認済み事実

### Integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_xyzinormal_pcd_pipeline_roundtrip` | PCD 入出力、downsampled/intensity/normals/label 保持 |
| `mvp_xyzinormal_gpu_voxel_matches_cpu` | GPU/CPU voxel 後の xyz・intensity・normal_z 一致（±1e-4） |

### MVP end-to-end ベンチ（leaf=4.0, centroid, `without_gpu_min_points()`）

| 点数 | cpu_full_pipeline | gpu_full_pipeline |
|------|-------------------|-------------------|
| 20k | **~4.2 ms** | — |
| 50k | **~13.7 ms** | — |
| 100k | **~19.7 ms** | ~36.3 ms |
| 500k | — | **~67.3 ms** |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-pipeline --features pipeline-mvp-gpu --bench mvp_xyzinormal` |
| 入力 | 256×256 平面格子 + intensity + (0,0,1) normal |
| 推論 | 100k では normals/RANSAC/cluster が支配的で GPU voxel 恩恵が相殺。500k+ で GPU フルパイプラインが実用的 |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | xyzinormal PCD + GPU/CPU 一致テスト |
| `SpatialRust/crates/spatialrust-pipeline/benches/mvp_xyzinormal.rs` | end-to-end ベンチ |

## 未確認/要確認項目

- 500k/1M 点での CPU フルパイプライン計測
- xyzinormal + rgb 複合スキーマ
- COPC 入力 → MVP（`--resolution` 併用）

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC CLI `--resolution` ベンチ

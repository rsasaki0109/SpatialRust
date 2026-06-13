# 実ファイル LAS/COPC MVP end-to-end（Epic 41 / 2026-06-12）

## 結論

スキャン相当の合成点群（50k xyzi + classification + 外れクラスタ）を **実 `.las` / `.copc.laz` ファイル**に書き出し、読み込み → MVP フルパイプラインが動作することを確認。COPC では **multi-resolution octree**（`max_points_per_node=512`）と **`--resolution` 相当の LOD クエリ**後も MVP が完走する。

## 確認済み事実

### Integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_scan_like_las_file_pipeline` | 50k → `.las` → MVP → `.las` 出力 |
| `mvp_scan_like_copc_file_pipeline` | 50k → `.copc.laz` フル読み込み → MVP |
| `mvp_scan_like_copc_resolution_file_pipeline` | COPC LOD（`spacing×4`）読み込み → MVP |

| 項目 | 内容 |
|------|------|
| 入力 | fract 分布 xyzi + classification（地面=2）+ z=2.5 バンプ 100 点 |
| MVP 設定 | leaf=4.0, centroid, CPU voxel, min_inliers=10 |
| ファイル | temp dir 上の実 LAS/COPC（リポジトリ外） |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | scan-like ヘルパ + 3 テスト |

## 未確認/要確認項目

- 外部配布の実スキャン LAS/LAZ（数百万点・U16 RGB）
- HTTP COPC (`read_copc_url`) + MVP
- CLI `--resolution` on 50k scan-like COPC（Epic 36 は 7k fixture）

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC での CLI `--resolution` ベンチ

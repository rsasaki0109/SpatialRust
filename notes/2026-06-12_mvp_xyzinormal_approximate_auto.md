# MVP xyzinormal approximate-first Auto @1M（Epic 47 / 2026-06-12）

## 結論

Epic 46 の upload pool 最適化後、**xyzinormal + approximate-first + Auto** が **1M 点で GPU voxel path** を選択することを voxel 単体テストで確認し、**MVP end-to-end smoke（1M + outlier bump）** も通過。**500k では Auto=CPU**、**1M では Auto=GPU**（`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY = 1_000_000`）。

## 確認済み事実

### Voxel Auto 閾値テスト

| テスト | 内容 |
|--------|------|
| `auto_approximate_first_uses_cpu_below_heavy_threshold` | 500k xyzinormal approximate → Auto 出力 = CPU |
| `auto_approximate_first_uses_gpu_at_heavy_threshold` | 1M xyzinormal approximate → Auto 出力 = GPU |
| `auto_approximate_first_uses_cpu_for_xyzinormal` | 小点群（128）は gpu_min_points=10 でも Auto=CPU |

### MVP integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_xyzinormal_approximate_auto_1m_smoke` | 1M 格子 + 100 bump、`approximate(4.0)` + `Auto`、plane/cluster/label 出力 |

| 項目 | 内容 |
|------|------|
| 設定 | `VoxelGridDownsampleConfig::approximate(4.0)`（デフォルト gpu 閾値）、`voxel_policy: Auto` |
| fixture | 256×256 格子 1M + z=2.5 bump @x≈90（Epic 41 scan-like パターン） |
| 実行 | `cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline mvp_xyzinormal_approximate_auto_1m_smoke` |
| debug 実行時間 | ~2 s（2026-06-12） |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-filtering/src/voxel.rs` | Auto 閾値 @500k/1M テスト |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | 1M MVP smoke + helpers |

## 未確認/要確認項目

- 外部実スキャン COPC で bounds + resolution 曲線の再現

## 次アクション

1. Epic 46–49 を `push!` でまとめて commit/push
2. 外部実スキャン COPC ファイル提供時に multiplier 曲線を再実行

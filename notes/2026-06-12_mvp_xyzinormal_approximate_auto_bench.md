# MVP xyzinormal approximate-first Auto release ベンチ（Epic 48 / 2026-06-12）

## 結論

**xyzinormal + approximate-first + Auto** の end-to-end release ベンチと **CLI `--voxel-mode approximate --voxel-policy auto`**（1M LAS）を追加。**500k では Auto≈CPU**、**1M/2M では Auto が GPU path を選び CPU より速い**ことを criterion で確認。CLI 統合テスト `mvp_cli_xyzinormal_approximate_auto_1m` も通過。

## 確認済み事実

### Release ベンチ（criterion, leaf=4.0, bump 100 点付き）

| 点数 | Auto | CPU | GPU |
|------|------|-----|-----|
| 500k | ~24 ms | ~25 ms | ~22 ms |
| 1M | ~40 ms | ~50 ms | ~43 ms |
| 2M | ~64 ms | ~100 ms | （Auto≈GPU path、2M gpu 単体は収集中断） |

- Auto @500k ≈ CPU（`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY = 1_000_000` 未満）
- Auto @1M/2M は GPU voxel を選択し、CPU 比 **~20%（1M）〜 ~36%（2M）** 短縮
- 実行: `cargo bench -p spatialrust-pipeline --features pipeline-mvp-gpu --bench mvp_xyzinormal_approximate_auto`

### CLI 統合テスト

| テスト | 内容 |
|--------|------|
| `mvp_cli_xyzinormal_approximate_auto_1m` | 1M xyzinormal LAS → `--leaf-size 4.0 --voxel-mode approximate --voxel-policy auto` → label 付き LAS |

| 項目 | 内容 |
|------|------|
| 実行 | `cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline mvp_cli_xyzinormal_approximate_auto_1m` |
| debug 実行時間 | ~6 s（2026-06-12、LAS 書込込み含む） |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-pipeline/benches/mvp_xyzinormal_approximate_auto.rs` | Auto/CPU/GPU end-to-end ベンチ |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | CLI 1M approximate Auto テスト |

### CLI release 計測（Epic 51 で追加）

| policy | input points | elapsed (release, LAS IO 込み) |
|--------|-------------:|-------------------------------:|
| auto | 1,000,100 | ~72 ms |
| cpu | 1,000,100 | ~61 ms |
| gpu | 1,000,100 | ~193 ms |

- プローブ: `probe_xyzinormal_approximate_auto_cli_release`（`SPATIALRUST_PROBE_RELEASE=1` + `--release`）
- pipeline-only criterion Auto @1M ~40 ms より大きい（IO + デフォルト MVP 設定）

## 未確認/要確認項目

- CLI GPU policy warmup 後の再計測
- 外部実スキャン COPC で bounds + resolution 曲線の再現
- 2M gpu_full_pipeline の criterion 100 サンプル完走

## 次アクション

1. Epic 46–51 を `push!` でまとめて commit/push
2. 外部実スキャン COPC ファイル提供時に multiplier 曲線を再実行

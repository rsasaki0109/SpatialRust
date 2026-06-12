# GPU Voxel Readback 最適化（2026-06-12）

## 結論

GPU filter の staging/readback まわりを整理し、属性 reduce の GPU submit/readback をバッチ化した。**Epic 33** では xyz + 属性を **1 submit / 1 map** に統合し、属性付き filter の ping-pong readback を解消した。再ベンチでは **xyzinormal 500k GPU ~49 ms（Epic 31: ~64 ms、約24%改善）**、**xyzrgb 1M GPU ~89 ms（Epic 31: ~94 ms）**。

## 確認済み事実

| 項目 | 内容 |
|------|------|
| 共有 readback | `SpatialRust/crates/spatialrust-gpu/src/readback.rs` |
| 属性 Average バッチ | `reduce_voxel_average_f32_multi_gpu` — 1 submit / 1 map |
| Epic 33 統合 API | `reduce_voxel_centroids_xyz_and_average_multi_gpu` / `_gather_first_` / `gather_voxel_first_xyz_and_multi_gpu` / `_and_average_multi_gpu` |
| filter 接続 | 属性あり GPU filter は unified readback 経由（xyz-only は従来 pipeline） |
| テスト | `spatialrust-gpu` 18 passed / `spatialrust-filtering` 10 passed |

### Epic 33 再ベンチ（centroid, leaf=4.0）

| 点数 | スキーマ | Epic 31 GPU | Epic 33 GPU |
|------|----------|------------|------------|
| 500k | xyzrgb | ~59 ms | ~61 ms |
| 1M | xyzrgb | ~94 ms | **~89 ms** |
| 500k | xyzinormal | ~64 ms | **~49 ms** |

### 変更ファイル

| パス | 変更 |
|------|------|
| `crates/spatialrust-gpu/src/readback.rs` | `split_xyz_and_attribute_blocks` |
| `crates/spatialrust-gpu/src/kernels/voxel_reduce.rs` | unified xyz+attr reduce readback |
| `crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | unified xyz+attr gather readback |
| `crates/spatialrust-filtering/src/voxel.rs` | 属性あり GPU path を unified readback に接続 |

## 未確認/要確認項目

- U8 RGB 専用 reduce kernel（現状 F32 変換 upload のオーバーヘッド残存）
- approximate-first × 属性多数 end-to-end

## 次アクション

1. U8 RGB 専用 reduce / 単一 staging 後の再ベンチ
2. 多解像度 COPC 実ファイルでの `--resolution` 点数削減効果

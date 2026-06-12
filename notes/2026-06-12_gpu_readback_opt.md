# GPU Voxel Readback 最適化（2026-06-12）

## 結論

GPU filter の staging/readback まわりを整理し、属性 reduce の GPU submit/readback をバッチ化した。**Epic 33** では xyz + 属性を **1 submit / 1 map** に統合。**Epic 35** では U8 属性を packed upload + 専用 kernel + mixed f32/u8 staging readback に拡張し、xyzrgb 1M GPU を **~89 ms → ~36 ms** に短縮した。

## 確認済み事実

| 項目 | 内容 |
|------|------|
| 共有 readback | `SpatialRust/crates/spatialrust-gpu/src/readback.rs` |
| 属性 Average バッチ | `reduce_voxel_average_f32_multi_gpu` — 1 submit / 1 map |
| Epic 33 統合 API | `reduce_voxel_centroids_xyz_and_average_multi_gpu` / `_gather_first_` / `gather_voxel_first_xyz_and_multi_gpu` / `_and_average_multi_gpu`（Epic 35: `u8_attribute_channels` 追加） |
| Epic 35 U8 | `voxel_reduce_u8.wgsl` / `voxel_gather_u8.wgsl` + `read_staging_f32_and_u8` |
| filter 接続 | F32/U8 属性を partition して unified readback 経由 |
| テスト | `spatialrust-gpu` 19 passed / `spatialrust-filtering` 11 passed |

### Epic 33 再ベンチ（centroid, leaf=4.0）

| 点数 | スキーマ | Epic 31 GPU | Epic 33 GPU |
|------|----------|------------|------------|
| 500k | xyzrgb | ~59 ms | ~61 ms |
| 1M | xyzrgb | ~94 ms | **~36 ms** (Epic 35) |
| 500k | xyzinormal | ~64 ms | **~49 ms** |

### 変更ファイル

| パス | 変更 |
|------|------|
| `crates/spatialrust-gpu/src/readback.rs` | `split_xyz_and_attribute_blocks` |
| `crates/spatialrust-gpu/src/kernels/voxel_reduce.rs` | unified xyz+attr reduce readback |
| `crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | unified xyz+attr gather readback |
| `crates/spatialrust-filtering/src/voxel.rs` | 属性あり GPU path を unified readback に接続 |

## 未確認/要確認項目

- U8+F32 混在スキーマ end-to-end
- approximate-first × 属性多数 end-to-end

## 次アクション

1. 多解像度 COPC 実ファイルでの `--resolution` 点数削減効果

# GPU U8 RGB 専用 reduce/gather kernel（Epic 35 / 2026-06-12）

## 結論

U8 属性（xyzrgb の r/g/b）を **F32 変換 upload なし**で GPU reduce/gather する専用 kernel と unified readback を実装した。xyzrgb centroid ベンチでは **500k GPU ~22.6 ms（Epic 33: ~61 ms、~63% 短縮）**、**1M GPU ~36 ms（Epic 33: ~89 ms、~60% 短縮）**となり、CPU 比 **~2.1× / ~2.9×** まで GPU 優位を回復した。

## 確認済み事実

| 点数 | スキーマ | cpu_centroid | gpu_centroid (Epic 35) | GPU/CPU |
|------|----------|-------------|------------------------|---------|
| 500k | xyzrgb | ~48 ms | **~22.6 ms** | **~2.1×** |
| 1M | xyzrgb | ~104 ms | **~36 ms** | **~2.9×** |

| 項目 | 内容 |
|------|------|
| WGSL | `voxel_reduce_u8.wgsl` / `voxel_gather_u8.wgsl`（packed `array<u32>` 入力、セル出力は u32 下位 byte） |
| Rust glue | `record_voxel_reduce_u8_pass` / `record_voxel_gather_u8_pass`、unified API に `u8_attribute_channels` 追加 |
| readback | `pad_u8_for_gpu_storage` / `unpack_u8_outputs_from_u32_staging` / `read_staging_f32_and_u8` |
| filter | U8/F32 属性を分離して native u8 GPU path、`gpu_policy_averages_u8_rgb_on_gpu` テスト追加 |
| テスト | `spatialrust-gpu` 19 passed / `spatialrust-filtering` 11 passed |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_reduce_u8.wgsl` | U8 average kernel |
| `SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_gather_u8.wgsl` | U8 first kernel |
| `SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_reduce.rs` | u8 pass + mixed readback |
| `SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | 同上 |
| `SpatialRust/crates/spatialrust-filtering/src/voxel.rs` | U8 属性 partition + GPU 接続 |

## 未確認/要確認項目

- U8 + F32 混在スキーマ（例: rgb + intensity）の end-to-end ベンチ
- approximate-first × xyzrgb
- GPU 出力 u32 セルバッファの readback サイズ（現状 cells×4 byte staging）

## 次アクション

1. 多解像度 COPC 実ファイル `--resolution` 検証
2. xyzinormal / 複合スキーマ MVP end-to-end 計測
3. approximate-first × 属性多数ベンチ

# GPU Voxel Readback 最適化（2026-06-12）

## 結論

GPU filter の staging/readback まわりを整理し、属性 reduce の GPU submit/readback をバッチ化した。xyz チャンネル分割の余分な `Vec` コピーを削減し、filter 出力組み立てを per-point `push` から一括 `set_field_from_f32` に変更した。

## 確認済み事実

| 項目 | 内容 |
|------|------|
| 共有 readback | `SpatialRust/crates/spatialrust-gpu/src/readback.rs` |
| 属性 Average バッチ | `reduce_voxel_average_f32_multi_gpu` — 1 submit / 1 map |
| filter 出力 | `set_field_from_f32` で bulk insert |
| テスト | `spatialrust-gpu` 17 passed / `spatialrust-filtering` 9 passed |

### 変更ファイル

| パス | 変更 |
|------|------|
| `crates/spatialrust-gpu/src/readback.rs` | 新規 |
| `crates/spatialrust-gpu/src/kernels/voxel_reduce.rs` | multi reduce + split 最適化 |
| `crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | 共有 readback / split |
| `crates/spatialrust-filtering/src/voxel.rs` | multi reduce + bulk 出力 |

## 未確認/要確認項目

- 500k 点 end-to-end ベンチの改善率（Epic 30 で再計測済み: centroid GPU ~51 ms vs CPU ~94 ms）
- xyz reduce と属性 reduce の readback 完全統合（1 staging）

## 次アクション

1. 2M 点 approximate-first クロスオーバー計測
2. xyz + 属性 readback 完全統合

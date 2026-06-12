# GPU Voxel 属性多数ベンチ（Epic 31 / 2026-06-12）

## 結論

500k–1M 点・centroid・leaf=4.0・`without_gpu_min_points()` で、**F32 属性（intensity / normals）は CPU コスト増に対し GPU 優位は維持**（1M xyzinormal: CPU ~193 ms vs GPU ~99 ms）。一方 **U8 RGB（xyzrgb）は GPU 相対性能を大きく悪化**（1M: CPU ~143 ms vs GPU ~94 ms、同点数 xyz GPU ~32 ms と比較して readback/集約コストが支配的）。Epic 33（xyz + 属性 readback 統合）の優先度を上げる根拠になる。

## 確認済み事実

| 点数 | スキーマ | cpu_centroid | gpu_centroid | GPU/CPU |
|------|----------|-------------|-------------|---------|
| 500k | xyz | ~63 ms | ~26 ms | **~2.4×** |
| 500k | xyzi | ~80 ms | ~35 ms | **~2.3×** |
| 500k | xyzrgb | ~82 ms | ~59 ms | **~1.4×** |
| 500k | xyzinormal | ~124 ms | ~64 ms | **~1.9×** |
| 1M | xyz | ~119 ms | ~32 ms | **~3.7×** |
| 1M | xyzi | ~120 ms | ~46 ms | **~2.6×** |
| 1M | xyzrgb | ~143 ms | ~94 ms | **~1.5×** |
| 1M | xyzinormal | ~193 ms | ~99 ms | **~2.0×** |

| 項目 | 結果 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample_attributes` |
| 計測日 | 2026-06-12 |
| スキーマ | `point_xyz` / `point_xyzi` / `point_xyzrgb` / `point_xyzinormal` |
| 推論 | U8 RGB は GPU 属性 reduce/readback がボトルネック。F32 属性追加は CPU 側の増分が大きく GPU 優位は残る |

## 未確認/要確認項目

- xyzrgb GPU パスで U8 専用 kernel / 単一 readback にした場合の改善率
- RGB+intensity+normals 複合スキーマ（実 LAS 相当）での end-to-end
- approximate-first × 属性多数

## 次アクション

1. xyz + 属性 readback 完全統合（U8 RGB 含む）
2. MVP end-to-end で xyzinormal 入力の計測

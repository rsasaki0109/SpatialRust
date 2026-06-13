# GPU Voxel 属性多数ベンチ（Epic 31 / 2026-06-12）

## 結論

500k–1M 点・centroid・leaf=4.0・`without_gpu_min_points()` で、**F32 属性（intensity / normals）は CPU コスト増に対し GPU 優位は維持**（1M xyzinormal: CPU ~193 ms vs GPU ~99 ms）。**U8 RGB（xyzrgb）は Epic 35 の専用 kernel により GPU 優位を回復**（1M: CPU ~104 ms vs GPU ~36 ms、**~2.9×**）。

## 確認済み事実

| 点数 | スキーマ | cpu_centroid | gpu_centroid | GPU/CPU |
|------|----------|-------------|-------------|---------|
| 500k | xyz | ~63 ms | ~26 ms | **~2.4×** |
| 500k | xyzi | ~80 ms | ~35 ms | **~2.3×** |
| 500k | xyzrgb | ~82 ms | **~22.6 ms** (Epic 35) | **~2.1×** |
| 500k | xyzinormal | ~124 ms | **~49 ms** (Epic 33) | **~2.5×** |
| 1M | xyz | ~119 ms | ~32 ms | **~3.7×** |
| 1M | xyzi | ~120 ms | ~46 ms | **~2.6×** |
| 1M | xyzrgb | ~143 ms | **~36 ms** (Epic 35) | **~2.9×** |
| 1M | xyzinormal | ~193 ms | ~99 ms (Epic 31) | **~2.0×** |

| 項目 | 結果 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample_attributes` |
| 計測日 | 2026-06-12 |
| スキーマ | `point_xyz` / `point_xyzi` / `point_xyzrgb` / `point_xyzinormal` |
| 推論 | U8 RGB は Epic 35 専用 kernel で GPU 優位を回復。F32 属性追加は CPU 側の増分が大きく GPU 優位は残る |

## 未確認/要確認項目

- RGB+intensity+normals 複合スキーマ（実 LAS 相当）での end-to-end
- xyzinormal approximate-first GPU ボトルネック（Epic 38: 全規模 CPU 優位、Epic 42: Auto 閾値で回避）

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC で bounds + resolution 曲線の再現

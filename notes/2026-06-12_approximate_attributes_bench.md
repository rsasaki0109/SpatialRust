# approximate-first × 属性多数ベンチ（Epic 38 / 2026-06-12）

## 結論

500k–2M 点・leaf=4.0・`without_gpu_min_points()` で **centroid は全スキーマで 500k 以上 GPU 優位を維持**。**approximate-first は xyz / xyzi / xyzrgb では 1M 前後から GPU 優位**（U8 RGB は gather が軽く 500k でも GPU ≈ CPU）。**xyzinormal（F32×4 属性）は approximate-first GPU が全規模で CPU より遅い**（2M: CPU ~93 ms vs GPU ~192 ms）。`DEFAULT_GPU_MIN_POINTS_APPROXIMATE = 2_000_000` は position-only / 軽属性には妥当だが、**重属性スキーマでは Auto 閾値の見直しが必要**。

## 確認済み事実

### centroid（GPU/CPU 比 @500k / 1M / 2M）

| 点数 | xyz | xyzi | xyzrgb | xyzinormal |
|------|-----|------|--------|------------|
| 500k | ~2.1× | ~2.1× | ~2.1× | ~2.1× |
| 1M | ~3.6× | ~3.1× | ~4.0× | ~2.4× |
| 2M | ~4.3× | ~3.9× | ~5.3× | ~2.3× |

### approximate-first（代表 median ms、太字=優位側）

| 点数 | スキーマ | cpu_approx | gpu_approx | 優位 |
|------|----------|-----------|------------|------|
| 500k | xyz | ~27 | ~30 | CPU |
| 500k | xyzi | ~26 | ~36 | CPU |
| 500k | xyzrgb | ~27 | ~28 | ≈同等 |
| 500k | xyzinormal | **~32** | ~82 | **CPU ~2.6×** |
| 1M | xyz | ~38 | **~27** | **GPU ~1.4×** |
| 1M | xyzi | ~51 | **~40** | **GPU ~1.3×** |
| 1M | xyzrgb | ~65 | **~43** | **GPU ~1.5×** |
| 1M | xyzinormal | **~66** | ~141 | **CPU ~2.1×** |
| 2M | xyz | ~113 | **~61** | **GPU ~1.9×** |
| 2M | xyzi | ~101 | **~79** | **GPU ~1.3×** |
| 2M | xyzrgb | ~100 | **~57** | **GPU ~1.8×** |
| 2M | xyzinormal | **~93** | ~192 | **CPU ~2.1×** |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample_attributes` |
| 計測日 | 2026-06-12 |
| 変更 | ベンチに `cpu/gpu_approximate_first` と 2M 点数を追加 |
| 推論 | approximate-first GPU は gather + 多 ch F32 readback コストが大きい。U8 RGB は Epic 35 kernel で損失が小さい |

## 未確認/要確認項目

- xyzinormal approximate-first GPU ボトルネック（gather ch 数 vs readback サイズ）のプロファイル
- 実スキャン LAS（xyzi+rgb 複合）での end-to-end

## 次アクション

1. 外部実スキャン COPC で bounds + resolution 曲線の再現
2. GPU attribute buffer キャッシュ / pinned upload（Epic 45 参照）

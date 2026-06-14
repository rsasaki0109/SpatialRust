# approximate-first xyzinormal GPU 最適化（Epic 45 / 2026-06-12）

## 結論

Epic 38 の **xyzinormal approximate-first GPU 劣後**に対し、属性 gather を **multi4 バッチ化**し、**xyz + F32×4 を 1 dispatch / 1 staging copy** に融合する `voxel_gather_xyz_attrs4.wgsl` を追加。**1M で GPU ~124 ms（Epic 38 ~141 ms 比 ~12% 改善）** だが **500k/2M では計測分散あり、CPU 優位（~2×）は解消せず**。Auto 閾値（Epic 42）は維持。

## 確認済み事実

### 変更

| 項目 | 内容 |
|------|------|
| 融合 kernel | `voxel_gather_xyz_attrs4.wgsl` — 1 cell あたり xyz + 4 attrs を packed output へ |
| バッチ gather | `record_gather_f32_attribute_channels_to_staging` — multi2/multi4 encoder 記録 |
| 接続 | `gather_voxel_first_xyz_and_multi_gpu` が 4 F32 属性時に融合 path を選択 |
| テスト | `gpu_approximate_first_xyzinormal_matches_cpu_downsample` |

### 再ベンチ（centroid, leaf=4.0, `without_gpu_min_points()`）

| 点数 | cpu_approx | gpu_approx (Epic 45) | Epic 38 gpu_approx | 優位 |
|------|-----------:|---------------------:|-------------------:|------|
| 500k | ~32 ms | **~90 ms** | ~82 ms | CPU ~2.8× |
| 1M | ~59 ms | **~124 ms** | ~141 ms | CPU ~2.1× |
| 2M | ~115 ms | **~245 ms** | ~192 ms | CPU ~2.1× |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample_attributes` |
| 推論 | dispatch 統合は有効だが、毎回の CPU→GPU 属性 upload と MAP_READ が支配的。2M は計測分散あり |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_gather_xyz_attrs4.wgsl` | 融合 gather shader |
| `SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | 融合 path + batched attr gather |
| `SpatialRust/crates/spatialrust-gpu/src/pipeline_cache.rs` | xyz_attrs4 pipeline |
| `SpatialRust/crates/spatialrust-filtering/src/voxel.rs` | xyzinormal GPU/CPU 一致テスト |

## 未確認/要確認項目

- 永続 GPU attribute buffer（upload 削減）での crossover 点
- MVP `--voxel-mode approximate` + xyzinormal 実計測

## 次アクション

1. 外部実スキャン COPC で bounds + resolution 曲線の再現
2. ~~GPU attribute buffer キャッシュ~~ → Epic 46 で upload pool 実装済み

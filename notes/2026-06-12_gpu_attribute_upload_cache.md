# GPU attribute upload cache + zero-copy ベンチ（Epic 46 / 2026-06-12）

## 結論

Epic 45 後も残っていた **per-call 属性 CPU コピー + `create_buffer_init`** を解消。**F32 属性の slice 借用**（`borrow_attribute_f32_channels`）と **`GpuUploadPool`（`write_buffer` + サイズ別再利用）** により、xyzinormal approximate-first GPU が **1M+ で CPU 優位**に反転（2M: **~62 ms vs ~122 ms**）。Auto 閾値を **`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY = 1_000_000`** に更新（Epic 42 の `usize::MAX` から復帰）。

## 確認済み事実

### 変更

| 項目 | 内容 |
|------|------|
| Upload pool | `SpatialRust/crates/spatialrust-gpu/src/upload_cache.rs` |
| 位置 upload | `compute_voxel_keys_gpu_buffers` → `upload_f32_storage` + `GpuVoxelKeyBuffers::recycle` |
| 属性 upload | gather/融合 path で pool 使用、submit 後 recycle |
| CPU 回避 | `borrow_attribute_f32_channels` で PointCloud F32 フィールドを直接参照 |
| Auto 閾値 | heavy approximate: **1M**（500k では CPU ≈ GPU、1M+ で GPU 優位） |

### 再ベンチ（approximate-first xyzinormal, leaf=4.0）

| 点数 | cpu_approx | gpu_approx (Epic 46) | Epic 45 gpu_approx | 優位 |
|------|-----------:|---------------------:|-------------------:|------|
| 500k | ~30 ms | **~34 ms** | ~90 ms | ≈同等 |
| 1M | ~59 ms | **~54 ms** | ~124 ms | **GPU ~1.1×** |
| 2M | ~122 ms | **~62 ms** | ~245 ms | **GPU ~2.0×** |

| 項目 | 内容 |
|------|------|
| ベンチ | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample_attributes` |
| テスト | `spatialrust-gpu` 20 passed / `spatialrust-filtering` 14 passed |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-gpu/src/upload_cache.rs` | 新規 |
| `SpatialRust/crates/spatialrust-gpu/src/runtime.rs` | pool API |
| `SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs` | pooled upload + recycle |
| `SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_keys.rs` | pooled position upload + recycle |
| `SpatialRust/crates/spatialrust-filtering/src/voxel.rs` | slice borrow + Auto 閾値 |

## 未確認/要確認項目

- MVP `--voxel-mode approximate` + xyzinormal @1M Auto 実計測
- 外部実スキャン COPC bounds + resolution 曲線

## 次アクション

1. 外部実スキャン COPC で bounds + resolution 曲線の再現
2. MVP CLI approximate Auto release ベンチ（Epic 47: library/MVP smoke 確認済み）

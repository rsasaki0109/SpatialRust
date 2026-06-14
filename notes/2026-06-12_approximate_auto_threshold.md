# approximate-first スキーマ別 Auto 閾値（Epic 42 / 2026-06-12、Epic 46 更新）

## 結論

Epic 38 で **xyzinormal approximate-first GPU が全規模 CPU 劣後**だった問題に対し、Epic 42 で `effective_gpu_min_points()` を追加し heavy schema では Auto=CPU を強制。**Epic 46** の upload pool + zero-copy 属性借用後、1M+ で GPU 優位が復活したため、heavy approximate の Auto 閾値を **`DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY = 1_000_000`** に設定（Epic 42 の `usize::MAX` を解除）。

## 確認済み事実

| 項目 | 内容 |
|------|------|
| 定数 | `APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS = 4` |
| 定数 | `DEFAULT_GPU_MIN_POINTS_APPROXIMATE_HEAVY = 1_000_000` |
| API | `VoxelGridDownsampleConfig::effective_gpu_min_points(schema)` |
| Auto 挙動 | xyz: approximate 閾値 2M、xyzinormal approximate: **1M** |
| テスト | `effective_gpu_min_points_blocks_heavy_approximate_schema` |
| テスト | `auto_approximate_first_uses_cpu_for_xyzinormal`（小点群） |
| テスト | **`auto_approximate_first_uses_cpu_below_heavy_threshold`（500k、Epic 47）** |
| テスト | **`auto_approximate_first_uses_gpu_at_heavy_threshold`（1M、Epic 47）** |

| スキーマ | approximate Auto GPU 閾値 |
|----------|---------------------------|
| `point_xyz` | 2_000_000（従来通り） |
| `point_xyzi` | 2_000_000 |
| `point_xyzrgb` | 2_000_000 |
| `point_xyzinormal` | **1_000_000**（Epic 46） |

## 未確認/要確認項目

- MVP CLI `--voxel-mode approximate` + `--voxel-policy auto` on 1M xyzinormal release 計測
- U16 RGB（LAS 読み込み後）の heavy 判定拡張

## 次アクション

1. 外部実スキャン COPC で bounds + resolution 曲線の再現
2. MVP CLI approximate Auto release ベンチ

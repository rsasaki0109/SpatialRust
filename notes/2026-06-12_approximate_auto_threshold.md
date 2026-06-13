# approximate-first スキーマ別 Auto 閾値（Epic 42 / 2026-06-12）

## 結論

Epic 38 で **xyzinormal approximate-first GPU が全規模 CPU 劣後**だった問題に対し、`VoxelGridDownsampleConfig::effective_gpu_min_points()` を追加。**非 position F32 属性が 4 本以上**（`point_xyzinormal`: intensity + normal×3）の approximate-first では **Auto が GPU を選ばない**（effective 閾値 = `usize::MAX`）。centroid / 軽属性 approximate-first の既存閾値（500k / 2M）は維持。

## 確認済み事実

| 項目 | 内容 |
|------|------|
| 定数 | `APPROXIMATE_HEAVY_F32_ATTRIBUTE_CHANNELS = 4` |
| API | `VoxelGridDownsampleConfig::effective_gpu_min_points(schema)` |
| Auto 挙動 | xyz: approximate 閾値 2M、xyzinormal approximate: 常に CPU |
| テスト | `effective_gpu_min_points_blocks_heavy_approximate_schema` |
| テスト | `auto_approximate_first_uses_cpu_for_xyzinormal`（gpu_min_points=10 でも Auto=CPU） |

| スキーマ | approximate Auto GPU 閾値 |
|----------|---------------------------|
| `point_xyz` | 2_000_000（従来通り） |
| `point_xyzi` | 2_000_000 |
| `point_xyzrgb` | 2_000_000 |
| `point_xyzinormal` | **無効（usize::MAX）** |

## 未確認/要確認項目

- approximate-first GPU gather/readback の根本最適化（xyzinormal で GPU 優位を取り戻す）
- U16 RGB（LAS 読み込み後）の heavy 判定拡張
- MVP CLI `--voxel-mode approximate` + `--voxel-policy auto` on xyzinormal 実計測

## 次アクション

1. 外部実スキャン COPC CLI `--resolution` ベンチ
2. approximate-first xyzinormal GPU kernel/readback 最適化

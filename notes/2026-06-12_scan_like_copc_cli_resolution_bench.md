# scan-like COPC CLI `--resolution` ベンチ（Epic 43 / 2026-06-12）

## 結論

Epic 41 の **50k scan-like xyzi** 点群（+100 バンプ）を multi-resolution COPC（`max_points_per_node=512`）に書き出し、**MVP CLI `--resolution spacing×4`** で **50100 → 512 点（~99% 削減）** を確認。**release** 計測では coarse **~1.3 ms** vs full **~11 ms**（MVP 込み end-to-end）。Epic 36 の 7k 格子 fixture より大規模でも LOD + CLI が有効。

## 確認済み事実

### CLI integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_cli_scan_like_copc_resolution_reduces_input_points` | 50k scan-like COPC、`--leaf-size 4.0 --voxel-policy cpu` |

### 計測（2026-06-12、leaf=4.0、centroid、CPU voxel）

| モード | input points | MVP elapsed (release) |
|--------|-------------|-------------------------|
| full COPC | 50,100 | ~11 ms |
| `--resolution spacing×4` | **512** | **~1.3 ms** |

| 項目 | 内容 |
|------|------|
| octree spacing | ~0.35 m |
| coarse resolution | ~1.41 m（spacing×4） |
| 削減率 | ~99%（512 / 50100） |
| テスト | `cargo test -p spatialrust --features mvp mvp_cli_scan_like_copc_resolution` |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | CLI scan-like resolution テスト + elapsed パーサ |

## 未確認/要確認項目

- 外部配布の実スキャン COPC（本番 octree）での同等 LOD 曲線
- HTTP COPC リモート URL + `--resolution`

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC で bounds + resolution 曲線の再現（Epic 44 で合成 50k 確認済み）

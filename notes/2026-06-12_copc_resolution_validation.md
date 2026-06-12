# 多解像度 COPC `--resolution` 検証（Epic 36 / 2026-06-12）

## 結論

7,000 点の多階層 COPC フィクスチャ（`max_points_per_node=96`）で、`--resolution` / `CopcQuery::with_resolution` により **読み込み点数が単調に制御できる**ことを確認した。粗い LOD（`spacing×4`）では **96 点（~98.6% 削減）**、中間（`spacing`）も 96 点、やや細かい（`spacing/4`）では **3,936 点**、フル読み込みは **7,000 点**。MVP CLI `--resolution` でも `input points:` がフルより少なくなることを integration test で確認。

## 確認済み事実

| 読み込みモード | 点数 | 備考 |
|----------------|------|------|
| full | 7,000 | `read_copc_file` |
| `--resolution spacing×4` | 96 | CLI + `CopcQuery::with_resolution` |
| `--resolution spacing` | 96 | level 0 のみ（本フィクスチャ） |
| `--resolution spacing/4` | 3,936 | level 0–2 相当 |
| `with_level(0)` | 96 | 明示 level でも一致 |
| `with_level(2)` | 3,936 | resolution 細かめと一致 |

| 項目 | 内容 |
|------|------|
| 新 API | `write_copc_file_with_params` / `CopcWriterParams` 公開 |
| IO テスト | `multi_resolution_copc_resolution_query_reduces_point_count` |
| MVP テスト | `mvp_copc_resolution_query_pipeline`（7k 多階層 fixture） |
| CLI テスト | `mvp_cli_copc_resolution_reduces_input_points` |
| フィクスチャ | 31×29×23 格子 7,000 点、`CopcWriterParams { max_points_per_node: 96, max_depth: 8 }` |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust-io/src/copc/writer.rs` | `write_copc_file_with_params` |
| `SpatialRust/crates/spatialrust-io/src/copc/reader.rs` | 多解像度 readback テスト |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | MVP + CLI resolution 検証 |

## 未確認/要確認項目

- 実スキャン由来 COPC（LAS 変換・本番 octree）での `--resolution` 効果
- `--bounds` と `--resolution` 併用時の end-to-end レイテンシ
- HTTP COPC (`read_copc_url_with_query`) での同等 LOD 挙動

## 次アクション

1. approximate-first × 属性多数ベンチ
2. 実スキャン規模（数百万点）end-to-end

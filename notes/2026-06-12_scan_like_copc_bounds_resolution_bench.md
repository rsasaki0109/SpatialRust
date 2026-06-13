# scan-like COPC CLI `--bounds` + `--resolution` ベンチ（Epic 44 / 2026-06-12）

## 結論

Epic 41 の **50k scan-like xyzi** COPC に対し、ROI **`--bounds 0,0,-0.01,40,20,0.5`**（主格子の約半分）と **`--resolution spacing×4`** を併用。**50100 → 5267（bounds のみ）→ 46 点（併用、~99.9% 削減）** を library query と CLI の両方で一致確認。**release** 計測では combined **~0.27 ms** vs bounds-only **~1.5 ms** vs full **~9.2 ms**（`--leaf-size 4.0 --voxel-policy cpu`）。

## 確認済み事実

### CLI integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_cli_scan_like_copc_bounds_resolution_reduces_input_points` | bounds + resolution 併用、bounds-only、full の 3 モード比較 |
| `write_scan_like_copc_fixture` | 50k scan-like COPC fixture 共通化（Epic 43/44） |

### 計測（2026-06-12、leaf=4.0、centroid、CPU voxel）

| モード | input points | MVP elapsed (release) |
|--------|-------------:|----------------------:|
| full COPC | 50,100 | ~9.2 ms |
| `--bounds` のみ（ROI 半領域） | 5,267 | ~1.5 ms |
| `--bounds` + `--resolution spacing×4` | **46** | **~0.27 ms** |

| 項目 | 内容 |
|------|------|
| ROI bounds | `0,0,-0.01,40,20,0.5`（x/y 半分、低 z 帯） |
| coarse resolution | spacing×4（Epic 43 と同じ） |
| 削減率 | full→combined ~99.9%（46 / 50100） |
| テスト | `cargo test -p spatialrust --features mvp mvp_cli_scan_like_copc_bounds_resolution` |

### 副次修正

| 項目 | 内容 |
|------|------|
| `parse_duration_debug_ms` | release の `µs` 表記が `s` suffix に誤マッチしていたため、`µs`/`us` を `s` より先に判定 |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | bounds+resolution CLI テスト、fixture helper、elapsed パーサ修正 |

## 未確認/要確認項目

- 外部配布の実スキャン COPC で ROI + LOD 併用時の octree ヒット率
- HTTP COPC リモート URL + `--bounds` + `--resolution`

## 次アクション

1. approximate-first xyzinormal GPU kernel/readback 最適化
2. 外部実スキャン COPC で bounds + resolution 曲線の再現

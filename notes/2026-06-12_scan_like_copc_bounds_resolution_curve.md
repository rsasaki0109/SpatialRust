# scan-like COPC bounds + resolution 曲線（Epic 49 / 2026-06-12）

## 結論

Epic 41/43/44 の **50k scan-like xyzi COPC** に対し、**`spacing×{1,2,4,8,16}`** の resolution 曲線と **ROI bounds 併用曲線**を library + CLI で検証。**点数は multiplier 增大に対し単調非増加**（本 fixture では x1 時点で飽和し平坦）。CLI の `input points:` は library query と **5 段すべて一致**。

## 確認済み事実

### 点数曲線（library query, spacing≈0.35 m）

| multiplier | root+resolution | roi+resolution |
|-----------:|----------------:|---------------:|
| full / roi-only | 50,100 | 5,267 |
| ×1 | 512 | 46 |
| ×2 | 512 | 46 |
| ×4 | 512 | 46 |
| ×8 | 512 | 46 |
| ×16 | 512 | 46 |

- root 側は x1 時点で **512 点**（Epic 43 の spacing×4 相当）に飽和
- roi 側は x1 時点で **46 点**（Epic 44 の bounds+resolution 併用相当）に飽和
- `max_points_per_node=512` octree では finer multiplier でも点数は増えない（平坦曲線）

### Release CLI 代表点（leaf=4.0, centroid, CPU voxel）

| モード | input points | elapsed |
|--------|-------------:|--------:|
| full | 50,100 | ~6.2 ms |
| `--bounds` のみ | 5,267 | ~1.1 ms |
| `--bounds` + `--resolution spacing×4` | 46 | ~0.20 ms |

### Integration テスト

| テスト | 内容 |
|--------|------|
| `mvp_scan_like_copc_bounds_resolution_curve_monotonic` | library 曲線の単調非増加 |
| `mvp_cli_scan_like_copc_bounds_resolution_curve` | CLI 5 段 × root/roi、library 一致 + 単調非増加 |
| `probe_scan_like_copc_resolution_curve_counts` | `#[ignore]` 手動プローブ（点数表 + 任意 release 計測） |

| 項目 | 内容 |
|------|------|
| 実行 | `cargo test -p spatialrust --features mvp bounds_resolution_curve` |
| release プローブ | `SPATIALRUST_PROBE_RELEASE=1 cargo test -p spatialrust --features mvp --release --test mvp_pipeline probe_scan_like_copc_resolution_curve_counts -- --ignored --nocapture` |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | 曲線 helpers + 2 テスト + ignore プローブ |

## 未確認/要確認項目

- 外部配布の実スキャン COPC（本番 octree）での multiplier 曲線（飽和点・段差の有無）
- HTTP COPC + bounds + resolution

## 次アクション

1. Epic 46–50 を `push!` でまとめて commit/push
2. 外部実スキャン COPC ファイル提供時に multiplier 曲線を再実行

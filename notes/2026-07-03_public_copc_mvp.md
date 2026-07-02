# 公開 COPC + MVP 再現（Epic 60 / 2026-07-03）

## 結論

公開 PCL [`table_scene_lms400.pcd`](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd)（460,400 点）を COPC に書き出し、**bounds / resolution クエリ** と **MVP パイプライン** が end-to-end で動作することを確認した。再現ハーネスは `bench/public_copc/run.py`。

## 確認済み事実

### データセット

| 項目 | 値 |
| --- | --- |
| ソース | PCL `table_scene_lms400.pcd` |
| キャッシュ | `target/bench-data/table_scene_lms400.pcd` |
| 全点数 | 460,400 |

### COPC クエリ（release, Windows ローカル 1 回）

| クエリ | 点数 | 削減率 |
| --- | ---: | ---: |
| フル読込 | 460,400 | — |
| ROI bounds（中心 60% AABB） | 76,919 | ~83% |
| ROI + resolution（spacing×4） | 26 | ~99.99% |

| 項目 | 値 |
| --- | --- |
| COPC spacing | 0.008031 |
| coarse resolution | 0.032125（spacing×4） |

### MVP パイプライン（root bounds + spacing×4 クエリ結果）

| 項目 | 値 |
| --- | --- |
| plane inliers | 54 |
| clusters | 36 |

### 実行方法

```bash
python bench/public_copc/run.py
```

Rust テスト単体:

```bash
cargo test -p spatialrust --features mvp --test mvp_public_copc --release -- --nocapture
```

HTTP Autzen COPC（要ネットワーク）:

```bash
python bench/public_copc/run.py --http
```

### 主な追加ファイル

| パス | 内容 |
| --- | --- |
| `bench/public_copc/run.py` | 再現ハーネス |
| `crates/spatialrust/tests/mvp_public_copc.rs` | 統合テスト |
| `docs/API_STABILITY.md` | Epic 62: core API 安定性方針 |
| `crates/spatialrust-py/examples/pyg_pointnet_demo.py` | Epic 63: PyG end-to-end 例 |

## 未確認/要確認項目

- HTTP Autzen COPC smoke（`--http`、~80 MB）— 本環境では未実行
- Linux/macOS release 計測の再現

## 次アクション

1. Epic 61: GPU RANSAC plane prototype
2. CI に `bench/public_copc/run.py --fetch-only` + 統合テストを追加（任意）

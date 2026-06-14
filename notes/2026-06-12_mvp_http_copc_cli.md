# HTTP COPC MVP CLI bounds + resolution（Epic 50 / 2026-06-12）

## 結論

**MVP CLI** が **HTTP(S) COPC URL** を入力として受け付け、`--bounds` / `--resolution` クエリを **range request** 経由で適用できるようにした。ローカル HTTP サーバー + 50k scan-like COPC で **library query と CLI `input points:` が一致**することを確認。

## 確認済み事実

### Feature

| feature | 内容 |
|---------|------|
| `mvp-http` | `mvp` + `io-copc-http`（ureq range request） |

### CLI 変更

| 項目 | 内容 |
|------|------|
| HTTP 入力 | `http(s)://.../*.copc.laz\|las` を Path 存在チェックから除外 |
| query 構築 | `read_copc_url_info` で root bounds 取得 |
| 読込 | `read_copc_url_with_query` / フル読込は `read_copc_url_with_query(url, None)` |
| build | `cargo build -p spatialrust --features mvp,mvp-http --bin spatialrust-mvp` |

### Integration テスト

| テスト | crate | 内容 |
|--------|-------|------|
| `read_copc_url_with_query_matches_local_file` | spatialrust-io | 2k COPC、HTTP vs ローカル query 一致 |
| `mvp_cli_http_copc_bounds_resolution_matches_local_query` | spatialrust | 50k scan-like、ROI+bounds+resolution、CLI 一致 |
| `detect_input_format_accepts_http_copc_urls` | spatialrust-mvp bin | URL 形式検出 |

| 実行 | |
|------|---|
| IO | `cargo test -p spatialrust-io --features io-copc-http --test copc_http_local` |
| MVP | `cargo test -p spatialrust --features mvp,mvp-http --test mvp_http_copc_cli` |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/src/bin/spatialrust_mvp.rs` | HTTP COPC load path |
| `SpatialRust/crates/spatialrust/Cargo.toml` | `mvp-http` feature |
| `SpatialRust/crates/spatialrust/tests/mvp_http_copc_cli.rs` | CLI HTTP 統合 |
| `SpatialRust/crates/spatialrust-io/tests/copc_http_local.rs` | IO HTTP 統合 |

## 未確認/要確認項目

- 外部公開 COPC URL（本番 CDN / S3 presigned URL）での end-to-end
- 外部実スキャン COPC multiplier 曲線（Epic 49: 合成 50k で平坦曲線確認済み）

## 次アクション

1. Epic 46–51 を `push!` でまとめて commit/push
2. 外部 COPC URL があれば HTTP CLI smoke を再実行

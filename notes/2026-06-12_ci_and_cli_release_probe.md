# CI + approximate Auto CLI release 計測（Epic 51 / 2026-06-12）

## 結論

Epic 46–50 の退行防止のため **CI テスト行列**を拡張し、Epic 48 未確認だった **CLI release 単体計測（1M xyzinormal LAS IO 込み）** を `probe_xyzinormal_approximate_auto_cli_release` で取得。**CLI end-to-end では Auto ~72 ms**（criterion pipeline-only ~40 ms より大きい＝LAS IO + デフォルト MVP 設定のオーバーヘッド）。

## 確認済み事実

### CI 追加（`.github/workflows/ci.yml`）

| job | 内容 |
|-----|------|
| `spatialrust-io-copc-http-local` | `copc_http_local` integration |
| `spatialrust-mvp-http` | `mvp_http_copc_cli` integration |
| `spatialrust-mvp-gpu-integration` | `mvp_pipeline` @ `mvp,pipeline-mvp-gpu`（**26 passed**） |
| bench compile | `mvp_xyzinormal_approximate_auto` compile-only |

### CLI release 計測（1M LAS, leaf=4.0, approximate, 2026-06-12）

| policy | input points | elapsed (release) |
|--------|-------------:|------------------:|
| auto | 1,000,100 | ~72 ms |
| cpu | 1,000,100 | ~61 ms |
| gpu | 1,000,100 | ~193 ms |

- criterion `mvp_xyzinormal_approximate_auto` Auto @1M ~40 ms は **pipeline のみ**（メモリ上点群、カスタム MVP 設定）
- CLI は **LAS 読込/書込 + `MvpPipelineConfig::default()` 派生設定** を含むため数値は直接比較不可
- GPU CLI が遅いのは **毎回 wgpu 初期化 + デフォルト plane/normal 設定** の影響（要別途 profiling）

| 実行 | |
|------|---|
| プローブ | `SPATIALRUST_PROBE_RELEASE=1 cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --release --test mvp_pipeline probe_xyzinormal_approximate_auto_cli_release -- --ignored --nocapture` |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/.github/workflows/ci.yml` | CI 行列 + bench compile |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | release CLI プローブ |

## 未確認/要確認項目

- CLI GPU policy の wgpu 初期化コスト分離（warmup 後再計測）
- GitHub Actions 上での `spatialrust-mvp-gpu-integration` 安定性（software renderer）

## 次アクション

1. Epic 46–51 を `push!` でまとめて commit/push
2. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行

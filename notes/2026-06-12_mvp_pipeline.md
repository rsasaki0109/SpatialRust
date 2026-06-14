# MVP パイプライン検証（2026-06-12）

## 結論

MVP チェーン（PCD/LAS/COPC/HTTP COPC → voxel → normals → RANSAC plane → Euclidean cluster → ラベル付き出力）は integration test で検証済み。**approximate-first xyzinormal + Auto** は 1M 点で GPU voxel path を選択（Epic 46–47）。COPC CLI は `--bounds` / `--resolution` / HTTP URL に対応（Epic 43–50）。

## 確認済み事実

| 項目 | 結果 |
|------|------|
| 実行日 | 2026-06-12 |
| テスト (`mvp`) | `cargo test -p spatialrust --features mvp --test mvp_pipeline` → **23 passed**, 1 ignored |
| テスト (`mvp` + GPU) | **27 passed**, 5 ignored |
| HTTP COPC CLI | `cargo test -p spatialrust --features mvp,mvp-http --test mvp_http_copc_cli` → **1 passed** |
| CLI bin テスト | `cargo test -p spatialrust --features mvp --bin spatialrust-mvp` → **13 passed** |
| パイプライン段 | voxel → normal → plane → cluster → optional ICP |
| CLI voxel | `--voxel-mode centroid\|approximate`、`--voxel-policy auto\|cpu\|gpu` |
| COPC query | `--bounds`、`--resolution`、HTTP(S) URL（`mvp-http`） |
| ベンチ | `--repeat N`（反復計測 + summary、Epic 53） |

### Feature 早見

| feature | 用途 |
|---------|------|
| `mvp` | フル MVP + PCD/LAS/COPC |
| `mvp-http` | HTTP COPC range read |
| `pipeline-mvp-gpu` | GPU voxel + GPU integration tests |

### 実行例

```bash
cargo test -p spatialrust --features mvp --test mvp_pipeline
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline
cargo test -p spatialrust --features mvp,mvp-http --test mvp_http_copc_cli
cargo run -p spatialrust --features mvp,mvp-http --bin spatialrust-mvp -- \
  --bounds 0,0,-0.01,40,20,0.5 --resolution 1.4 \
  https://example.com/scan.copc.laz out.copc.laz
```

### コード入口

| API | パス |
|-----|------|
| `MvpPipeline::run` | `SpatialRust/crates/spatialrust-pipeline/src/mvp.rs` |
| MVP CLI | `SpatialRust/crates/spatialrust/src/bin/spatialrust_mvp.rs` |

## 未確認/要確認項目

- 外部実スキャン COPC / HTTP URL での end-to-end
- CLI 初回 Auto/GPU の wgpu cold init コスト低減（`--repeat` で warm 計測可能、Epic 53）

## 次アクション

1. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行
2. `--repeat` 付き release プローブを approximate Auto 1M LAS で記録

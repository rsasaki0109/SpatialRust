# MVP CLI `--repeat N` ベンチモード（Epic 53 / 2026-06-12）

## 結論

MVP CLI に **`--repeat N`** を追加。入力読込は 1 回、`MvpPipeline` も再利用して pipeline を N 回実行し、**反復毎の elapsed** と **min/max/avg summary** を stderr に出力。Epic 52 の wgpu cold/warm 分離を **同一 CLI プロセス内**で計測可能。

## 確認済み事実

### CLI 仕様

| 項目 | 内容 |
|------|------|
| オプション | `--repeat <N>`（正の整数、default 1） |
| 入力 | 1 回だけ load |
| pipeline | `MvpPipeline` を N 回 `run`（同一インスタンス） |
| 出力 | 最終反復結果のみ write |
| stderr | `repeat i/N elapsed:`、`repeat summary: min/max/avg`、`elapsed:`（最終反復） |

### Integration テスト

| テスト | 内容 |
|--------|------|
| `parse_repeat_*` | bin unit（accept/reject） |
| `mvp_cli_repeat_logs_per_iteration_timing` | 5k COPC、`--repeat 2`、summary 行確認 |

| 実行 | |
|------|---|
| bin | `cargo test -p spatialrust --features mvp --bin spatialrust-mvp parse_repeat` |
| CLI | `cargo test -p spatialrust --features mvp mvp_cli_repeat_logs` |

### 使用例（warmup 込み GPU 計測）

```bash
cargo run -p spatialrust --features mvp,pipeline-mvp-gpu --release --bin spatialrust-mvp -- \
  --leaf-size 4.0 --voxel-mode approximate --voxel-policy gpu --repeat 3 \
  scan.las out.las
```

- `repeat 1/N` ≈ cold（wgpu init 込み）
- `repeat 2/N` 以降 ≈ warm（`WgpuRuntime::shared` 再利用）

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/src/bin/spatialrust_mvp.rs` | `--repeat` 実装 |
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | CLI integration |

### Release 計測（Epic 54 で `--repeat 3` 記録済み）

| policy | repeat 1 | repeat 3 | avg (n=3) |
|--------|----------|----------|-----------|
| cpu | ~37 ms | ~37 ms | ~37 ms |
| auto | ~38 ms | ~35 ms | ~36 ms |
| gpu | ~35 ms | ~35 ms | ~35 ms |

詳細: `SpatialRust/notes/2026-06-12_mvp_cli_repeat_release_probe.md`

## 未確認/要確認項目

- GitHub Actions software renderer での `--repeat` + GPU policy
- Epic 51 単発 CLI gpu ~193 ms との差分要因

## 次アクション

1. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行
2. 必要なら Epic 51 単発 vs `--repeat 1` 同条件比較

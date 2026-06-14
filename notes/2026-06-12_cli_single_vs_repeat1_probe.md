# Epic 51 単発 vs `--repeat 1` 同条件比較（Epic 55 / 2026-06-12）

## 結論

1M xyzinormal LAS + approximate CLI について、**単発実行と `--repeat 1` は同程度（~37–41 ms）**。**Epic 51 の gpu ~193 ms は再現せず**（同条件で ~39 ms）。**計測方法の差ではなく、当時の環境要因（GPU デバイス状態 / 負荷等）** と判断。

## 確認済み事実

### 単発 vs `--repeat 1`（release, separate process each, 2026-06-12）

| policy | single | repeat 1 | delta |
|--------|-------:|---------:|------:|
| cpu | ~37.0 ms | ~41.1 ms | ~+4.0 ms |
| auto | ~38.2 ms | ~39.1 ms | ~+0.9 ms |
| gpu | ~38.2 ms | ~37.8 ms | ~−0.4 ms |

→ **`--repeat 1` は単発と実質同等**（repeat 行は `repeat > 1` のみ出力）

### Epic 51 再現（auto → cpu → gpu 単発、2026-06-12）

| policy | elapsed |
|--------|--------:|
| auto | ~37.0 ms |
| cpu | ~40.1 ms |
| gpu | ~39.5 ms |

Epic 51 記録（同日较早）: auto ~72 ms / cpu ~61 ms / gpu ~193 ms → **再現不可**

### gpu 単発 ×2（別プロセス連続）

| run | elapsed |
|-----|--------:|
| 1 | ~39.4 ms |
| 2 | ~38.1 ms |

→ プロセス間 cold init スパイクは **今回未観測**

### プローブ

```bash
SPATIALRUST_PROBE_RELEASE=1 cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --release \
  --test mvp_pipeline probe_xyzinormal_approximate_auto_cli_single_vs_repeat1_release -- --ignored --nocapture
```

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | 比較プローブ + `run_approximate_cli_probe` helper |

## 未確認/要確認項目

- Epic 51 計測時の GPU エラー（device lost 等）ログの有無
- GitHub Actions software renderer での安定性

## 次アクション

1. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行
2. 古いノートの outdated 次アクション（push! 済み等）を整理

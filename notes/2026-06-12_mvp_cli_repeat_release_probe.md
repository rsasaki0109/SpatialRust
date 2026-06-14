# approximate Auto CLI `--repeat 3` release 計測（Epic 54 / 2026-06-12）

## 結論

1M xyzinormal LAS + **`--repeat 3`** の release CLI プローブを記録。**LAS 読込は 1 回のみ**、反復 elapsed は pipeline のみ。Epic 51 の単発 CLI（gpu ~193 ms）と比べ、**同一プロセス内 repeat では auto/gpu とも ~35 ms 台で安定**（cold/warm 差は小さい）。

## 確認済み事実

### Release 計測（leaf=4.0, approximate, 1M LAS, default MVP 設定, 2026-06-12）

| policy | repeat 1 | repeat 2 | repeat 3 | min | max | avg |
|--------|----------|----------|----------|-----|-----|-----|
| cpu | ~36.8 ms | ~37.2 ms | ~36.9 ms | 36.8 | 37.2 | 37.0 |
| auto | ~38.1 ms | ~35.9 ms | ~35.0 ms | 35.0 | 38.1 | 36.3 |
| gpu | ~35.3 ms | ~35.4 ms | ~35.2 ms | 35.2 | 35.4 | 35.3 |

- **IO は repeat 外**（`loading` + `input points:` の後に反復計測）
- auto/gpu は 1M 点で GPU voxel path（Epic 46 閾値）
- Epic 51 単発 CLI（別プロセス ×3）: auto ~72 ms / cpu ~61 ms / gpu ~193 ms → **プロセス毎の cold init + 計測方法差**を疑う

### プローブ

| テスト | 内容 |
|--------|------|
| `probe_xyzinormal_approximate_auto_cli_repeat_release` | cpu/auto/gpu × `--repeat 3` |

```bash
SPATIALRUST_PROBE_RELEASE=1 cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --release \
  --test mvp_pipeline probe_xyzinormal_approximate_auto_cli_repeat_release -- --ignored --nocapture
```

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | repeat プローブ + stderr パーサ |

## 未確認/要確認項目

- GitHub Actions software renderer での `--repeat` + GPU policy
- Epic 51 単発 gpu ~193 ms → **Epic 55: 再現せず、環境要因と判断**

## 次アクション

1. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行
2. 古いノートの outdated 次アクション整理

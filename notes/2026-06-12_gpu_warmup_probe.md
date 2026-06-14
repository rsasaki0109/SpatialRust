# xyzinormal approximate Auto GPU warmup プローブ（Epic 52 / 2026-06-12）

## 結論

Epic 51 で CLI `--voxel-policy gpu` が ~193 ms と遅かった原因の一つは **プロセス毎の wgpu 初期化**（CLI は 1 実行 = 1 プロセス）。同一プロセス内では `WgpuRuntime::shared()` により **2 回目以降の GPU pipeline は大幅に短縮**されることを in-process プローブで確認。

## 確認済み事実

### In-process プローブ（release, 1M, custom MVP config, LAS IO なし）

| 実行 | elapsed |
|------|--------:|
| gpu cold | ~255 ms（`WgpuRuntime::shared` 初回 init 込み） |
| gpu warm | ~38 ms |
| cpu | ~47 ms |
| auto | ~41 ms（shared runtime 温まった後の Auto=GPU path） |

- gpu cold→warm で **~85% 短縮**（init が支配的）
- CLI `--voxel-policy gpu` ~193 ms は **プロセス毎の cold init + LAS IO + デフォルト MVP 設定** と整合
- auto を gpu の後に計測しているため、**初回 Auto の cold cost** は gpu cold と同等（別プロセス CLI では毎回発生しうる）

| 項目 | 内容 |
|------|------|
| テスト | `probe_xyzinormal_approximate_auto_gpu_warmup` |
| 実行 | `SPATIALRUST_PROBE_RELEASE=1 cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --release --test mvp_pipeline probe_xyzinormal_approximate_auto_gpu_warmup -- --ignored --nocapture` |

### CLI vs in-process の整理

| 経路 | wgpu init | 備考 |
|------|-----------|------|
| CLI `--voxel-policy gpu` | 毎回 | LAS IO + デフォルト MVP 設定も加算 |
| in-process 2 回目 GPU | なし | `WgpuRuntime::shared()` 再利用 |
| criterion bench | 1 回 / プロセス | pipeline オブジェクト再利用 |

### 主な変更ファイル

| パス | 変更 |
|------|------|
| `SpatialRust/crates/spatialrust/tests/mvp_pipeline.rs` | in-process warmup プローブ |
| `SpatialRust/notes/2026-06-12_mvp_pipeline.md` | テスト件数・feature 表更新 |

## 未確認/要確認項目

- CLI 内で shared runtime を活かす長寿命 daemon / `--repeat` ベンチモード
- GitHub Actions software renderer での gpu-integration 安定性

## 次アクション

1. 外部 COPC URL / 実スキャンファイルがあれば HTTP CLI / multiplier 曲線を再実行
2. 必要なら CLI `--repeat N` で warmup 込みベンチを追加

# GPU Voxel パイプライン ベンチマーク（2026-06-12）

## 結論

end-to-end filter ベンチ（leaf=4.0, `without_gpu_min_points()`）では **centroid モードで GPU が CPU を上回るのは 500k 点付近から**（~29 ms vs ~36 ms）。200k 点では CPU がわずかに速い。**100k 点以下では CPU が 2–5 倍速い**。MVP デフォルトが centroid のため、`DEFAULT_GPU_MIN_POINTS` を **500_000** に設定し、それ未満は CPU にフォールバックする。

## 確認済み事実

| 点数 | cpu_centroid | gpu_centroid | cpu_approx | gpu_approx |
|------|-------------|-------------|------------|------------|
| 10k | **~0.8 ms** | ~17 ms | **~0.5 ms** | ~17 ms |
| 65k | **~4.7 ms** | ~14.7 ms | **~2.2 ms** | ~14.0 ms |
| 100k | **~7.0 ms** | ~17.2 ms | **~3.5 ms** | ~18.2 ms |
| 200k | **~23.8 ms** | ~26.3 ms | **~7.7 ms** | ~22.7 ms |
| 500k | **~36.0 ms** | **~29.4 ms** | **~18.6 ms** | ~25.6 ms |

| 項目 | 結果 |
|------|------|
| ベンチコマンド | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample` |
| GPU kernel only（100k, centroid pipeline） | **~24 ms**（filter 全体 ~17 ms と別計測） |
| Epic 24 | adapter limits + 4ch multi gather |
| Epic 25 | `gpu_min_points` + `Auto` policy + MVP デフォルト `Auto` |
| Epic 26 | 100k/200k/500k クロスオーバー評価 → デフォルト閾値 **500_000** |
| テスト | `spatialrust-gpu` 16 / `spatialrust-filtering` 9 |

## 未確認/要確認項目

- 300k–400k 点での centroid クロスオーバー精密値
- 属性多数（RGB+intensity+normals）時の GPU メリット
- ping-pong readback 削減後の filter レイテンシ

## 次アクション

1. GPU filter の staging/readback 最適化
2. MVP example binary
3. approximate_first モード向け別閾値の検討（500k でも CPU 優位）

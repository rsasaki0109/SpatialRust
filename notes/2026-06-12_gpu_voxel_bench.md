# GPU Voxel パイプライン ベンチマーク（2026-06-12）

## 結論

end-to-end filter ベンチ（leaf=4.0, `without_gpu_min_points()`）再計測（Epic 30）では **centroid は 500k 点以上で GPU 優位**（500k: ~51 ms vs ~94 ms、1M: ~56 ms vs ~155 ms）。**approximate-first は 1M 点まで CPU 優位**（1M: ~45 ms vs ~50 ms）のため `DEFAULT_GPU_MIN_POINTS_APPROXIMATE = 2_000_000` に引き上げ。**Epic 32（2M 点）で approximate-first も GPU 優位を確認**（~70 ms vs ~81 ms）し、閾値 2M を妥当と判断。

## 確認済み事実

| 点数 | cpu_centroid | gpu_centroid | cpu_approx | gpu_approx |
|------|-------------|-------------|------------|------------|
| 10k | **~0.8 ms** | ~17 ms | **~0.5 ms** | ~17 ms |
| 65k | **~4.7 ms** | ~14.7 ms | **~2.2 ms** | ~14.0 ms |
| 100k | **~7.0 ms** | ~17.2 ms | **~3.5 ms** | ~18.2 ms |
| 200k | **~23.8 ms** | ~26.3 ms | **~7.7 ms** | ~22.7 ms |
| 500k | **~94 ms** | **~51 ms** | **~21 ms** | ~31 ms |
| 750k | **~148 ms** | **~48 ms** | **~41 ms** | ~53 ms |
| 1M | **~155 ms** | **~56 ms** | **~45 ms** | ~50 ms |
| 2M | **~389 ms** | **~101 ms** | **~81 ms** | **~70 ms** |

| 項目 | 結果 |
|------|------|
| ベンチコマンド | `cargo bench -p spatialrust-filtering --features filter-voxel-gpu --bench voxel_downsample` |
| Epic 30 再計測日 | 2026-06-12（500k–1M centroid + approximate） |
| Epic 32 再計測日 | 2026-06-12（2M approximate-first GPU クロスオーバー） |
| GPU kernel only（100k, centroid pipeline） | **~24 ms**（filter 全体 ~17 ms と別計測） |
| Epic 24 | adapter limits + 4ch multi gather |
| Epic 25 | `gpu_min_points` + `Auto` policy + MVP デフォルト `Auto` |
| Epic 26 | 100k/200k/500k クロスオーバー評価 → centroid 閾値 **500_000** |
| Epic 28 | approximate-first 初回閾値 750_000（Epic 30 で 1M まで CPU 優位を確認） |
| Epic 30 | 1M centroid GPU ~2.8×、approximate 閾値 **2_000_000** へ引き上げ |
| Epic 32 | 2M approximate-first GPU ~1.16× → 閾値 **2_000_000** を維持 |
| テスト | `spatialrust-gpu` 17 / `spatialrust-filtering` 10 |

## 未確認/要確認項目

- 300k–400k 点での centroid クロスオーバー精密値
- 1M–2M 点での approximate-first クロスオーバー精密値（Epic 32: 2M で GPU 優位を確認、1M では CPU 優位）
- ping-pong readback 削減後の filter レイテンシ（Epic 33/35 で xyzrgb・xyzinormal を改善済み）

## 次アクション

1. 外部実スキャン COPC CLI `--resolution` ベンチ
2. approximate-first xyzinormal GPU kernel/readback 最適化

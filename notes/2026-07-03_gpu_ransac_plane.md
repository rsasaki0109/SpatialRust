# GPU RANSAC plane prototype（Epic 61 / 2026-07-03）

## 結論

RANSAC plane の **仮説スコアリング（inlier カウント）** を wgpu + **WGSL** で並列化した。CPU 版と同じ RNG で仮説を生成し、GPU で全 iteration を一括評価 → CPU で refine / マスク抽出。

## アーキテクチャ

| 層 | 言語 | 役割 |
| --- | --- | --- |
| `spatialrust-gpu` | Rust + **WGSL** | `score_ransac_plane_hypotheses_gpu` — 1 thread / hypothesis |
| `spatialrust-segmentation` | Rust | `GpuRansacPlaneSegmenter` — オーケストレーション |
| 共有 | Rust | `plane_ransac.rs` — RNG / refine / inlier 収集 |

## Feature

```toml
segment-ransac-plane-gpu = ["segment-ransac-plane", "dep:spatialrust-gpu"]
```

Meta crate:

```toml
segment-ransac-plane-gpu = [
    "spatialrust-segmentation/segment-ransac-plane-gpu",
    "gpu-wgpu",
]
```

## 使い方

```rust
use spatialrust::{GpuRansacPlaneSegmenter, RansacPlaneConfig};

let result = GpuRansacPlaneSegmenter::new(RansacPlaneConfig::default())
    .segment(&cloud)?;
```

## テスト

```bash
cargo test -p spatialrust-gpu --features gpu-wgpu scores_planar --release
cargo test -p spatialrust-segmentation --features segment-ransac-plane,segment-ransac-plane-gpu gpu_matches --release
```

`gpu_matches_cpu_on_planar_patch`: 100 点平面 + 2 outlier で CPU/GPU の inlier 数一致を確認。

## 主な追加ファイル

| パス | 内容 |
| --- | --- |
| `spatialrust-gpu/src/kernels/ransac_plane.rs` | WGSL compute + dispatch |
| `spatialrust-segmentation/src/plane_ransac.rs` | 共有 RANSAC ヘルパ |
| `spatialrust-segmentation/src/plane_gpu.rs` | `GpuRansacPlaneSegmenter` |

## 次アクション

1. 公開 PCD（460k 点）での CPU vs GPU ベンチ
2. MVP パイプラインへの `ExecutionPolicy` 統合（Auto 閾値）
3. Epic 64: upload pool / buffer pool の public API 化

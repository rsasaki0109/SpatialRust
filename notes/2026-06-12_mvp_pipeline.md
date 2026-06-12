# MVP パイプライン検証（2026-06-12）

## 結論

MVP チェーン（PCD/LAS → voxel → normals → RANSAC plane → Euclidean cluster → 任意 ICP → ラベル付き PCD/LAS 出力）は **`feature = "mvp"`** で integration test 10 件すべて通過。centroid voxel の GPU 優位は **500k 点以上**で確認されたため **`gpu_min_points` デフォルトを 500_000** に設定した。**Epic 34:** CLI `--voxel-mode approximate` を追加（デフォルトは `centroid`）。

## 確認済み事実

| 項目 | 結果 |
|------|------|
| 実行日 | 2026-06-12 |
| テスト | `cargo test -p spatialrust --features mvp --test mvp_pipeline` → **13 passed** |
| CLI テスト | `cargo test -p spatialrust --features mvp --bin spatialrust-mvp` → **10 passed** |
| パイプライン段 | voxel downsample → normal estimate → plane segment → cluster → optional ICP |
| 出力 | `label` フィールド付き `PointCloud`（plane inlier=0, cluster id≥1） |
| IO 検証 | PCD/LAS/COPC roundtrip、COPC query/resolution、**xyzinormal PCD → MVP（Epic 37）** |
| CLI voxel | **`--voxel-mode centroid\|approximate`**（Epic 34）、`--voxel-policy auto\|cpu\|gpu` |
| 修正 | `extract_indices([])` が空 buffers を返していた問題を修正（平面のみ点群で plane segmentation 後にクラッシュ） |
| デフォルト policy | `MvpPipelineConfig.voxel_policy = Auto`（点数 ≥ `gpu_min_points` で GPU） |

### 実行例

```bash
# MVP integration tests
cargo test -p spatialrust --features mvp

# COPC bounds query → MVP
cargo test -p spatialrust --features mvp mvp_copc_query_pipeline

# CLI（PCD/LAS/COPC 等 → ラベル付き出力）
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- input.copc.laz output.copc.laz
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- --leaf-size 0.2 --voxel-policy auto scan.las out.las
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --voxel-mode approximate --leaf-size 0.2 scan.las out.las
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --bounds 0,0,-0.01,0.85,0.85,0.01 scan.copc.laz roi.copc.laz
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --resolution 0.5 scan.copc.laz coarse.copc.laz

# GPU voxel 込み pipeline crate
cargo test -p spatialrust-pipeline --features pipeline-mvp,pipeline-mvp-gpu

# 段階 smoke（feature 個別）
cargo test -p spatialrust --features io-pcd,filter-voxel mvp_load_voxel_downsample
cargo test -p spatialrust --features mvp mvp_load_voxel_normals_plane_cluster
```

### コード入口

| API | パス |
|-----|------|
| `MvpPipeline::run` | `SpatialRust/crates/spatialrust-pipeline/src/mvp.rs` |
| MVP CLI | `SpatialRust/crates/spatialrust/src/bin/spatialrust_mvp.rs` |
| 公開 re-export | `spatialrust` crate（`feature = "mvp"`） |

## 未確認/要確認項目

- 実スキャン規模（数百万点）での end-to-end 時間
- 実スキャン由来 COPC での `--resolution` 効果（Epic 36: 合成 7k 多階層 fixture で確認済み）

## 次アクション

1. approximate-first × 属性多数ベンチ
2. 実スキャン規模（数百万点）end-to-end

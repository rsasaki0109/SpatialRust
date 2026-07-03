# Epic 73 — Parallel staging in normal and outlier hot paths (2026-07-03)

## Goal

Unify CPU parallel chunk boundaries across feature estimation and filtering with the
same staging policy as [`SpatialTensor`] / Epic 70–71.

## Changes

| Area | Before | After |
| --- | --- | --- |
| `spatialrust-search::staging` | — | `parallel_worker_count`, `parallel_index_ranges`, `parallel_index_for_each` |
| `NormalEstimator` | local `normal_worker_count` + ad-hoc ranges | shared staging helpers |
| SOR / ROR | local `outlier_worker_count` + `chunks_mut` | `parallel_index_for_each` |

Threshold: `PARALLEL_STAGING_MIN_POINTS` (4096), chunk size: `DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE` (16_384).

## Tests

```bash
cargo test -p spatialrust-search staging
cargo test -p spatialrust-features
cargo test -p spatialrust-filtering --features filter-outlier
```

## Next steps

- GPU upload helper for `AoSoAXyzChunk`
- Optional `search-parallel` fast path in radius normal mode via `radius_search_spatial_tensor_parallel`
- DLPack export

[`SpatialTensor`]: ../../crates/spatialrust-core/src/tensor.rs

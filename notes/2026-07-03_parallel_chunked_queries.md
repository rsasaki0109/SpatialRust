# Epic 71 — Parallel chunked spatial queries (2026-07-03)

## Goal

Dispatch [`ChunkedRadiusSearchIndex`](../../crates/spatialrust-search/src/chunked.rs) work
across [`SpatialTensor`] chunks using `std::thread` (no new deps), building on Epic 70.

## API (`spatialrust-search/parallel`)

| Symbol | Role |
| --- | --- |
| `PARALLEL_CHUNK_QUERY_MIN_POINTS` | 4_096 — below this, sequential path |
| `radius_search_spatial_tensor_parallel` | One thread per tensor chunk |
| `radius_search_spatial_tensor_parallel_into` | Append merged results |
| `nearest_k_spatial_tensor_parallel` | k-NN variant |
| `nearest_k_spatial_tensor_parallel_into` | k-NN append variant |

Each thread owns a local `Vec`; results merge without locks because index search is read-only.

Meta crate re-exports when `search-kdtree` + `parallel` features are enabled.

## Tests

```bash
cargo test -p spatialrust-search --features parallel
```

## Next steps

- AoSoA chunk packing (interleaved xyz within chunk, feature flag)
- Wire parallel chunk queries into normal estimation / outlier removal hot paths
- DLPack export of chunk views

[`SpatialTensor`]: ../../crates/spatialrust-core/src/tensor.rs

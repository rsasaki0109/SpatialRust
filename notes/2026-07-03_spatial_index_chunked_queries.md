# Epic 70 — SpatialIndex chunked neighbor queries (2026-07-03)

## Goal

Extend the architecture model (`SpatialTensor` + `SpatialIndex`) with chunk-aligned
neighbor queries so parallel CPU/GPU algorithms can stage work the same way as
[`SpatialTensor`] iteration.

## API (provisional)

| Symbol | Role |
| --- | --- |
| `SpatialIndex::preferred_chunk_size` | Defaults to `DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE` (16_384) |
| `ChunkQueryRange` | `Range<usize>` alias matching `SpatialTensorChunk::range` |
| `ChunkedRadiusSearchIndex` | `radius_search_chunk_into`, `radius_search_at` |
| `ChunkedNearestNeighborIndex` | `nearest_k_chunk_into` |
| `radius_search_spatial_tensor` | Iterate tensor chunks → tagged `(query_index, Neighbor)` |
| `nearest_k_spatial_tensor` | Same for k-NN |

Implementations: `KdTree`, `BruteForceIndex`.

`KdTree::from_point_cloud` already builds an index from [`PointCloud`].

## Design constraints

- Chunk APIs reuse existing single-query backends (no new index structure)
- Output pairs preserve query index for parallel reduction / GPU batching
- Stability: **Provisional** (search crate)

## Next steps

- AoSoA chunk packing (interleaved xyz within chunk)
- Wire parallel chunk queries into normal/outlier pipelines
- DLPack export of chunk views for AI integration

[`SpatialTensor`]: ../../crates/spatialrust-core/src/tensor.rs
[`PointCloud`]: ../../crates/spatialrust-core/src/pointcloud.rs

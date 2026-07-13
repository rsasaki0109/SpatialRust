# AoSoA attribute layouts (Epic 78)

## Outcome

`AoSoAAttributeLayout` explicitly describes record stride and position,
intensity, and normal offsets. Three capability-based layouts are supported:

- `[x, y, z, intensity]`
- `[x, y, z, nx, ny, nz]`
- `[x, y, z, intensity, nx, ny, nz]`

`SpatialTensorChunk` packs each form without changing the schema-SoA source.
Missing `HasIntensity` or `HasNormals3` capabilities return an error.

`GpuAoSoAttributeChunk` retains the same layout metadata through upload and
supports explicit recycling. `reduce_voxel_attributes_aoso_chunks` consumes
global `GpuVoxelSegments` and reduces complete interleaved records with either
`Average` or `First`. Attribute chunks are joined by GPU-to-GPU copies, so
voxel aggregation remains correct across chunk boundaries.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-core/src/tensor_aoso.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_reduce_attributes_aoso.wgsl`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/pipeline_cache.rs`

## Next steps

- Normalize averaged normal vectors as an optional semantic post-pass.
- Benchmark attribute layouts and chunk sizes on scan-scale inputs.

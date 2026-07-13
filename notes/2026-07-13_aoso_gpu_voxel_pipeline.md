# AoSoA GPU voxel pipeline (Epic 76)

## Outcome

`downsample_voxel_centroid_aoso_chunks` now connects uploaded interleaved XYZ
chunks to the existing GPU sort and segment stages and a new interleaved
centroid reduction kernel.

The chunks are copied into one combined storage buffer with GPU-to-GPU copies.
A global key and sort pass therefore merges voxels that span input chunk
boundaries. Position data is not read back or uploaded again; only final
centroids return to the CPU. Global `GpuVoxelSegments` remain available for
follow-up GPU operations.

## Correctness

The test deliberately places two points from the same voxel on opposite sides
of a chunk boundary. It verifies unique global keys and the centroid of every
occupied voxel.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_reduce_aoso.wgsl`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/pipeline_cache.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/upload_cache.rs`

## Next steps

- Add interleaved normal and intensity packers.
- Benchmark the AoSoA pipeline across chunk sizes on million-point clouds.

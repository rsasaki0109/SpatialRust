# AoSoA GPU voxel dispatch (Epic 75)

## Outcome

`compute_voxel_keys_aoso_chunks` dispatches a dedicated WGSL kernel directly
against `GpuAoSoXyzChunk` buffers. The API preserves chunk boundaries in its
result and applies one shared origin and inverse leaf size, so voxel keys remain
globally comparable.

The implementation avoids converting the interleaved upload back into three
coordinate columns or performing a hidden position upload. Only the generated
keys are copied back to the CPU.

## Correctness

The GPU crate test covers multiple uneven chunks, negative coordinates, and a
final partial chunk. Flattened GPU results are compared with a global CPU
reference to verify that chunk boundaries do not change key assignment.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_keys_aoso.wgsl`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/pipeline_cache.rs`

## Next steps

- Keep key and segment buffers on the GPU across chunks.
- Add an end-to-end streaming voxel reduction over chunk batches.
- Add halo-aware chunk dispatch before using AoSoA chunks for neighborhood
  algorithms such as normal estimation.

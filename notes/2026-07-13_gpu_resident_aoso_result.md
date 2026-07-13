# GPU-resident AoSoA result (Epic 77)

## Outcome

`AoSoAVoxelCentroidResult` retains its combined interleaved source positions in
a `GpuAoSoXyzBuffer`. Downstream kernels can use `buffer()` and `point_count()`
without a CPU readback or position re-upload.

The wrapper owns the wgpu buffer and exposes `recycle()` for an explicit return
to the runtime buffer pool. `AoSoAVoxelCentroidResult::recycle()` consumes the
whole result, recycles positions, and drops segment and CPU output metadata.

Empty inputs return a valid zero-point wrapper backed by a minimal recyclable
storage buffer, keeping the lifecycle contract uniform.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/lib.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/upload_cache.rs`

## Next steps

- Consume `GpuAoSoXyzBuffer` from a downstream normal kernel.
- Benchmark chained execution and buffer reuse across repeated frames.

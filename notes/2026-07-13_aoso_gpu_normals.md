# AoSoA GPU normal chaining (Epic 79)

## Outcome

`estimate_normals_aoso_gpu` consumes a retained `GpuAoSoXyzBuffer` directly.
Position data is not copied back to the CPU or uploaded as three SoA columns.
The caller supplies one global neighbor-index array, so neighborhoods can cross
all original chunk boundaries.

The returned `GpuAoSoNormals` owns `(nx, ny, nz, curvature)` records in GPU
storage. `readback()` is optional and `recycle()` explicitly returns storage to
the runtime pool.

The implementation shares the established covariance and Jacobi eigensolver
shader logic with the existing SoA path. A parity test compares both paths on a
planar patch.

## Current boundary

The direct path currently uploads global neighbor indices. Radius-mode grid
construction still requires CPU coordinates in the existing implementation.
A future GPU grid builder is needed before the full radius neighborhood stage
can operate without CPU-side position access.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/normals.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mod.rs`

## Next steps

- Consume the GPU sparse radius grid directly from the normal gather kernel.
- Cache the AoSoA normal pipeline instead of deriving it per call.
- Benchmark chained voxel-to-normal execution including neighbor preparation.

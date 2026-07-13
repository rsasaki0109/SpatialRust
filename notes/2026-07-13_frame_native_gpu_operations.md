# Frame-native GPU operations (Epic 82)

## Outcome

`GpuSpatialFrame` now continues execution without exposing intermediate buffer
ownership:

- `rebuild_radius_grid` replaces sparse radius metadata.
- `estimate_normals` reuses a matching grid or rebuilds it, then safely recycles
  the previous normal buffer.
- `reduce_attributes` applies `Average` or `First` against retained voxel
  segments and records the explicit output readback.

Positions, normals, and attribute wrappers carry their originating device
identity. Frame construction and attachment reject buffers from another wgpu
device before dispatch. Point-count checks remain mandatory for every attached
capability.

Receipt stages are appended only for work performed: a matching radius reuses
the grid, while a changed radius records both grid and normal stages.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/gpu_frame.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging.rs`

## Next steps

- Add frame-native GPU plane scoring and Euclidean clustering.
- Normalize averaged normal attributes as an optional semantic stage.
- Instrument exact internal dispatch counts in `WgpuRuntime`.

# AoSoA GPU module cleanup

## Scope

The 1,521-line AoSoA staging implementation was split by responsibility while
preserving the existing public facade:

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/types.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/upload.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/attributes.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/normals.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/voxel.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/tests.rs`

The facade is now 90 lines and the largest responsibility module is 360 lines.
No public symbol names or feature gates changed.

## CI fix

The main CI run for commit `72ce819` exposed internal normal-shader re-exports
under every `gpu-wgpu` build. Narrow feature jobs treated those unused imports
as errors through `-D warnings`. The re-exports are now enabled only by
`gpu-aoso-staging`, their sole consumer.

## Performance equivalence

The same 10,000-point Criterion command was run before and after the split.

- Before: 23.9–25.3 ms
- After: 23.4–26.9 ms
- Criterion comparison: no performance change detected (`p = 0.51`)

The measurements are local stabilization evidence, not cross-machine claims.

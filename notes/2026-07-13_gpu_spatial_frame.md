# GPU-resident spatial frame (Epic 81)

## Outcome

`GpuSpatialFrame` owns a point schema and GPU-resident positions, optional
normal/curvature output, attribute chunks, voxel segments, and a sparse radius
grid. Capability checks let downstream algorithms reject missing state before
dispatch.

Every frame records the originating wgpu device identity. Public operations
validate the supplied runtime before using buffers, preventing accidental
cross-device dispatch. Attached buffers are also checked against the source
point count.

`readback_positions` is explicit and records device-to-host bytes.
`recycle` consumes the frame and returns pooled positions, normals, and
attributes together.

`run_aoso_voxel_normal_frame` provides the first chained execution path:

`upload → voxel segments → sparse radius grid → radius normals`

The receipt records host upload bytes, GPU-to-GPU position copies, explicit
readback bytes, and logical pipeline stages. Logical stages are reported rather
than pretending to count internal sort/scan dispatches.

## Files

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/gpu_frame.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/src/lib.rs`

## Next steps

- Record exact internal dispatch counts through runtime command instrumentation.
- Add GPU segmentation stages that consume `GpuSpatialFrame` directly.

## Stabilization benchmark

`/home/sasaki/workspace/SpatialRust/crates/spatialrust-gpu/benches/aoso_frame_pipeline.rs`
measures the complete upload → voxel → sparse radius grid → normals path and
recycles the frame each iteration.

A short local release run on 10,000 synthetic points completed at
23.9–25.3 ms (approximately 395–418 Kpoints/s). This is a stabilization
baseline, not a cross-machine performance claim; future optimization runs must
use the same input, feature set, and Criterion parameters.

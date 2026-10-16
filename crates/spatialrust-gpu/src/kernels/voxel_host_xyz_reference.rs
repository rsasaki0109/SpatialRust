// Complete host XYZ parent frozen from 6ce2bdddb51bdb3124cc7d36b50bc0a81f28d42f. Scalar upload/dispatch/readback delegates are source-identical.
use super::*;
pub(super) fn baseline_reduce_voxel_centroids_xyz(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::BufferLengthMismatch { expected: x.len(), found: y.len() });
    }

    let out_x = reduce_voxel_average_f32(runtime, x, segments)?;
    let out_y = reduce_voxel_average_f32(runtime, y, segments)?;
    let out_z = reduce_voxel_average_f32(runtime, z, segments)?;
    Ok((out_x, out_y, out_z))
}

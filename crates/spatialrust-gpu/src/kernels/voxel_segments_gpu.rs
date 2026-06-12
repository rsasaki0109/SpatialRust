use spatialrust_core::SpatialResult;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_keys::compute_voxel_keys_gpu_buffers;
use crate::kernels::voxel_segments::VoxelSegments;
use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys_buffer;
use crate::runtime::WgpuRuntime;

/// Computes voxel keys on the GPU and builds sorted segments using GPU sorting.
pub fn build_voxel_segments_from_positions_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<VoxelSegments> {
    let gpu_segments =
        build_voxel_segments_from_positions_gpu_buffers(runtime, x, y, z, origin, inv_leaf)?;
    gpu_segments.to_voxel_segments(runtime)
}

/// Builds GPU-resident voxel segments from positions without intermediate readbacks.
pub fn build_voxel_segments_from_positions_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<GpuVoxelSegments> {
    let positions = compute_voxel_keys_gpu_buffers(runtime, x, y, z, origin, inv_leaf)?;
    let point_count = positions.point_count();
    let padded_count = point_count.next_power_of_two();
    build_voxel_segments_gpu_from_keys_buffer(
        runtime,
        positions.keys_buffer(),
        point_count,
        padded_count,
    )
}

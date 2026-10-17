// Complete immediate-parent CPU-key entry points; only names change.
use super::super::{
    build_voxel_segments_gpu_from_keys_buffer, empty_gpu_segments, validate_sort_size,
};
use crate::kernels::voxel_keys::VoxelKeyOutput;
use crate::{GpuVoxelSegments, VoxelSegments, WgpuRuntime};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

pub(super) fn previous(
    runtime: &WgpuRuntime,
    keys: &[(i64, i64, i64)],
) -> SpatialResult<VoxelSegments> {
    let gpu_segments = previous_from_keys(runtime, keys)?;
    gpu_segments.to_voxel_segments(runtime)
}

pub(super) fn previous_from_keys(
    runtime: &WgpuRuntime,
    keys: &[(i64, i64, i64)],
) -> SpatialResult<GpuVoxelSegments> {
    if keys.is_empty() {
        return empty_gpu_segments(runtime);
    }
    let point_count = u32::try_from(keys.len()).map_err(|_| {
        SpatialError::InvalidArgument("voxel keys exceed the GPU point limit".to_owned())
    })?;
    validate_sort_size(runtime, point_count)?;
    let key_outputs: Vec<_> = keys
        .iter()
        .map(|&(ix, iy, iz)| VoxelKeyOutput {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            _pad: 0,
        })
        .collect();
    let keys_buffer = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-sort-keys-input"),
        contents: bytemuck::cast_slice(&key_outputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    build_voxel_segments_gpu_from_keys_buffer(runtime, &keys_buffer, point_count, point_count)
}

pub(super) fn prepare(keys: &[(i64, i64, i64)]) -> Vec<VoxelKeyOutput> {
    keys.iter()
        .map(|&(ix, iy, iz)| VoxelKeyOutput {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            _pad: 0,
        })
        .collect()
}

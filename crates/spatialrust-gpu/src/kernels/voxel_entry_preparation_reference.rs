// Complete immediate-parent CPU-key APIs and preparation; only names/paths change.
use crate::kernels::voxel_radix::sort_entries_radix_gpu;
use crate::kernels::voxel_sort::{empty_gpu_segments, validate_sort_size, VoxelSortEntry};
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
    let entries = serial(keys);
    let entries_buffer = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-sort-entries-input"),
        contents: bytemuck::cast_slice(&entries),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let sorted = sort_entries_radix_gpu(runtime, entries_buffer, point_count, None);
    crate::kernels::voxel_compact::compact_voxel_segments_gpu_buffers_with_profiler(
        runtime,
        &sorted,
        point_count,
        None,
    )
}

pub(super) fn serial(keys: &[(i64, i64, i64)]) -> Vec<VoxelSortEntry> {
    keys.iter()
        .enumerate()
        .map(|(index, &(ix, iy, iz))| VoxelSortEntry {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            point_index: index as u32,
        })
        .collect()
}

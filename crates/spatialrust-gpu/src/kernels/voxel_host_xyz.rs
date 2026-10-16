use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{VoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostXyzPlan {
    pub channel_bytes: u64,
    pub staging_bytes: u64,
}

pub(crate) fn plan_host_xyz(
    limits: &wgpu::Limits,
    cells: usize,
    points: usize,
    host_limit: u64,
) -> Option<HostXyzPlan> {
    let cells = u32::try_from(cells).ok()?;
    let points = u32::try_from(points).ok()?;
    if cells == 0
        || points == 0
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_bindings_per_bind_group < 5
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 16
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 256
        || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let channel_bytes = u64::from(cells) * 4;
    let index_bytes = u64::from(points) * 4;
    let staging_bytes = channel_bytes * 3;
    if channel_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || index_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || index_bytes > limits.max_buffer_size
        || staging_bytes > limits.max_buffer_size
        || staging_bytes > host_limit
    {
        return None;
    }
    Some(HostXyzPlan { channel_bytes, staging_bytes })
}

pub(crate) fn try_reduce_host_xyz(
    runtime: &WgpuRuntime,
    xyz: [&[f32]; 3],
    segments: &VoxelSegments,
) -> Option<SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if segments.is_empty() {
        return Some(Ok((Vec::new(), Vec::new(), Vec::new())));
    }
    // Preserve the public wrapper's original empty-value error through its fallback.
    if xyz[0].is_empty() || segments.cell_starts.is_empty() {
        return None;
    }
    let limits = runtime.device().limits();
    for elements in [xyz[0].len(), segments.cell_starts.len()] {
        let bytes = (elements as u64).checked_mul(4)?;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return None;
        }
    }
    let plan =
        plan_host_xyz(&limits, segments.len(), segments.point_indices.len(), isize::MAX as u64)?;
    Some(dispatch_host_xyz(runtime, xyz, segments, plan))
}

fn dispatch_host_xyz(
    runtime: &WgpuRuntime,
    xyz: [&[f32]; 3],
    segments: &VoxelSegments,
    plan: HostXyzPlan,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let device = runtime.device();
    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-host-xyz-indices"),
        contents: bytemuck::cast_slice(&segments.point_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let starts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-host-xyz-starts"),
        contents: bytemuck::cast_slice(&segments.cell_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-host-xyz-staging"),
        size: plan.staging_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-host-xyz-encoder"),
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-host-xyz-output"),
        size: plan.channel_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for (axis, values) in xyz.into_iter().enumerate() {
        let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-host-xyz-values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        super::voxel_reduce::record_voxel_reduce_f32_pass(
            &mut encoder,
            runtime,
            &values,
            &indices,
            &starts,
            segments.len() as u32,
            segments.point_indices.len() as u32,
            &output,
        )?;
        // Preserve each axis before the next scalar pass overwrites this scratch buffer.
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            &staging,
            axis as u64 * plan.channel_bytes,
            plan.channel_bytes,
        );
    }
    runtime.queue().submit(Some(encoder.finish()));
    crate::readback::read_staging_xyz(device, &staging, segments.len())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_host_xyz_tests.rs"]
mod tests;

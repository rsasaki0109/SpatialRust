use bytemuck::{Pod, Zeroable};
use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum XyzMode {
    First,
    Centroid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XyzPlan {
    pub bytes: u64,
}

pub(crate) fn plan_packed_xyz(
    limits: &wgpu::Limits,
    cells: u32,
    host_limit: u64,
) -> Option<XyzPlan> {
    if cells == 0
        // Keep the original XYZ capability envelope, including its separate outputs.
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_bindings_per_bind_group < 9
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
    let words = u64::from(cells).checked_mul(3)?;
    let bytes = words.checked_mul(4)?;
    if words > u64::from(u32::MAX)
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
        || bytes > limits.max_buffer_size
        || bytes > host_limit
    {
        return None;
    }
    Some(XyzPlan { bytes })
}

pub(crate) fn plan_packed_xyz_to_staging(
    limits: &wgpu::Limits,
    cells: u32,
    offset: u64,
    target_bytes: u64,
    host_limit: u64,
) -> Option<XyzPlan> {
    let plan = plan_packed_xyz(limits, cells, host_limit)?;
    if offset % 4 != 0
        || offset.checked_add(plan.bytes)? > target_bytes
        || target_bytes > limits.max_buffer_size
        || target_bytes > host_limit
    {
        return None;
    }
    Some(plan)
}

pub(crate) fn try_record_packed_xyz_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    mode: XyzMode,
) -> bool {
    let Some(plan) = plan_packed_xyz_to_staging(
        &runtime.device().limits(),
        segments.cell_count(),
        offset,
        staging.size(),
        isize::MAX as u64,
    ) else {
        return false;
    };
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-combined-xyz-packed-output"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    record_packed_xyz(encoder, runtime, xyz, segments, mode, &output);
    encoder.copy_buffer_to_buffer(&output, 0, staging, offset, plan.bytes);
    true
}

pub(crate) fn try_voxel_xyz_packed(
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    mode: XyzMode,
) -> Option<SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    if segments.cell_count() == 0 {
        return None;
    }
    let plan =
        plan_packed_xyz(&runtime.device().limits(), segments.cell_count(), isize::MAX as u64)?;
    Some(dispatch_packed_xyz(runtime, xyz, segments, mode, plan))
}

fn dispatch_packed_xyz(
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    mode: XyzMode,
    plan: XyzPlan,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let device = runtime.device();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-xyz-packed-output"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-xyz-packed-staging"),
        size: plan.bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-xyz-packed-encoder"),
    });
    record_packed_xyz(&mut encoder, runtime, xyz, segments, mode, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, plan.bytes);
    runtime.queue().submit(Some(encoder.finish()));
    crate::readback::read_staging_xyz(device, &staging, segments.cell_count() as usize)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct XyzUniform {
    cells: u32,
    points: u32,
    _pad: [u32; 2],
}

pub(crate) fn record_packed_xyz(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    segments: &GpuVoxelSegments,
    mode: XyzMode,
    output: &wgpu::Buffer,
) {
    if mode == XyzMode::First {
        super::voxel_gather_packed::record_packed_voxel_gather(
            encoder, runtime, &xyz, segments, output,
        );
        return;
    }
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_reduce.packed_xyz_pipeline(device);
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-xyz-packed-uniform"),
        contents: bytemuck::bytes_of(&XyzUniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            _pad: [0; 2],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buffers = [
        &control,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        xyz[0],
        xyz[1],
        xyz[2],
        output,
    ];
    let entries: [_; 7] = std::array::from_fn(|binding| wgpu::BindGroupEntry {
        binding: binding as u32,
        resource: buffers[binding].as_entire_binding(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-xyz-packed-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-reduce-xyz-packed-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
}

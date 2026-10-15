use bytemuck::{Pod, Zeroable};
use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedGatherPlan {
    pub width: usize,
    pub channel_bytes: u64,
    pub offset: u64,
}

pub(crate) fn plan_packed_voxel_attributes(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    width: usize,
    target_bytes: u64,
    offset: u64,
    host_limit: u64,
) -> Option<PackedGatherPlan> {
    if cells == 0
        || channels < 2
        || ![2, 4].contains(&width)
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
        || offset % 4 != 0
        || target_bytes > limits.max_buffer_size
        || target_bytes > host_limit
    {
        return None;
    }
    let words = u64::from(cells).checked_mul(channels.min(width) as u64)?;
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let output_bytes = words.checked_mul(4)?;
    let attribute_bytes = channel_bytes.checked_mul(u64::try_from(channels).ok()?)?;
    if words > u64::from(u32::MAX)
        || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || offset.checked_add(attribute_bytes)? > target_bytes
    {
        return None;
    }
    Some(PackedGatherPlan { width, channel_bytes, offset })
}

pub(crate) fn try_record_packed_voxel_attributes(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    upload_recycle: &mut Vec<wgpu::Buffer>,
) -> SpatialResult<bool> {
    if channels.len() < 2 {
        return Ok(false);
    }
    let Some(plan) = plan_packed_voxel_attributes(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
        runtime.max_gather_channels() as usize,
        staging.size(),
        offset,
        isize::MAX as u64,
    ) else {
        return Ok(false);
    };
    record_packed_voxel_attributes(
        encoder,
        runtime,
        channels,
        segments,
        staging,
        upload_recycle,
        plan,
    )?;
    Ok(true)
}

pub(crate) fn record_packed_voxel_attributes(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    upload_recycle: &mut Vec<wgpu::Buffer>,
    plan: PackedGatherPlan,
) -> SpatialResult<()> {
    let device = runtime.device();
    // Keep the original batch width: a scalar tail returns +0 for empty cells,
    // whereas the original multi-channel kernels gather input point zero.
    for (batch, chunk) in channels.chunks(plan.width).enumerate() {
        let first_upload = upload_recycle.len();
        for channel in chunk {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        let values = &upload_recycle[first_upload..];
        let bytes = plan.channel_bytes * chunk.len() as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-attrs-packed-output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if chunk.len() == 1 {
            super::voxel_gather::record_voxel_gather_f32_pass(
                encoder,
                runtime,
                &values[0],
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                segments.cell_count(),
                segments.point_count(),
                &output,
            )?;
        } else {
            record_packed_voxel_gather(encoder, runtime, values, segments, &output);
        }
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            staging,
            plan.offset + plan.channel_bytes * (batch * plan.width) as u64,
            bytes,
        );
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    cells: u32,
    points: u32,
    channels: u32,
    _pad: u32,
}

pub(crate) fn record_packed_voxel_gather(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
) {
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_gather.packed_attributes_pipeline(device);
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-attrs-packed-uniform"),
        contents: bytemuck::bytes_of(&Uniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            channels: values.len() as u32,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // Inactive read-only slots alias the first input and are never read.
    let value = |i| values.get(i).unwrap_or(&values[0]);
    let buffers = [
        &control,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        value(0),
        value(1),
        value(2),
        value(3),
        output,
    ];
    let entries: [_; 8] = std::array::from_fn(|binding| wgpu::BindGroupEntry {
        binding: binding as u32,
        resource: buffers[binding].as_entire_binding(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-attrs-packed-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-attrs-packed-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
}

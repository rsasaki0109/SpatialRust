use bytemuck::{Pod, Zeroable};
use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AverageBatchPlan {
    pub channel_bytes: u64,
    pub staging_bytes: u64,
}

pub(crate) fn plan_voxel_average_batch(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
) -> Option<AverageBatchPlan> {
    plan_voxel_average_batch_with_host_limit(limits, cells, channels, isize::MAX as u64)
}

pub(crate) fn plan_voxel_average_batch_with_host_limit(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    host_limit: u64,
) -> Option<AverageBatchPlan> {
    if cells == 0
        || channels < 2
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let words = u64::from(cells).checked_mul(channels.min(4) as u64)?;
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let output_bytes = words.checked_mul(4)?;
    let staging_bytes = channel_bytes.checked_mul(u64::try_from(channels).ok()?)?;
    if words > u64::from(u32::MAX)
        || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || staging_bytes > limits.max_buffer_size
        || staging_bytes > host_limit
    {
        return None;
    }
    Some(AverageBatchPlan { channel_bytes, staging_bytes })
}

pub(crate) fn try_reduce_voxel_average_batched(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> Option<SpatialResult<Vec<Vec<f32>>>> {
    if channels.len() < 2 {
        return None;
    }
    let plan = plan_voxel_average_batch(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
    )?;
    Some(reduce_batched(runtime, channels, segments, plan))
}

fn reduce_batched(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    plan: AverageBatchPlan,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-multi-staging"),
        size: plan.staging_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-multi-encoder"),
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-multi-packed-output"),
        size: plan.channel_bytes * channels.len().min(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut controls = AverageBatchControls::default();
    for (batch, chunk) in channels.chunks(4).enumerate() {
        let values: Vec<_> = chunk
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-multi-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let bytes = plan.channel_bytes * chunk.len() as u64;
        if chunk.len() == 1 {
            super::voxel_reduce::record_voxel_reduce_f32_pass(
                &mut encoder,
                runtime,
                &values[0],
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                segments.cell_count(),
                segments.point_count(),
                &output,
            )?;
        } else {
            record_voxel_average_batch_with_controls(
                &mut encoder,
                runtime,
                &values,
                segments,
                &output,
                &mut controls,
            );
        }
        // Preserve this batch before the next pass overwrites the same output.
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            &staging,
            plan.channel_bytes * (batch * 4) as u64,
            bytes,
        );
    }
    runtime.queue().submit(Some(encoder.finish()));
    crate::readback::read_staging_f32_channels(
        device,
        &staging,
        channels.len(),
        segments.cell_count() as usize,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct Uniform {
    cells: u32,
    points: u32,
    channels: u32,
    _pad: u32,
}

#[derive(Default)]
struct AverageBatchControls {
    current: Option<(Uniform, wgpu::Buffer)>,
}

impl AverageBatchControls {
    fn get(&mut self, device: &wgpu::Device, uniform: Uniform) -> &wgpu::Buffer {
        if self.current.as_ref().map(|(key, _)| *key) != Some(uniform) {
            // Earlier encoded passes retain their original immutable control.
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-reduce-multi-packed-uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            self.current = Some((uniform, buffer));
        }
        &self.current.as_ref().expect("voxel average control initialized").1
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn record_voxel_average_batch(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
) {
    record_voxel_average_batch_with_controls(
        encoder,
        runtime,
        values,
        segments,
        output,
        &mut AverageBatchControls::default(),
    );
}

fn record_voxel_average_batch_with_controls(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
    controls: &mut AverageBatchControls,
) {
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_reduce.multi_average_pipeline(device);
    let control = controls.get(
        device,
        Uniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            channels: values.len() as u32,
            _pad: 0,
        },
    );
    // Inactive read-only slots alias the first input; the shader skips them.
    let value = |i| values.get(i).unwrap_or(&values[0]);
    let buffers = [
        control,
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
        label: Some("voxel-reduce-multi-packed-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-reduce-multi-packed-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
}

pub(crate) fn plan_voxel_attribute_average_batch(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    target_bytes: u64,
    offset: u64,
) -> Option<AverageBatchPlan> {
    plan_voxel_attribute_average_batch_with_host_limit(
        limits,
        cells,
        channels,
        target_bytes,
        offset,
        isize::MAX as u64,
    )
}

pub(crate) fn plan_voxel_attribute_average_batch_with_host_limit(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    target_bytes: u64,
    offset: u64,
    host_limit: u64,
) -> Option<AverageBatchPlan> {
    let plan = plan_voxel_average_batch_with_host_limit(limits, cells, channels, host_limit)?;
    if offset % 4 != 0
        || target_bytes > limits.max_buffer_size
        || target_bytes > host_limit
        || offset.checked_add(plan.staging_bytes)? > target_bytes
    {
        return None;
    }
    Some(plan)
}

pub(crate) fn try_record_voxel_attribute_averages_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
) -> SpatialResult<bool> {
    if channels.len() < 2 {
        return Ok(false);
    }
    let Some(plan) = plan_voxel_attribute_average_batch(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
        staging.size(),
        offset,
    ) else {
        return Ok(false);
    };
    record_voxel_attribute_averages_to_staging(
        encoder,
        runtime,
        channels,
        segments,
        staging,
        offset,
        plan.channel_bytes,
    )?;
    Ok(true)
}

pub(crate) fn record_voxel_attribute_averages_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    channel_bytes: u64,
) -> SpatialResult<()> {
    if channels.is_empty() {
        return Ok(());
    }
    let device = runtime.device();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-attrs-packed-output"),
        size: channel_bytes * channels.len().min(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut controls = AverageBatchControls::default();
    for (batch, chunk) in channels.chunks(4).enumerate() {
        let values: Vec<_> = chunk
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-reduce-xyz-attrs-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let bytes = channel_bytes * chunk.len() as u64;
        if chunk.len() == 1 {
            super::voxel_reduce::record_voxel_reduce_f32_pass(
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
            record_voxel_average_batch_with_controls(
                encoder,
                runtime,
                &values,
                segments,
                &output,
                &mut controls,
            );
        }
        // Copy only active planes, including a partial final batch.
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            staging,
            offset + channel_bytes * (batch * 4) as u64,
            bytes,
        );
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_average_reuse_tests.rs"]
mod reuse_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_average_control_tests.rs"]
mod control_tests;

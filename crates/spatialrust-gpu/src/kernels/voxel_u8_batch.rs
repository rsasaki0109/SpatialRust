use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BatchedU8Plan {
    pub output_bytes: u64,
}

pub(crate) fn plan_batched_voxel_u8(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
) -> Option<BatchedU8Plan> {
    if cells == 0
        || !(2..=4).contains(&channels)
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_storage_size < 4096
        || limits.max_compute_workgroups_per_dimension < cells.div_ceil(256)
    {
        return None;
    }
    let words = u64::from(cells).div_ceil(4).checked_mul(channels as u64)?;
    let output_bytes = words.checked_mul(4)?;
    if words > u64::from(u32::MAX)
        || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || output_bytes > limits.max_buffer_size
    {
        return None;
    }
    Some(BatchedU8Plan { output_bytes })
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
pub(crate) struct BatchedU8Controls {
    current: Option<(Uniform, wgpu::Buffer)>,
}

impl BatchedU8Controls {
    fn get(&mut self, device: &wgpu::Device, uniform: Uniform) -> &wgpu::Buffer {
        if self.current.as_ref().map(|(key, _)| *key) != Some(uniform) {
            // Earlier encoded passes retain their original immutable control.
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-u8-batch-uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            self.current = Some((uniform, buffer));
        }
        &self.current.as_ref().expect("voxel u8 control initialized").1
    }
}

pub(crate) fn record_voxel_u8_batch_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
    average: bool,
) {
    record_voxel_u8_batch_pass_with_controls(
        encoder,
        runtime,
        values,
        segments,
        output,
        average,
        &mut BatchedU8Controls::default(),
    );
}

pub(crate) fn record_voxel_u8_batch_pass_with_controls(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
    average: bool,
    controls: &mut BatchedU8Controls,
) {
    let device = runtime.device();
    let (layout, pipeline) = if average {
        runtime.pipelines().voxel_reduce.batched_u8_pipeline(device)
    } else {
        runtime.pipelines().voxel_gather.batched_u8_pipeline(device)
    };
    let control = controls.get(
        device,
        Uniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            channels: values.len() as u32,
            _pad: 0,
        },
    );
    // Inactive read-only bindings alias an active channel and are never read.
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
        label: Some("voxel-u8-batch-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-u8-batch-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(256), 1, 1);
}

pub(crate) fn try_record_voxel_u8_batch_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[u8]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    average: bool,
) -> bool {
    let Some(plan) =
        plan_batched_voxel_u8(&runtime.device().limits(), segments.cell_count(), channels.len())
    else {
        return false;
    };
    let values: Vec<_> = channels
        .iter()
        .map(|channel| {
            runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(if average {
                    "voxel-reduce-xyz-u8-values"
                } else {
                    "voxel-gather-xyz-u8-values"
                }),
                contents: &crate::readback::u8_storage_upload_contents(channel),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-u8-batch-output"),
        size: plan.output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    record_voxel_u8_batch_pass(encoder, runtime, &values, segments, &output, average);
    encoder.copy_buffer_to_buffer(&output, 0, staging, offset, plan.output_bytes);
    true
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_u8_control_tests.rs"]
mod control_tests;

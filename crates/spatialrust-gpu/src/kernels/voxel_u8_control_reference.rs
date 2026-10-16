// Complete original batched u8 pass from merged PR 249.
use super::super::*;

pub(super) fn baseline_record_voxel_u8_batch_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
    average: bool,
) {
    let device = runtime.device();
    let (layout, pipeline) = if average {
        runtime.pipelines().voxel_reduce.batched_u8_pipeline(device)
    } else {
        runtime.pipelines().voxel_gather.batched_u8_pipeline(device)
    };
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-u8-batch-uniform"),
        contents: bytemuck::bytes_of(&Uniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            channels: values.len() as u32,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // Inactive read-only bindings alias an active channel and are never read.
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

// Complete original packed average pass from merged PR 248.
use super::super::*;

pub(super) fn baseline_record_voxel_average_batch(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
) {
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_reduce.multi_average_pipeline(device);
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-multi-packed-uniform"),
        contents: bytemuck::bytes_of(&Uniform {
            cells: segments.cell_count(),
            points: segments.point_count(),
            channels: values.len() as u32,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // Inactive read-only slots alias the first input; the shader skips them.
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

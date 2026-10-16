// Complete original packed gather pass from merged PR 247.
use super::super::*;

pub(super) fn baseline_record_packed_voxel_gather<B: std::borrow::Borrow<wgpu::Buffer>>(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[B],
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
    let value = |i| values.get(i).unwrap_or(&values[0]).borrow();
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

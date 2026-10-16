// Complete original two-/four-output gather passes from merged PR 250.
use super::super::*;

pub(super) fn baseline_record_voxel_gather_multi2_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI2_CHANNELS],
    segments: &GpuVoxelSegments,
    outputs: &[&wgpu::Buffer; MULTI2_CHANNELS],
    channel_count: u32,
) -> SpatialResult<()> {
    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let cell_count = segments.cell_count();
    let uniform =
        GatherUniform { cell_count, point_count: segments.point_count(), channel_count, _pad: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi-bind-group"),
        layout: &pipelines.voxel_gather.multi_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: outputs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: outputs[1].as_entire_binding() },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-multi-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_gather.multi_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(super) fn baseline_record_voxel_gather_multi4_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    outputs: &[&wgpu::Buffer; MULTI4_CHANNELS],
    channel_count: u32,
) -> SpatialResult<()> {
    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let multi4_pipeline = pipelines.voxel_gather.multi4_pipeline.as_ref().ok_or_else(|| {
        SpatialError::InvalidArgument(
            "4-channel gather pipeline is unavailable on this gpu adapter".to_owned(),
        )
    })?;
    let multi4_layout =
        pipelines.voxel_gather.multi4_bind_group_layout.as_ref().expect("multi4 layout");
    let cell_count = segments.cell_count();
    let uniform =
        GatherUniform { cell_count, point_count: segments.point_count(), channel_count, _pad: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi4-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi4-bind-group"),
        layout: multi4_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: values[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: values[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: outputs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: outputs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: outputs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: outputs[3].as_entire_binding() },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-multi4-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(multi4_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

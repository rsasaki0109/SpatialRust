use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FirstAverageUniform {
    cells: u32,
    points: u32,
    attributes: u32,
    _pad: u32,
}

pub(crate) fn record_fused_voxel_first_average(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    attributes: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
) {
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_gather.first_xyz_average_pipeline(device);
    let uniform = FirstAverageUniform {
        cells: segments.cell_count(),
        points: segments.point_count(),
        attributes: attributes.len() as u32,
        _pad: 0,
    };
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-fused-first-average-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // Inactive read-only bindings alias the first channel; the shader never reads them.
    let attribute = |index| attributes.get(index).unwrap_or(&attributes[0]);
    let buffers = [
        &control,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        xyz[0],
        xyz[1],
        xyz[2],
        attribute(0),
        attribute(1),
        attribute(2),
        attribute(3),
        output,
    ];
    let entries: [_; 11] = std::array::from_fn(|binding| wgpu::BindGroupEntry {
        binding: binding as u32,
        resource: buffers[binding].as_entire_binding(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-fused-first-average-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-fused-first-average-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(WORKGROUP_SIZE), 1, 1);
}

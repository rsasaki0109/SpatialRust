use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

const WORKGROUP_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FusedAveragePlan {
    pub(crate) output_bytes: u64,
}

pub(crate) fn plan_fused_voxel_average(
    limits: &wgpu::Limits,
    cells: u32,
    attributes: usize,
) -> Option<FusedAveragePlan> {
    if cells == 0
        || !(1..=4).contains(&attributes)
        || limits.max_storage_buffers_per_shader_stage < 10
        || limits.max_bindings_per_bind_group < 11
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || cells.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let words = u64::from(cells).checked_mul((3 + attributes) as u64)?;
    let output_bytes = words.checked_mul(std::mem::size_of::<f32>() as u64)?;
    if words > u64::from(u32::MAX)
        || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || output_bytes > limits.max_buffer_size
    {
        return None;
    }
    Some(FusedAveragePlan { output_bytes })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedAverageUniform {
    cells: u32,
    points: u32,
    attributes: u32,
    _pad: u32,
}

pub(crate) fn record_fused_voxel_average(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    attributes: &[wgpu::Buffer],
    segments: &GpuVoxelSegments,
    output: &wgpu::Buffer,
) {
    let device = runtime.device();
    let (layout, pipeline) = runtime.pipelines().voxel_reduce.fused_xyz_attributes_pipeline(device);
    let uniform = FusedAverageUniform {
        cells: segments.cell_count(),
        points: segments.point_count(),
        attributes: attributes.len() as u32,
        _pad: 0,
    };
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-fused-average-uniform"),
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
        label: Some("voxel-fused-average-group"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-fused-average-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(segments.cell_count().div_ceil(WORKGROUP_SIZE), 1, 1);
}

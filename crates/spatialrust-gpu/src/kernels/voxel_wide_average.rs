use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use super::{
    voxel_average_batch::{
        plan_voxel_attribute_average_batch_with_host_limit,
        record_voxel_attribute_averages_to_staging,
    },
    voxel_first_average_fused::record_fused_voxel_first_average,
    voxel_reduce_fused::{plan_fused_voxel_average, record_fused_voxel_average},
    voxel_xyz_packed::XyzMode,
};
use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WideAveragePlan {
    pub prefix_bytes: u64,
    pub channel_bytes: u64,
}

pub(crate) fn plan_wide_voxel_average(
    limits: &wgpu::Limits,
    cells: u32,
    attributes: usize,
    target_bytes: u64,
    host_limit: u64,
) -> Option<WideAveragePlan> {
    if attributes <= 4
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 16
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || target_bytes > limits.max_buffer_size
        || target_bytes > host_limit
    {
        return None;
    }
    let prefix = plan_fused_voxel_average(limits, cells, 4)?;
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let float_bytes = channel_bytes.checked_mul(u64::try_from(attributes).ok()?.checked_add(3)?)?;
    if float_bytes > target_bytes || prefix.output_bytes > host_limit {
        return None;
    }
    if attributes > 5 {
        plan_voxel_attribute_average_batch_with_host_limit(
            limits,
            cells,
            attributes - 4,
            target_bytes,
            prefix.output_bytes,
            host_limit,
        )?;
    }
    Some(WideAveragePlan { prefix_bytes: prefix.output_bytes, channel_bytes })
}

pub(crate) fn try_record_wide_voxel_average(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    attributes: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    mode: XyzMode,
) -> SpatialResult<bool> {
    let Some(plan) = plan_wide_voxel_average(
        &runtime.device().limits(),
        segments.cell_count(),
        attributes.len(),
        staging.size(),
        isize::MAX as u64,
    ) else {
        return Ok(false);
    };
    let device = runtime.device();
    let values: Vec<_> = attributes[..4]
        .iter()
        .map(|channel| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-reduce-xyz-attrs-values"),
                contents: bytemuck::cast_slice(channel),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-wide-average-prefix-output"),
        size: plan.prefix_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    match mode {
        XyzMode::First => {
            record_fused_voxel_first_average(encoder, runtime, xyz, &values, segments, &output);
        }
        XyzMode::Centroid => {
            record_fused_voxel_average(encoder, runtime, xyz, &values, segments, &output);
        }
    }
    encoder.copy_buffer_to_buffer(&output, 0, staging, 0, plan.prefix_bytes);
    record_voxel_attribute_averages_to_staging(
        encoder,
        runtime,
        &attributes[4..],
        segments,
        staging,
        plan.prefix_bytes,
        plan.channel_bytes,
    )?;
    Ok(true)
}

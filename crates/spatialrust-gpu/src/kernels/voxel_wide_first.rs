use spatialrust_core::SpatialResult;

use super::{
    voxel_centroid_first_fused::record_fused_voxel_centroid_first,
    voxel_first_gather_fused::record_fused_voxel_first_gather,
    voxel_gather_packed::{
        plan_packed_voxel_attributes, record_packed_voxel_attributes, PackedGatherPlan,
    },
    voxel_reduce_fused::plan_fused_voxel_average,
    voxel_xyz_packed::XyzMode,
};
use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WideFirstPlan {
    pub prefix_bytes: u64,
    pub tail: PackedGatherPlan,
}

pub(crate) fn plan_wide_voxel_first(
    limits: &wgpu::Limits,
    cells: u32,
    attributes: usize,
    width: usize,
    target_bytes: u64,
    host_limit: u64,
) -> Option<WideFirstPlan> {
    if attributes <= 4
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 16
        || limits.max_buffer_size < 16
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
    {
        return None;
    }
    let prefix = plan_fused_voxel_average(limits, cells, 4)?;
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let attributes_plan = plan_packed_voxel_attributes(
        limits,
        cells,
        attributes,
        width,
        target_bytes,
        channel_bytes.checked_mul(3)?,
        host_limit,
    )?;
    // Four prefix attributes preserve the original two-/four-wide batch boundaries.
    // The original full attribute plan bounds every remaining batch, including a scalar tail.
    Some(WideFirstPlan {
        prefix_bytes: prefix.output_bytes,
        tail: PackedGatherPlan { offset: prefix.output_bytes, ..attributes_plan },
    })
}

pub(crate) fn try_record_wide_voxel_first(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    attributes: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    mode: XyzMode,
    upload_recycle: &mut Vec<wgpu::Buffer>,
) -> SpatialResult<bool> {
    let Some(plan) = plan_wide_voxel_first(
        &runtime.device().limits(),
        segments.cell_count(),
        attributes.len(),
        runtime.max_gather_channels() as usize,
        staging.size(),
        isize::MAX as u64,
    ) else {
        return Ok(false);
    };
    record_wide_voxel_first(
        encoder,
        runtime,
        xyz,
        attributes,
        segments,
        staging,
        mode,
        upload_recycle,
        plan,
    )?;
    Ok(true)
}

// Keep borrowed inputs, destinations and the checked plan explicit at this recording boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_wide_voxel_first(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    xyz: [&wgpu::Buffer; 3],
    attributes: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    mode: XyzMode,
    upload_recycle: &mut Vec<wgpu::Buffer>,
    plan: WideFirstPlan,
) -> SpatialResult<()> {
    let first_upload = upload_recycle.len();
    for channel in &attributes[..4] {
        upload_recycle.push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
    }
    let values = &upload_recycle[first_upload..first_upload + 4];
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-wide-first-prefix-output"),
        size: plan.prefix_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    match mode {
        XyzMode::First => {
            record_fused_voxel_first_gather(encoder, runtime, xyz, values, segments, &output);
        }
        XyzMode::Centroid => {
            record_fused_voxel_centroid_first(encoder, runtime, xyz, values, segments, &output);
        }
    }
    encoder.copy_buffer_to_buffer(&output, 0, staging, 0, plan.prefix_bytes);
    record_packed_voxel_attributes(
        encoder,
        runtime,
        &attributes[4..],
        segments,
        staging,
        upload_recycle,
        plan.tail,
    )
}

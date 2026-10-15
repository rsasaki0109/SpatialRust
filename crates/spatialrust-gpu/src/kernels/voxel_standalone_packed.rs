use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedSubmissionPlan {
    pub width: usize,
    pub channels_per_submission: usize,
    pub channel_bytes: u64,
}

pub(crate) fn plan_packed_voxel_submission(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    width: usize,
    host_limit: u64,
) -> Option<PackedSubmissionPlan> {
    if cells == 0
        || channels == 0
        || ![2, 4].contains(&width)
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_storage_buffers_per_shader_stage < (2 + 2 * width) as u32
        || limits.max_bindings_per_bind_group < (3 + 2 * width) as u32
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let channels_per_submission = if channels > width {
        // Preserve the original coalesced policy and its acceptance limits.
        super::voxel_gather_submit::plan_voxel_gather_submission_with_host_limit(
            limits, cells, channels, width, host_limit,
        )?
        .channels_per_submission
    } else {
        channels
    };
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let output_words = u64::from(cells).checked_mul(channels.min(width) as u64)?;
    let output_bytes = output_words.checked_mul(4)?;
    let staging_bytes = channel_bytes.checked_mul(channels_per_submission as u64)?;
    if output_words > u64::from(u32::MAX)
        || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || staging_bytes > limits.max_buffer_size
        || staging_bytes > host_limit
    {
        return None;
    }
    Some(PackedSubmissionPlan { width, channels_per_submission, channel_bytes })
}

pub(crate) fn try_gather_voxel_channels_packed(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> Option<SpatialResult<Vec<Vec<f32>>>> {
    if channels.is_empty() {
        return None;
    }
    let plan = plan_packed_voxel_submission(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
        runtime.max_gather_channels() as usize,
        isize::MAX as u64,
    )?;
    Some(gather_voxel_channels_packed_with_plan(runtime, channels, segments, plan))
}

pub(crate) fn gather_voxel_channels_packed_with_plan(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    plan: PackedSubmissionPlan,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let mut gathered = Vec::with_capacity(channels.len());
    // At most two original batches retain inputs/outputs through each wait.
    for window in channels.chunks(plan.channels_per_submission) {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-packed-staging"),
            size: plan.channel_bytes * window.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-gather-packed-encoder"),
        });
        record_packed_voxel_window(&mut encoder, runtime, window, segments, &staging, 0, plan);
        runtime.queue().submit(Some(encoder.finish()));
        gathered.extend(crate::readback::read_staging_f32_channels(
            device,
            &staging,
            window.len(),
            segments.cell_count() as usize,
        )?);
    }
    Ok(gathered)
}

pub(crate) fn record_packed_voxel_window(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    window: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging: &wgpu::Buffer,
    offset: u64,
    plan: PackedSubmissionPlan,
) {
    let device = runtime.device();
    for (batch, chunk) in window.chunks(plan.width).enumerate() {
        let values: Vec<_> = chunk
            .iter()
            .map(|channel| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-gather-multi-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let bytes = plan.channel_bytes * chunk.len() as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-packed-output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Standalone one-channel tails also gather point zero in empty cells.
        super::voxel_gather_packed::record_packed_voxel_gather(
            encoder, runtime, &values, segments, &output,
        );
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            staging,
            offset + plan.channel_bytes * (batch * plan.width) as u64,
            bytes,
        );
    }
}

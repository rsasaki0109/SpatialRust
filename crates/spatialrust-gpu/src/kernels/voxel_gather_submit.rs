use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::{GpuVoxelSegments, WgpuRuntime};

use super::voxel_gather::{
    record_voxel_gather_multi2_pass_with_controls, record_voxel_gather_multi4_pass_with_controls,
    LegacyGatherControls,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatherSubmissionPlan {
    pub batch_width: usize,
    pub channels_per_submission: usize,
    pub channel_bytes: u64,
}

pub(crate) fn plan_voxel_gather_submission(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    batch_width: usize,
) -> Option<GatherSubmissionPlan> {
    plan_voxel_gather_submission_with_host_limit(
        limits,
        cells,
        channels,
        batch_width,
        isize::MAX as u64,
    )
}

pub(crate) fn plan_voxel_gather_submission_with_host_limit(
    limits: &wgpu::Limits,
    cells: u32,
    channels: usize,
    batch_width: usize,
    host_staging_limit: u64,
) -> Option<GatherSubmissionPlan> {
    if cells == 0
        || ![2, 4].contains(&batch_width)
        || channels <= batch_width
        || limits.max_storage_buffers_per_shader_stage < (2 + 2 * batch_width) as u32
        || limits.max_bindings_per_bind_group < (3 + 2 * batch_width) as u32
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || cells.div_ceil(256) > limits.max_compute_workgroups_per_dimension
    {
        return None;
    }
    let channel_bytes = u64::from(cells).checked_mul(4)?;
    let channels_per_submission = channels.min(2 * batch_width);
    let staging_bytes = channel_bytes.checked_mul(channels_per_submission as u64)?;
    if channel_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || staging_bytes > limits.max_buffer_size
        || staging_bytes > host_staging_limit
    {
        return None;
    }
    Some(GatherSubmissionPlan { batch_width, channels_per_submission, channel_bytes })
}

pub(crate) fn try_gather_voxel_channels_coalesced(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> Option<SpatialResult<Vec<Vec<f32>>>> {
    let batch_width = runtime.max_gather_channels() as usize;
    if channels.len() <= batch_width {
        return None;
    }
    let plan = plan_voxel_gather_submission(
        &runtime.device().limits(),
        segments.cell_count(),
        channels.len(),
        batch_width,
    )?;
    Some(gather_voxel_channels_with_plan(runtime, channels, segments, plan))
}

pub(crate) fn gather_voxel_channels_with_plan(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    plan: GatherSubmissionPlan,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let empty = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut gathered = Vec::with_capacity(channels.len());
    // At most two original batches retain device inputs/outputs until each wait.
    for window in channels.chunks(plan.channels_per_submission) {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-coalesced-staging"),
            size: plan.channel_bytes * window.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-gather-coalesced-encoder"),
        });
        record_window(&mut encoder, runtime, window, segments, &empty, &staging, plan)?;
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

fn record_window(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    empty: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    plan: GatherSubmissionPlan,
) -> SpatialResult<()> {
    if channels.is_empty() {
        return Ok(());
    }
    let device = runtime.device();
    let output_width = if channels.len().min(plan.batch_width) > 2 { 4 } else { 2 };
    let outputs: Vec<_> = (0..output_width)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if output_width == 4 {
                    "voxel-gather-multi4-output"
                } else {
                    "voxel-gather-multi-output"
                }),
                size: plan.channel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let mut controls = LegacyGatherControls::default();
    for (batch, chunk) in channels.chunks(plan.batch_width).enumerate() {
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
        // Preserve the original two/four-output pipeline, including unused slots.
        let width = if chunk.len() > 2 { 4 } else { 2 };
        if width == 4 {
            let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
            let output_refs = std::array::from_fn(|i| &outputs[i]);
            record_voxel_gather_multi4_pass_with_controls(
                encoder,
                runtime,
                &inputs,
                segments,
                &output_refs,
                chunk.len() as u32,
                &mut controls,
            )?;
        } else {
            let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
            let output_refs = std::array::from_fn(|i| &outputs[i]);
            record_voxel_gather_multi2_pass_with_controls(
                encoder,
                runtime,
                &inputs,
                segments,
                &output_refs,
                chunk.len() as u32,
                &mut controls,
            )?;
        }
        for (index, output) in outputs.iter().take(chunk.len()).enumerate() {
            let channel = batch * plan.batch_width + index;
            encoder.copy_buffer_to_buffer(
                output,
                0,
                staging,
                plan.channel_bytes * channel as u64,
                plan.channel_bytes,
            );
        }
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_legacy_reuse_tests.rs"]
mod reuse_tests;

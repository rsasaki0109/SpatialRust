// Complete legacy fallback call and window recorder from merged PR 243.
use super::super::GatherSubmissionPlan;
use crate::kernels::voxel_gather::{
    record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
};
use crate::{GpuVoxelSegments, WgpuRuntime};
use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

pub(super) fn baseline_gather_voxel_channels_with_plan(
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
        baseline_record_window(&mut encoder, runtime, window, segments, &empty, &staging, plan)?;
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

pub(super) fn baseline_record_window(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    empty: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    plan: GatherSubmissionPlan,
) -> SpatialResult<()> {
    let device = runtime.device();
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
        let outputs: Vec<_> = (0..width)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if width == 4 {
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
        if width == 4 {
            let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
            let output_refs = std::array::from_fn(|i| &outputs[i]);
            record_voxel_gather_multi4_pass(
                encoder,
                runtime,
                &inputs,
                segments,
                &output_refs,
                chunk.len() as u32,
            )?;
        } else {
            let inputs = std::array::from_fn(|i| values.get(i).unwrap_or(empty));
            let output_refs = std::array::from_fn(|i| &outputs[i]);
            record_voxel_gather_multi2_pass(
                encoder,
                runtime,
                &inputs,
                segments,
                &output_refs,
                chunk.len() as u32,
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

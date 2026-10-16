// Original combined recorder from merged PR 245; fallback body is unchanged.
use super::super::*;

pub(super) fn baseline_record_preferred(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    attribute_channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging_buffer: &wgpu::Buffer,
    channel_len: u64,
    upload_recycle: &mut Vec<wgpu::Buffer>,
) -> SpatialResult<()> {
    if attribute_channels.is_empty() {
        return Ok(());
    }

    if super::voxel_gather_packed::try_record_packed_voxel_attributes(
        encoder,
        runtime,
        attribute_channels,
        segments,
        staging_buffer,
        channel_len * 3,
        upload_recycle,
    )? {
        return Ok(());
    }

    baseline_record_fallback(
        encoder,
        runtime,
        attribute_channels,
        segments,
        staging_buffer,
        channel_len,
        upload_recycle,
    )
}

pub(super) fn baseline_record_fallback(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    attribute_channels: &[&[f32]],
    segments: &GpuVoxelSegments,
    staging_buffer: &wgpu::Buffer,
    channel_len: u64,
    upload_recycle: &mut Vec<wgpu::Buffer>,
) -> SpatialResult<()> {
    let device = runtime.device();
    let max_channels = runtime.max_gather_channels().max(1) as usize;

    for chunk_start in (0..attribute_channels.len()).step_by(max_channels) {
        let chunk_end = (chunk_start + max_channels).min(attribute_channels.len());
        let chunk = &attribute_channels[chunk_start..chunk_end];
        let channels_in_chunk = chunk.len();

        let chunk_upload_start = upload_recycle.len();
        for channel in chunk {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        let value_buffers = &upload_recycle[chunk_upload_start..];

        let output_buffers: Vec<wgpu::Buffer> = (0..channels_in_chunk)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel-gather-xyz-attrs-output"),
                    size: channel_len,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        match channels_in_chunk {
            1 => record_voxel_gather_f32_pass(
                encoder,
                runtime,
                &value_buffers[0],
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                segments.cell_count(),
                segments.point_count(),
                &output_buffers[0],
            )?,
            2 => {
                let values: [&wgpu::Buffer; MULTI2_CHANNELS] =
                    [&value_buffers[0], &value_buffers[1]];
                let outputs: [&wgpu::Buffer; MULTI2_CHANNELS] =
                    [&output_buffers[0], &output_buffers[1]];
                record_voxel_gather_multi2_pass(encoder, runtime, &values, segments, &outputs, 2)?;
            }
            _ => {
                if runtime.pipelines().voxel_gather.multi4_pipeline.is_some() {
                    record_voxel_gather_multi4_active_outputs(
                        encoder,
                        runtime,
                        value_buffers,
                        segments,
                        &output_buffers,
                        channel_len,
                    )?;
                } else {
                    for (local_index, channel) in chunk.iter().enumerate() {
                        let chunk_upload_start = upload_recycle.len();
                        upload_recycle.push(
                            runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?,
                        );
                        record_voxel_gather_f32_pass(
                            encoder,
                            runtime,
                            &upload_recycle[chunk_upload_start],
                            segments.point_indices_buffer(),
                            segments.cell_starts_buffer(),
                            segments.cell_count(),
                            segments.point_count(),
                            &output_buffers[local_index],
                        )?;
                    }
                }
            }
        }

        for (local_index, output_buffer) in output_buffers.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                output_buffer,
                0,
                staging_buffer,
                channel_len * (3 + chunk_start + local_index) as u64,
                channel_len,
            );
        }
    }

    Ok(())
}

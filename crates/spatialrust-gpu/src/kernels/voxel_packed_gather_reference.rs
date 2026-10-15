// Complete public calls and shared attribute helper frozen from 381ecfe849ec3ed9686eccd3a405ebb0c4c1b2e1.
use super::*;
use crate::kernels::voxel_centroid_first_fused::record_fused_voxel_centroid_first;
use crate::kernels::voxel_first_gather_fused::record_fused_voxel_first_gather;
use crate::kernels::voxel_gather::{
    record_voxel_gather_f32_pass, record_voxel_gather_multi2_pass, record_voxel_gather_multi4_pass,
    record_voxel_gather_u8_packed_pass, record_voxel_gather_xyz_and_attrs4_packed_pass,
    record_voxel_gather_xyz_pass,
};
use crate::kernels::voxel_reduce::record_voxel_reduce_xyz_pass;
use crate::kernels::voxel_reduce_fused::plan_fused_voxel_average;
use crate::readback::{
    packed_u8_output_staging_bytes, pad_u8_for_gpu_storage, read_staging_xyz_and_packed_attributes,
};
use wgpu::util::DeviceExt;
const MULTI2_CHANNELS: usize = 2;
const MULTI4_CHANNELS: usize = 4;
pub(super) fn baseline_gather_voxel_first_xyz_and_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    let attribute_count = attribute_channels.len();
    let u8_attribute_count = u8_attribute_channels.len();
    if segments.cell_count() == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![Vec::new(); attribute_count],
            vec![Vec::new(); u8_attribute_count],
        ));
    }

    let point_count = segments.point_count() as usize;
    for channel in attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }
    for channel in u8_attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let f32_channel_count = 3 + attribute_count;
    let u8_staging_len = packed_u8_output_staging_bytes(cells, u8_attribute_count);
    let u8_channel_len = packed_u8_output_staging_bytes(cells, 1);
    let fused_plan = plan_fused_voxel_average(&device.limits(), cell_count, attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let fused_xyz_attrs4 = attribute_count == 4
        && u8_attribute_count == 0
        && runtime.pipelines().voxel_gather.xyz_attrs4_pipeline.is_some();

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-attrs-encoder"),
    });
    let mut upload_recycle = Vec::new();

    if fused_xyz_attrs4 {
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs4-values", channel)?);
        }
        let attr_refs: [&wgpu::Buffer; MULTI4_CHANNELS] =
            std::array::from_fn(|index| &upload_recycle[index]);
        let packed_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-attrs4-packed-output"),
            size: (channel_len * f32_channel_count) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        record_voxel_gather_xyz_and_attrs4_packed_pass(
            &mut encoder,
            runtime,
            x,
            y,
            z,
            &attr_refs,
            segments,
            &packed_output,
        )?;
        encoder.copy_buffer_to_buffer(
            &packed_output,
            0,
            &staging_buffer,
            0,
            (channel_len * f32_channel_count) as u64,
        );
    } else if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-first-gather-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        record_fused_voxel_first_gather(
            &mut encoder,
            runtime,
            [x, y, z],
            &upload_recycle,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else {
        let output_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-out-x"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-out-y"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_z = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-out-z"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_gather_xyz_pass(
            &mut encoder,
            runtime,
            x,
            y,
            z,
            segments,
            &output_x,
            &output_y,
            &output_z,
        )?;

        encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
        encoder.copy_buffer_to_buffer(
            &output_y,
            0,
            &staging_buffer,
            channel_len as u64,
            channel_len as u64,
        );
        encoder.copy_buffer_to_buffer(
            &output_z,
            0,
            &staging_buffer,
            (channel_len * 2) as u64,
            channel_len as u64,
        );

        record_gather_f32_attribute_channels_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            channel_len as u64,
            &mut upload_recycle,
        )?;
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        false,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-gather-xyz-u8-values"),
                contents: &pad_u8_for_gpu_storage(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            record_voxel_gather_u8_packed_pass(
                &mut encoder,
                runtime,
                &values_buffer,
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                cell_count,
                segments.point_count(),
                &output_buffer,
            )?;

            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                u8_region_offset + attribute_index as u64 * u8_channel_len as u64,
                u8_channel_len as u64,
            );
        }
    }

    queue.submit(Some(encoder.finish()));

    let output = read_staging_xyz_and_packed_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok(output)
}

pub(super) fn baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    use crate::kernels::voxel_gather::record_voxel_gather_u8_packed_pass;

    let attribute_count = attribute_channels.len();
    let u8_attribute_count = u8_attribute_channels.len();
    if segments.cell_count() == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![Vec::new(); attribute_count],
            vec![Vec::new(); u8_attribute_count],
        ));
    }

    let point_count = segments.point_count() as usize;
    for channel in attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }
    for channel in u8_attribute_channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let f32_channel_count = 3 + attribute_count;
    let u8_staging_len = packed_u8_output_staging_bytes(cells, u8_attribute_count);
    let u8_channel_len = packed_u8_output_staging_bytes(cells, 1);
    let fused_plan = plan_fused_voxel_average(&device.limits(), cell_count, attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-encoder"),
    });
    let mut upload_recycle = Vec::new();
    if let Some(plan) = fused_plan {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-fused-centroid-first-output"),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for channel in attribute_channels {
            upload_recycle
                .push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
        }
        record_fused_voxel_centroid_first(
            &mut encoder,
            runtime,
            [x, y, z],
            &upload_recycle,
            segments,
            &output,
        );
        encoder.copy_buffer_to_buffer(&output, 0, &staging_buffer, 0, plan.output_bytes);
    } else {
        let output_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-out-x"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-out-y"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_z = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-out-z"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_reduce_xyz_pass(
            &mut encoder,
            runtime,
            x,
            y,
            z,
            segments.point_indices_buffer(),
            segments.cell_starts_buffer(),
            cell_count,
            segments.point_count(),
            &output_x,
            &output_y,
            &output_z,
        )?;
        encoder.copy_buffer_to_buffer(&output_x, 0, &staging_buffer, 0, channel_len as u64);
        encoder.copy_buffer_to_buffer(
            &output_y,
            0,
            &staging_buffer,
            channel_len as u64,
            channel_len as u64,
        );
        encoder.copy_buffer_to_buffer(
            &output_z,
            0,
            &staging_buffer,
            (channel_len * 2) as u64,
            channel_len as u64,
        );

        record_gather_f32_attribute_channels_to_staging(
            &mut encoder,
            runtime,
            attribute_channels,
            segments,
            &staging_buffer,
            channel_len as u64,
            &mut upload_recycle,
        )?;
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    if !crate::kernels::voxel_u8_batch::try_record_voxel_u8_batch_to_staging(
        &mut encoder,
        runtime,
        u8_attribute_channels,
        segments,
        &staging_buffer,
        u8_region_offset,
        false,
    ) {
        for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
            let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-gather-xyz-u8-values"),
                contents: &pad_u8_for_gpu_storage(channel),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-gather-xyz-u8-output"),
                size: u8_channel_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            record_voxel_gather_u8_packed_pass(
                &mut encoder,
                runtime,
                &values_buffer,
                segments.point_indices_buffer(),
                segments.cell_starts_buffer(),
                cell_count,
                segments.point_count(),
                &output_buffer,
            )?;
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                u8_region_offset + attribute_index as u64 * u8_channel_len as u64,
                u8_channel_len as u64,
            );
        }
    }

    queue.submit(Some(encoder.finish()));
    let output = read_staging_xyz_and_packed_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok(output)
}

fn record_gather_f32_attribute_channels_to_staging(
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
            channel_count => {
                if runtime.pipelines().voxel_gather.multi4_pipeline.is_some() {
                    let empty = empty_storage_buffer(device)?;
                    let dummy_outputs: [wgpu::Buffer; MULTI4_CHANNELS] =
                        std::array::from_fn(|_| {
                            device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("voxel-gather-xyz-attrs-dummy-output"),
                                size: channel_len,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            })
                        });
                    let values: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
                        if index < channel_count {
                            &value_buffers[index]
                        } else {
                            &empty
                        }
                    });
                    let outputs: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
                        if index < channel_count {
                            &output_buffers[index]
                        } else {
                            &dummy_outputs[index]
                        }
                    });
                    record_voxel_gather_multi4_pass(
                        encoder,
                        runtime,
                        &values,
                        segments,
                        &outputs,
                        channel_count as u32,
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

fn empty_storage_buffer(device: &wgpu::Device) -> SpatialResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

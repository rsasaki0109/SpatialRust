// Complete public parent calls frozen from a884c9f383938fb5ef4be2d6b150be8380697e78.
use super::*;
use crate::kernels::voxel_gather::{
    record_gather_f32_attribute_channels_to_staging, record_voxel_gather_u8_pass,
    record_voxel_gather_xyz_and_attrs4_packed_pass, record_voxel_gather_xyz_pass,
};
use crate::kernels::voxel_reduce::{
    record_voxel_reduce_f32_pass, record_voxel_reduce_u8_pass, record_voxel_reduce_xyz_pass,
};
use crate::readback::{
    pad_u8_for_gpu_storage, read_staging_xyz_and_attributes, u8_output_staging_bytes,
};
use wgpu::util::DeviceExt;
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
    let u8_staging_len = u8_output_staging_bytes(cells, u8_attribute_count);
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
    for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-gather-xyz-u8-values"),
            contents: &pad_u8_for_gpu_storage(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-u8-output"),
            size: (cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_gather_u8_pass(
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
            u8_region_offset + attribute_index as u64 * (cells * std::mem::size_of::<u32>()) as u64,
            (cells * std::mem::size_of::<u32>()) as u64,
        );
    }

    queue.submit(Some(encoder.finish()));

    let output = read_staging_xyz_and_attributes(
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

pub(super) fn baseline_gather_voxel_first_xyz_and_average_multi_gpu(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_channels: &[&[f32]],
    u8_attribute_channels: &[&[u8]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
    use crate::kernels::voxel_reduce::record_voxel_reduce_f32_pass;
    use crate::kernels::voxel_reduce::record_voxel_reduce_u8_pass;

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
    let u8_staging_len = u8_output_staging_bytes(cells, u8_attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-reduce-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-reduce-attrs-encoder"),
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

    for (attribute_index, channel) in attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-xyz-attrs-values"),
            contents: bytemuck::cast_slice(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-attrs-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        record_voxel_reduce_f32_pass(
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
            (channel_len * (3 + attribute_index)) as u64,
            channel_len as u64,
        );
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-xyz-u8-values"),
            contents: &pad_u8_for_gpu_storage(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-u8-output"),
            size: (cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        record_voxel_reduce_u8_pass(
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
            u8_region_offset + attribute_index as u64 * (cells * std::mem::size_of::<u32>()) as u64,
            (cells * std::mem::size_of::<u32>()) as u64,
        );
    }

    queue.submit(Some(encoder.finish()));
    let output = read_staging_xyz_and_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
    Ok(output)
}

pub(super) fn baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu(
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
    let u8_staging_len = u8_output_staging_bytes(cells, u8_attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;

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
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-attrs-encoder"),
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

    for (attribute_index, channel) in attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-xyz-attrs-values"),
            contents: bytemuck::cast_slice(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-attrs-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_reduce_f32_pass(
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
            (channel_len * (3 + attribute_index)) as u64,
            channel_len as u64,
        );
    }

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-xyz-u8-values"),
            contents: &pad_u8_for_gpu_storage(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-xyz-u8-output"),
            size: (cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        record_voxel_reduce_u8_pass(
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
            u8_region_offset + attribute_index as u64 * (cells * std::mem::size_of::<u32>()) as u64,
            (cells * std::mem::size_of::<u32>()) as u64,
        );
    }

    queue.submit(Some(encoder.finish()));

    let output = read_staging_xyz_and_attributes(
        device,
        &staging_buffer,
        cells,
        attribute_count,
        u8_attribute_count,
    )?;
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
    use crate::kernels::voxel_gather::record_gather_f32_attribute_channels_to_staging;
    use crate::kernels::voxel_gather::record_voxel_gather_u8_pass;

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
    let u8_staging_len = u8_output_staging_bytes(cells, u8_attribute_count);
    let staging_size = channel_len * f32_channel_count + u8_staging_len;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-staging"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-gather-attrs-encoder"),
    });
    let mut upload_recycle = Vec::new();
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

    let u8_region_offset = (channel_len * f32_channel_count) as u64;
    for (attribute_index, channel) in u8_attribute_channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-gather-xyz-u8-values"),
            contents: &pad_u8_for_gpu_storage(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-xyz-u8-output"),
            size: (cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        record_voxel_gather_u8_pass(
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
            u8_region_offset + attribute_index as u64 * (cells * std::mem::size_of::<u32>()) as u64,
            (cells * std::mem::size_of::<u32>()) as u64,
        );
    }

    queue.submit(Some(encoder.finish()));
    let output = read_staging_xyz_and_attributes(
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

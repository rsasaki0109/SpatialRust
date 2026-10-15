// Complete parent public calls and decoder frozen from dcdbe2ea3a9a49bec16e3e01387a53b9945df58b.
use super::*;
use crate::kernels::voxel_gather::{
    record_gather_f32_attribute_channels_to_staging, record_voxel_gather_u8_pass,
    record_voxel_gather_xyz_and_attrs4_packed_pass, record_voxel_gather_xyz_pass,
};
use crate::kernels::voxel_reduce::{
    record_voxel_reduce_f32_pass, record_voxel_reduce_u8_pass, record_voxel_reduce_xyz_pass,
};
use crate::readback::{pad_u8_for_gpu_storage, u8_output_staging_bytes};
use wgpu::util::DeviceExt;
const MULTI4_CHANNELS: usize = 4;
const U32_OUTPUT_BYTES_PER_CELL: usize = std::mem::size_of::<u32>();

pub(super) fn parent_gather_voxel_first_xyz_and_multi_gpu(
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

    let (flat, u8_raw) = parent_read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        parent_unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        parent_split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        parent_split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

pub(super) fn parent_gather_voxel_first_xyz_and_average_multi_gpu(
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
    let (flat, u8_raw) = parent_read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        parent_unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        parent_split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        parent_split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

pub(super) fn parent_reduce_voxel_centroids_xyz_and_average_multi_gpu(
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

    let (flat, u8_raw) = parent_read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        parent_unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        parent_split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        parent_split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

pub(super) fn parent_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
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
    let (flat, u8_raw) = parent_read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        parent_unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        parent_split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        parent_split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

pub(super) fn parent_read_staging_f32_and_u8(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    f32_len: usize,
    u8_len: usize,
) -> SpatialResult<(Vec<f32>, Vec<u8>)> {
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;

    let data = slice.get_mapped_range();
    let f32_bytes = f32_len * std::mem::size_of::<f32>();
    let f32_values: Vec<f32> = bytemuck::cast_slice(&data[..f32_bytes])[..f32_len].to_vec();
    let u8_values = data[f32_bytes..f32_bytes + u8_len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok((f32_values, u8_values))
}

pub(super) fn parent_split_channel_blocks(
    mut flat: Vec<f32>,
    channels: usize,
    cells: usize,
) -> Vec<Vec<f32>> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return vec![flat];
    }

    let mut out = Vec::with_capacity(channels);
    for index in (1..channels).rev() {
        out.push(flat.split_off(index * cells));
    }
    out.push(flat);
    out.reverse();
    out
}

pub(super) fn parent_split_xyz_and_attribute_blocks(
    flat: Vec<f32>,
    attribute_count: usize,
    cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let blocks = parent_split_channel_blocks(flat, 3 + attribute_count, cells);
    let mut iter = blocks.into_iter();
    let out_x = iter.next().unwrap_or_default();
    let out_y = iter.next().unwrap_or_default();
    let out_z = iter.next().unwrap_or_default();
    let attributes = iter.collect();
    (out_x, out_y, out_z, attributes)
}

pub(super) fn parent_split_u8_channel_blocks(
    flat: Vec<u8>,
    channels: usize,
    cells: usize,
) -> Vec<Vec<u8>> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return vec![flat];
    }

    let mut out = Vec::with_capacity(channels);
    for index in 0..channels {
        let start = index * cells;
        out.push(flat[start..start + cells].to_vec());
    }
    out
}

pub(super) fn parent_unpack_u8_outputs_from_u32_staging(
    raw: Vec<u8>,
    cells: usize,
    channels: usize,
) -> Vec<u8> {
    let word_count = cells * channels;
    let words: &[u32] = bytemuck::cast_slice(&raw[..word_count * U32_OUTPUT_BYTES_PER_CELL]);
    words.iter().map(|word| (*word & 0xFF) as u8).collect()
}

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_segments::VoxelSegments;
use crate::readback::{
    pad_u8_for_gpu_storage, read_staging_f32, read_staging_f32_and_u8, split_channel_blocks,
    split_u8_channel_blocks, split_xyz_and_attribute_blocks, split_xyz_blocks,
    u8_output_staging_bytes, unpack_u8_outputs_from_u32_staging,
};
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;
const MULTI2_CHANNELS: usize = 2;
const MULTI4_CHANNELS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GatherUniform {
    cell_count: u32,
    point_count: u32,
    channel_count: u32,
    _pad: u32,
}

/// Gathers the first point's `f32` value within each voxel cell on the GPU.
pub fn gather_voxel_first_f32(
    runtime: &WgpuRuntime,
    values: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<Vec<f32>> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    if values.is_empty() {
        return Err(SpatialError::InvalidArgument(
            "cannot gather from empty value buffer".to_owned(),
        ));
    }

    let device = runtime.device();
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-values"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-indices"),
        contents: bytemuck::cast_slice(&segments.point_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-starts"),
        contents: bytemuck::cast_slice(&segments.cell_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });

    dispatch_voxel_gather_f32(
        runtime,
        &values_buffer,
        &indices_buffer,
        &starts_buffer,
        segments.len() as u32,
        segments.point_indices.len() as u32,
    )
}

/// Gathers the first point's `f32` value using GPU-resident segment buffers.
pub fn gather_voxel_first_f32_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<f32>> {
    dispatch_voxel_gather_f32(
        runtime,
        values,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        segments.cell_count(),
        segments.point_count(),
    )
}

/// Uploads `f32` values and gathers the first point per GPU-resident voxel segment.
pub fn gather_voxel_first_f32_gpu(
    runtime: &WgpuRuntime,
    values: &[f32],
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<f32>> {
    if segments.cell_count() == 0 {
        return Ok(Vec::new());
    }
    if values.len() != segments.point_count() as usize {
        return Err(SpatialError::BufferLengthMismatch {
            expected: segments.point_count() as usize,
            found: values.len(),
        });
    }

    let device = runtime.device();
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-values-upload"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    gather_voxel_first_f32_gpu_buffers(runtime, &values_buffer, segments)
}

/// Gathers multiple `f32` channels in one or more GPU dispatches.
pub fn gather_voxel_first_f32_multi_gpu(
    runtime: &WgpuRuntime,
    channels: &[&[f32]],
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<Vec<f32>>> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if segments.cell_count() == 0 {
        return Ok(vec![Vec::new(); channels.len()]);
    }

    let point_count = segments.point_count() as usize;
    for channel in channels {
        if channel.len() != point_count {
            return Err(SpatialError::BufferLengthMismatch {
                expected: point_count,
                found: channel.len(),
            });
        }
    }

    let max_channels = runtime.max_gather_channels() as usize;
    let device = runtime.device();
    let empty = empty_storage_buffer(device)?;
    let mut gathered = Vec::with_capacity(channels.len());

    for chunk in channels.chunks(max_channels) {
        let mut value_buffers = [None, None, None, None];
        for (index, channel) in chunk.iter().enumerate() {
            value_buffers[index] = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-gather-multi-values"),
                contents: bytemuck::cast_slice(channel),
                usage: wgpu::BufferUsages::STORAGE,
            }));
        }

        let value_refs: [&wgpu::Buffer; MULTI4_CHANNELS] =
            std::array::from_fn(|index| {
                if index < chunk.len() {
                    value_buffers[index].as_ref().expect("value buffer")
                } else {
                    &empty
                }
            });

        gathered.extend(dispatch_voxel_gather_multi_gpu_buffers(
            runtime,
            &value_refs,
            segments,
            chunk.len() as u32,
        )?);
    }

    Ok(gathered)
}

/// Gathers xyz and multiple f32/u8 attribute channels with one GPU submit/readback.
pub fn gather_voxel_first_xyz_and_multi_gpu(
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
        && runtime
            .pipelines()
            .voxel_gather
            .xyz_attrs4_pipeline
            .is_some();

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
            upload_recycle.push(runtime.upload_f32_storage("voxel-gather-xyz-attrs4-values", channel)?);
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

    let (flat, u8_raw) = read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    for buffer in upload_recycle {
        runtime.recycle_storage(buffer.size(), buffer);
    }
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

/// Gathers xyz and averages f32/u8 attribute channels with one GPU submit/readback.
pub fn gather_voxel_first_xyz_and_average_multi_gpu(
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
    let (flat, u8_raw) = read_staging_f32_and_u8(
        device,
        &staging_buffer,
        cells * f32_channel_count,
        u8_staging_len,
    )?;
    let u8_flat = if u8_attribute_count == 0 {
        Vec::new()
    } else {
        unpack_u8_outputs_from_u32_staging(u8_raw, cells, u8_attribute_count)
    };
    let (out_x, out_y, out_z, attributes) =
        split_xyz_and_attribute_blocks(flat, attribute_count, cells);
    Ok((
        out_x,
        out_y,
        out_z,
        attributes,
        split_u8_channel_blocks(u8_flat, u8_attribute_count, cells),
    ))
}

/// Gathers xyz coordinates of the first point within each voxel cell on the GPU.
pub fn gather_voxel_first_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if segments.cell_count() == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
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
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-xyz-staging"),
        size: (channel_len * 3) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-xyz-encoder"),
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
    encoder.copy_buffer_to_buffer(&output_y, 0, &staging_buffer, channel_len as u64, channel_len as u64);
    encoder.copy_buffer_to_buffer(
        &output_z,
        0,
        &staging_buffer,
        (channel_len * 2) as u64,
        channel_len as u64,
    );
    queue.submit(Some(encoder.finish()));

    let flat = read_staging_f32(device, &staging_buffer, cell_count as usize * 3)?;
    Ok(split_xyz_blocks(flat, cell_count as usize))
}

pub(crate) fn record_voxel_gather_xyz_and_attrs4_packed_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    attribute_buffers: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    packed_output: &wgpu::Buffer,
) -> SpatialResult<()> {
    if segments.cell_count() == 0 {
        return Ok(());
    }

    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let xyz_attrs4_pipeline = pipelines
        .voxel_gather
        .xyz_attrs4_pipeline
        .as_ref()
        .ok_or_else(|| {
            SpatialError::InvalidArgument(
                "fused xyz+4-attribute gather pipeline is unavailable on this gpu adapter".to_owned(),
            )
        })?;
    let xyz_attrs4_layout = pipelines
        .voxel_gather
        .xyz_attrs4_bind_group_layout
        .as_ref()
        .expect("xyz attrs4 layout");
    let cell_count = segments.cell_count();
    let uniform = GatherUniform {
        cell_count,
        point_count: segments.point_count(),
        channel_count: 0,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-xyz-attrs4-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-xyz-attrs4-bind-group"),
        layout: xyz_attrs4_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: z.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: attribute_buffers[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: attribute_buffers[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: attribute_buffers[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: attribute_buffers[3].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: packed_output.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-xyz-attrs4-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(xyz_attrs4_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

/// Records batched first-point f32 attribute gathers into a unified xyz staging buffer.
pub(crate) fn record_gather_f32_attribute_channels_to_staging(
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
            upload_recycle.push(runtime.upload_f32_storage("voxel-gather-xyz-attrs-values", channel)?);
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
                record_voxel_gather_multi2_pass(
                    encoder,
                    runtime,
                    &values,
                    segments,
                    &outputs,
                    2,
                )?;
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
                        upload_recycle.push(runtime.upload_f32_storage(
                            "voxel-gather-xyz-attrs-values",
                            channel,
                        )?);
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

pub(crate) fn record_voxel_gather_multi2_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI2_CHANNELS],
    segments: &GpuVoxelSegments,
    outputs: &[&wgpu::Buffer; MULTI2_CHANNELS],
    channel_count: u32,
) -> SpatialResult<()> {
    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let cell_count = segments.cell_count();
    let uniform = GatherUniform {
        cell_count,
        point_count: segments.point_count(),
        channel_count,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi-bind-group"),
        layout: &pipelines.voxel_gather.multi_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: outputs[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: outputs[1].as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-multi-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_gather.multi_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_gather_multi4_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    outputs: &[&wgpu::Buffer; MULTI4_CHANNELS],
    channel_count: u32,
) -> SpatialResult<()> {
    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let multi4_pipeline = pipelines
        .voxel_gather
        .multi4_pipeline
        .as_ref()
        .ok_or_else(|| {
            SpatialError::InvalidArgument(
                "4-channel gather pipeline is unavailable on this gpu adapter".to_owned(),
            )
        })?;
    let multi4_layout = pipelines
        .voxel_gather
        .multi4_bind_group_layout
        .as_ref()
        .expect("multi4 layout");
    let cell_count = segments.cell_count();
    let uniform = GatherUniform {
        cell_count,
        point_count: segments.point_count(),
        channel_count,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi4-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi4-bind-group"),
        layout: multi4_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: values[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: values[3].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: outputs[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: outputs[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: outputs[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: outputs[3].as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-multi4-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(multi4_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_gather_f32_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
    output_buffer: &wgpu::Buffer,
) -> SpatialResult<()> {
    if cell_count == 0 {
        return Ok(());
    }

    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count: 1,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-bind-group"),
        layout: &pipelines.voxel_gather.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: point_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_gather.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_gather_u8_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
    output_buffer: &wgpu::Buffer,
) -> SpatialResult<()> {
    if cell_count == 0 {
        return Ok(());
    }

    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count: 1,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-u8-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-u8-bind-group"),
        layout: &pipelines.voxel_gather.u8_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: point_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-u8-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_gather.u8_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_gather_xyz_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
    output_x: &wgpu::Buffer,
    output_y: &wgpu::Buffer,
    output_z: &wgpu::Buffer,
) -> SpatialResult<()> {
    if segments.cell_count() == 0 {
        return Ok(());
    }

    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count: 0,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-xyz-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-xyz-bind-group"),
        layout: &pipelines.voxel_gather.xyz_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: z.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: output_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: output_z.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("voxel-gather-xyz-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_gather.xyz_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

fn dispatch_voxel_gather_f32(
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
) -> SpatialResult<Vec<f32>> {
    if cell_count == 0 {
        return Ok(Vec::new());
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();

    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count: 1,
        _pad: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = cell_count as usize * std::mem::size_of::<f32>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-bind-group"),
        layout: &pipelines.voxel_gather.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: point_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_gather.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_len as u64);
    queue.submit(Some(encoder.finish()));

    read_staging_f32(device, &staging_buffer, cell_count as usize)
}

fn dispatch_voxel_gather_multi_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let pipelines = runtime.pipelines();
    if channel_count > 2 && pipelines.voxel_gather.multi4_pipeline.is_some() {
        dispatch_voxel_gather_multi4_gpu_buffers(runtime, values, segments, channel_count)
    } else {
        if channel_count > MULTI2_CHANNELS as u32 {
            return Err(SpatialError::InvalidArgument(format!(
                "gpu adapter supports only {} channels per gather dispatch",
                pipelines.voxel_gather.multi_max_channels
            )));
        }
        dispatch_voxel_gather_multi2_gpu_buffers(runtime, values, segments, channel_count)
    }
}

fn dispatch_voxel_gather_multi2_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();
    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let channels = channel_count as usize;

    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let mut outputs = [None, None];
    for output in outputs.iter_mut() {
        *output = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-multi-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let output_refs: [&wgpu::Buffer; MULTI2_CHANNELS] =
        std::array::from_fn(|index| outputs[index].as_ref().expect("output buffer"));

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-multi-staging"),
        size: (channel_len * channels) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi-bind-group"),
        layout: &pipelines.voxel_gather.multi_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_refs[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output_refs[1].as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-multi-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-multi-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_gather.multi_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    for (index, output) in output_refs.iter().take(channels).enumerate() {
        let offset = (channel_len * index) as u64;
        encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, offset, channel_len as u64);
    }
    queue.submit(Some(encoder.finish()));

    let flat = read_staging_f32(device, &staging_buffer, cell_count as usize * channels)?;
    Ok(split_channel_blocks(flat, channels, cell_count as usize))
}

fn dispatch_voxel_gather_multi4_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();
    let multi4_pipeline = pipelines
        .voxel_gather
        .multi4_pipeline
        .as_ref()
        .ok_or_else(|| {
            SpatialError::InvalidArgument(
                "4-channel gather pipeline is unavailable on this gpu adapter".to_owned(),
            )
        })?;
    let multi4_layout = pipelines
        .voxel_gather
        .multi4_bind_group_layout
        .as_ref()
        .expect("multi4 layout");

    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let channels = channel_count as usize;

    let uniform = GatherUniform {
        cell_count,
        point_count,
        channel_count,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-gather-multi4-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
    let mut outputs = [None, None, None, None];
    for output in outputs.iter_mut() {
        *output = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-gather-multi4-output"),
            size: channel_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let output_refs: [&wgpu::Buffer; MULTI4_CHANNELS] =
        std::array::from_fn(|index| outputs[index].as_ref().expect("output buffer"));

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-multi4-staging"),
        size: (channel_len * channels) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-gather-multi4-bind-group"),
        layout: multi4_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: values[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: values[3].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: output_refs[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: output_refs[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: output_refs[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: output_refs[3].as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-gather-multi4-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-gather-multi4-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(multi4_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    for (index, output) in output_refs.iter().take(channels).enumerate() {
        let offset = (channel_len * index) as u64;
        encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, offset, channel_len as u64);
    }
    queue.submit(Some(encoder.finish()));

    let flat = read_staging_f32(device, &staging_buffer, cell_count as usize * channels)?;
    Ok(split_channel_blocks(flat, channels, cell_count as usize))
}

fn empty_storage_buffer(device: &wgpu::Device) -> SpatialResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::{gather_voxel_first_f32, gather_voxel_first_f32_multi_gpu};
    use crate::kernels::gpu_segments::GpuVoxelSegments;
    use crate::kernels::voxel_segments::build_voxel_segments;
    use crate::runtime::WgpuRuntime;

    fn gpu_segments_from_keys(runtime: &WgpuRuntime, keys: &[(i64, i64, i64)]) -> GpuVoxelSegments {
        use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys;
        build_voxel_segments_gpu_from_keys(runtime, keys).expect("gpu segments")
    }

    #[test]
    fn gpu_first_gather_matches_cpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let values = [0.2_f32, 0.9, 10.0, 20.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = build_voxel_segments(&keys);

        let gpu = gather_voxel_first_f32(&runtime, &values, &segments).expect("gpu gather");
        assert!((gpu[0] - 0.2).abs() < 1e-5);
        assert!((gpu[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_multi_gather_matches_single_channel_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let intensity = [0.2_f32, 0.9, 10.0, 20.0];
        let classification = [1.0_f32, 2.0, 3.0, 4.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = gpu_segments_from_keys(&runtime, &keys);

        let multi = gather_voxel_first_f32_multi_gpu(
            &runtime,
            &[&intensity, &classification],
            &segments,
        )
        .expect("multi gather");

        assert!((multi[0][0] - 0.2).abs() < 1e-5);
        assert!((multi[0][1] - 10.0).abs() < 1e-5);
        assert!((multi[1][0] - 1.0).abs() < 1e-5);
        assert!((multi[1][1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_multi4_gather_handles_four_channels_when_supported() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        if runtime.max_gather_channels() < 4 {
            return;
        }

        let c0 = [0.2_f32, 0.9, 10.0, 20.0];
        let c1 = [1.0_f32, 2.0, 3.0, 4.0];
        let c2 = [5.0_f32, 6.0, 7.0, 8.0];
        let c3 = [9.0_f32, 10.0, 11.0, 12.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = gpu_segments_from_keys(&runtime, &keys);

        let multi = gather_voxel_first_f32_multi_gpu(&runtime, &[&c0, &c1, &c2, &c3], &segments)
            .expect("multi4 gather");

        assert_eq!(multi.len(), 4);
        assert!((multi[0][0] - 0.2).abs() < 1e-5);
        assert!((multi[0][1] - 10.0).abs() < 1e-5);
        assert!((multi[1][0] - 1.0).abs() < 1e-5);
        assert!((multi[1][1] - 3.0).abs() < 1e-5);
        assert!((multi[2][0] - 5.0).abs() < 1e-5);
        assert!((multi[2][1] - 7.0).abs() < 1e-5);
        assert!((multi[3][0] - 9.0).abs() < 1e-5);
        assert!((multi[3][1] - 11.0).abs() < 1e-5);
    }
}

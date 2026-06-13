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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ReduceUniform {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Averages per-point `f32` values within each voxel cell on the GPU.
pub fn reduce_voxel_average_f32(
    runtime: &WgpuRuntime,
    values: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<Vec<f32>> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    if values.is_empty() {
        return Err(SpatialError::InvalidArgument(
            "cannot reduce empty value buffer".to_owned(),
        ));
    }

    let device = runtime.device();
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-values"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-indices"),
        contents: bytemuck::cast_slice(&segments.point_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-starts"),
        contents: bytemuck::cast_slice(&segments.cell_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });

    dispatch_voxel_reduce_f32(
        runtime,
        &values_buffer,
        &indices_buffer,
        &starts_buffer,
        segments.len() as u32,
        segments.point_indices.len() as u32,
    )
}

/// Averages per-point `f32` values using GPU-resident segment buffers.
pub fn reduce_voxel_average_f32_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<Vec<f32>> {
    dispatch_voxel_reduce_f32(
        runtime,
        values,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        segments.cell_count(),
        segments.point_count(),
    )
}

/// Uploads `f32` values and averages them within GPU-resident voxel segments.
pub fn reduce_voxel_average_f32_gpu(
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
        label: Some("voxel-reduce-values-upload"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    reduce_voxel_average_f32_gpu_buffers(runtime, &values_buffer, segments)
}

/// Uploads multiple `f32` channels and averages them with one GPU submit/readback.
pub fn reduce_voxel_average_f32_multi_gpu(
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

    let device = runtime.device();
    let queue = runtime.queue();
    let cell_count = segments.cell_count();
    let cells = cell_count as usize;
    let channel_len = cells * std::mem::size_of::<f32>();
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-multi-staging"),
        size: (channel_len * channels.len()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-multi-encoder"),
    });

    for (channel_index, channel) in channels.iter().enumerate() {
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-reduce-multi-values"),
            contents: bytemuck::cast_slice(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-reduce-multi-output"),
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
            (channel_len * channel_index) as u64,
            channel_len as u64,
        );
    }

    queue.submit(Some(encoder.finish()));

    let flat = read_staging_f32(device, &staging_buffer, cells * channels.len())?;
    Ok(split_channel_blocks(flat, channels.len(), cells))
}

/// Averages xyz and multiple f32/u8 attribute channels with one GPU submit/readback.
pub fn reduce_voxel_centroids_xyz_and_average_multi_gpu(
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

/// Averages xyz and gathers the first f32/u8 attribute value per voxel with one readback.
pub fn reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
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

pub(crate) fn record_voxel_reduce_f32_pass(
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
    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-bind-group"),
        layout: &pipelines.voxel_reduce.bind_group_layout,
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
        label: Some("voxel-reduce-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_reduce.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_reduce_u8_pass(
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
    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-u8-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-u8-bind-group"),
        layout: &pipelines.voxel_reduce.u8_bind_group_layout,
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
        label: Some("voxel-reduce-u8-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_reduce.u8_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

pub(crate) fn record_voxel_reduce_xyz_pass(
    encoder: &mut wgpu::CommandEncoder,
    runtime: &WgpuRuntime,
    values_x: &wgpu::Buffer,
    values_y: &wgpu::Buffer,
    values_z: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
    output_x: &wgpu::Buffer,
    output_y: &wgpu::Buffer,
    output_z: &wgpu::Buffer,
) -> SpatialResult<()> {
    if cell_count == 0 {
        return Ok(());
    }

    let device = runtime.device();
    let pipelines = runtime.pipelines();
    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-xyz-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-xyz-bind-group"),
        layout: &pipelines.voxel_reduce.xyz_bind_group_layout,
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
                resource: cell_starts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: values_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: values_z.as_entire_binding(),
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
        label: Some("voxel-reduce-xyz-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.voxel_reduce.xyz_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    Ok(())
}

fn dispatch_voxel_reduce_f32(
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

    let uniform = ReduceUniform {
        cell_count,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_len = cell_count as usize * std::mem::size_of::<f32>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-bind-group"),
        layout: &pipelines.voxel_reduce.bind_group_layout,
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
        label: Some("voxel-reduce-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_reduce.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_len as u64);
    queue.submit(Some(encoder.finish()));

    read_staging_f32(device, &staging_buffer, cell_count as usize)
}

/// Averages xyz positions within each voxel cell on the GPU.
pub fn reduce_voxel_centroids_xyz(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    segments: &VoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::BufferLengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let out_x = reduce_voxel_average_f32(runtime, x, segments)?;
    let out_y = reduce_voxel_average_f32(runtime, y, segments)?;
    let out_z = reduce_voxel_average_f32(runtime, z, segments)?;
    Ok((out_x, out_y, out_z))
}

/// Averages xyz positions using GPU-resident buffers.
pub fn reduce_voxel_centroids_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    dispatch_voxel_reduce_xyz_f32(
        runtime,
        x,
        y,
        z,
        segments.point_indices_buffer(),
        segments.cell_starts_buffer(),
        segments.cell_count(),
        segments.point_count(),
    )
}

fn dispatch_voxel_reduce_xyz_f32(
    runtime: &WgpuRuntime,
    values_x: &wgpu::Buffer,
    values_y: &wgpu::Buffer,
    values_z: &wgpu::Buffer,
    point_indices: &wgpu::Buffer,
    cell_starts: &wgpu::Buffer,
    cell_count: u32,
    point_count: u32,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if cell_count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let channel_len = cell_count as usize * std::mem::size_of::<f32>();
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
        label: Some("voxel-reduce-xyz-staging"),
        size: (channel_len * 3) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-xyz-encoder"),
    });
    record_voxel_reduce_xyz_pass(
        &mut encoder,
        runtime,
        values_x,
        values_y,
        values_z,
        point_indices,
        cell_starts,
        cell_count,
        point_count,
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

#[cfg(test)]
mod tests {
    use super::{
        reduce_voxel_average_f32, reduce_voxel_average_f32_multi_gpu,
        reduce_voxel_centroids_xyz, reduce_voxel_centroids_xyz_and_average_multi_gpu,
        reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
    };
    use crate::kernels::voxel_segments::build_voxel_segments;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn gpu_centroid_reduction_matches_cpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = build_voxel_segments(&keys);

        let (gpu_x, gpu_y, gpu_z) =
            reduce_voxel_centroids_xyz(&runtime, &x, &y, &z, &segments).expect("gpu reduce");

        assert!((gpu_x[0] - 0.05).abs() < 1e-5);
        assert!((gpu_x[1] - 1.05).abs() < 1e-5);
        assert_eq!(gpu_y, vec![0.0, 0.0]);
        assert_eq!(gpu_z, vec![0.0, 0.0]);

        let intensity = [0.2_f32, 0.8, 10.0, 20.0];
        let gpu_i = reduce_voxel_average_f32(&runtime, &intensity, &segments).expect("gpu average");
        assert!((gpu_i[0] - 0.5).abs() < 1e-5);
        assert!((gpu_i[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_multi_reduce_matches_single_channel_reference() {
        use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys;

        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let intensity = [0.2_f32, 0.8, 10.0, 20.0];
        let classification = [1.0_f32, 2.0, 3.0, 4.0];
        let keys = vec![(0, 0, 0), (0, 0, 0), (2, 0, 0), (2, 0, 0)];
        let segments = build_voxel_segments_gpu_from_keys(&runtime, &keys).expect("gpu segments");

        let multi = reduce_voxel_average_f32_multi_gpu(
            &runtime,
            &[&intensity, &classification],
            &segments,
        )
        .expect("multi reduce");

        assert!((multi[0][0] - 0.5).abs() < 1e-5);
        assert!((multi[0][1] - 15.0).abs() < 1e-5);
        assert!((multi[1][0] - 1.5).abs() < 1e-5);
        assert!((multi[1][1] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn unified_xyz_and_attribute_readback_matches_staged_reference() {
        use crate::kernels::voxel_keys::compute_voxel_keys_gpu_buffers;
        use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys_buffer;

        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let intensity = [0.2_f32, 0.8, 10.0, 20.0];
        let positions =
            compute_voxel_keys_gpu_buffers(&runtime, &x, &y, &z, [0.0; 3], 2.0).expect("keys");
        let segments = build_voxel_segments_gpu_from_keys_buffer(
            &runtime,
            positions.keys_buffer(),
            positions.point_count(),
            4,
        )
        .expect("segments");

        let (out_x, out_y, out_z, attrs, _) = reduce_voxel_centroids_xyz_and_average_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &[&intensity],
            &[],
            &segments,
        )
        .expect("unified reduce");

        assert!((out_x[0] - 0.05).abs() < 1e-5);
        assert!((out_x[1] - 1.05).abs() < 1e-5);
        assert_eq!(out_y, vec![0.0, 0.0]);
        assert_eq!(out_z, vec![0.0, 0.0]);
        assert!((attrs[0][0] - 0.5).abs() < 1e-5);
        assert!((attrs[0][1] - 15.0).abs() < 1e-5);

        let (_, _, _, gathered, _) = reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &[&intensity],
            &[],
            &segments,
        )
        .expect("unified gather attrs");
        assert!((gathered[0][0] - 0.2).abs() < 1e-5);
        assert!((gathered[0][1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn unified_xyz_and_u8_attribute_readback_matches_reference() {
        use crate::kernels::voxel_keys::compute_voxel_keys_gpu_buffers;
        use crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys_buffer;

        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let x = [0.0_f32, 0.1, 1.0, 1.1];
        let y = [0.0_f32, 0.0, 0.0, 0.0];
        let z = [0.0_f32, 0.0, 0.0, 0.0];
        let red = [10_u8, 20, 100, 200];
        let green = [30_u8, 40, 50, 60];
        let positions =
            compute_voxel_keys_gpu_buffers(&runtime, &x, &y, &z, [0.0; 3], 2.0).expect("keys");
        let segments = build_voxel_segments_gpu_from_keys_buffer(
            &runtime,
            positions.keys_buffer(),
            positions.point_count(),
            4,
        )
        .expect("segments");

        let (_, _, _, _, u8_attrs) = reduce_voxel_centroids_xyz_and_average_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &[],
            &[&red, &green],
            &segments,
        )
        .expect("unified u8 reduce");

        assert_eq!(u8_attrs[0], vec![15, 150]);
        assert_eq!(u8_attrs[1], vec![35, 55]);

        let (_, _, _, _, gathered_u8) = reduce_voxel_centroids_xyz_and_gather_first_multi_gpu(
            &runtime,
            positions.x_buffer(),
            positions.y_buffer(),
            positions.z_buffer(),
            &[],
            &[&red, &green],
            &segments,
        )
        .expect("unified u8 gather");
        assert_eq!(gathered_u8[0], vec![10, 100]);
        assert_eq!(gathered_u8[1], vec![30, 50]);
    }
}

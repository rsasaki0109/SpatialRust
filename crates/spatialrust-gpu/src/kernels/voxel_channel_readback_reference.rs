// Complete parent calls and decoder frozen from 0982ff93fe33776a2e771f7c89d758349ef8a0f3.
use super::*;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GatherUniform {
    cell_count: u32,
    point_count: u32,
    channel_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ReduceUniform {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 256;

const MULTI2_CHANNELS: usize = 2;

const MULTI4_CHANNELS: usize = 4;

pub(super) fn parent_gather_voxel_first_f32_multi_gpu(
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
    let empty = parent_empty_storage_buffer(device)?;
    let mut gathered = Vec::with_capacity(channels.len());

    for chunk in channels.chunks(max_channels) {
        let mut value_buffers = [None, None, None, None];
        for (index, channel) in chunk.iter().enumerate() {
            value_buffers[index] =
                Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel-gather-multi-values"),
                    contents: bytemuck::cast_slice(channel),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
        }

        let value_refs: [&wgpu::Buffer; MULTI4_CHANNELS] = std::array::from_fn(|index| {
            if index < chunk.len() {
                value_buffers[index].as_ref().expect("value buffer")
            } else {
                &empty
            }
        });

        gathered.extend(parent_dispatch_voxel_gather_multi_gpu_buffers(
            runtime,
            &value_refs,
            segments,
            chunk.len() as u32,
        )?);
    }

    Ok(gathered)
}

fn parent_dispatch_voxel_gather_multi_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let pipelines = runtime.pipelines();
    if channel_count > 2 && pipelines.voxel_gather.multi4_pipeline.is_some() {
        parent_dispatch_voxel_gather_multi4_gpu_buffers(runtime, values, segments, channel_count)
    } else {
        if channel_count > MULTI2_CHANNELS as u32 {
            return Err(SpatialError::InvalidArgument(format!(
                "gpu adapter supports only {} channels per gather dispatch",
                pipelines.voxel_gather.multi_max_channels
            )));
        }
        parent_dispatch_voxel_gather_multi2_gpu_buffers(runtime, values, segments, channel_count)
    }
}

fn parent_dispatch_voxel_gather_multi2_gpu_buffers(
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

    let uniform = GatherUniform { cell_count, point_count, channel_count, _pad: 0 };
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
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_refs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: output_refs[1].as_entire_binding() },
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

    let flat = parent_read_staging_f32(device, &staging_buffer, cell_count as usize * channels)?;
    Ok(parent_split_channel_blocks(flat, channels, cell_count as usize))
}

fn parent_dispatch_voxel_gather_multi4_gpu_buffers(
    runtime: &WgpuRuntime,
    values: &[&wgpu::Buffer; MULTI4_CHANNELS],
    segments: &GpuVoxelSegments,
    channel_count: u32,
) -> SpatialResult<Vec<Vec<f32>>> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();
    let multi4_pipeline = pipelines.voxel_gather.multi4_pipeline.as_ref().ok_or_else(|| {
        SpatialError::InvalidArgument(
            "4-channel gather pipeline is unavailable on this gpu adapter".to_owned(),
        )
    })?;
    let multi4_layout =
        pipelines.voxel_gather.multi4_bind_group_layout.as_ref().expect("multi4 layout");

    let cell_count = segments.cell_count();
    let point_count = segments.point_count();
    let channels = channel_count as usize;

    let uniform = GatherUniform { cell_count, point_count, channel_count, _pad: 0 };
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
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: values[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: values[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: values[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: values[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: output_refs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: output_refs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: output_refs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: output_refs[3].as_entire_binding() },
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

    let flat = parent_read_staging_f32(device, &staging_buffer, cell_count as usize * channels)?;
    Ok(parent_split_channel_blocks(flat, channels, cell_count as usize))
}

fn parent_empty_storage_buffer(device: &wgpu::Device) -> SpatialResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-gather-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

pub(super) fn parent_reduce_voxel_average_f32_multi_gpu(
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

    let flat = parent_read_staging_f32(device, &staging_buffer, cells * channels.len())?;
    Ok(parent_split_channel_blocks(flat, channels.len(), cells))
}

pub(super) fn parent_read_staging_f32(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<f32>> {
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
    let values: Vec<f32> = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
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

// Complete public parent frozen from a876ea7b0d01c4ceb145dedff5663030b6032d3f.
use super::*;
use crate::readback::read_staging_f32_channels;
use wgpu::util::DeviceExt;
pub(super) fn baseline_reduce_voxel_average_f32_multi_gpu(
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

    read_staging_f32_channels(device, &staging_buffer, channels.len(), cells)
}

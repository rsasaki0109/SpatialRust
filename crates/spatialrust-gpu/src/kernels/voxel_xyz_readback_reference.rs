// Complete parent calls and decoder frozen from c4b3155f9d63faed8b1a60499601bc3ee0fddc51.
use super::*;

pub(super) fn parent_gather_voxel_first_xyz_gpu_buffers(
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
    queue.submit(Some(encoder.finish()));

    let flat = parent_read_staging_f32(device, &staging_buffer, cell_count as usize * 3)?;
    Ok(parent_split_xyz_blocks(flat, cell_count as usize))
}

pub(super) fn parent_reduce_voxel_centroids_xyz_gpu_buffers(
    runtime: &WgpuRuntime,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z: &wgpu::Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    parent_dispatch_voxel_reduce_xyz_f32(
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

fn parent_dispatch_voxel_reduce_xyz_f32(
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
    queue.submit(Some(encoder.finish()));

    let flat = parent_read_staging_f32(device, &staging_buffer, cell_count as usize * 3)?;
    Ok(parent_split_xyz_blocks(flat, cell_count as usize))
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

pub(super) fn parent_split_xyz_blocks(
    mut flat: Vec<f32>,
    cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let out_z = flat.split_off(cells * 2);
    let out_y = flat.split_off(cells);
    let out_x = flat;
    (out_x, out_y, out_z)
}

use super::parent_readback as metadata;
use crate::kernels::voxel_compact::{
    GpuVoxelSegments, SpatialError, SpatialResult, VoxelKeyOutput, VoxelSegments, WgpuRuntime,
};
use bytemuck::Pod;

pub(crate) fn read_segment_metadata(
    runtime: &WgpuRuntime,
    segments: &GpuVoxelSegments,
    cell_count: usize,
) -> SpatialResult<(Vec<(i64, i64, i64)>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let device = runtime.device();
    let queue = runtime.queue();
    let point_count = segments.point_count() as usize;

    let (keys, cell_starts, point_indices) =
        if let Some(result) = metadata::try_read(runtime, segments, cell_count) {
            result?
        } else {
            let key_values = read_keys(device, queue, segments.keys_buffer(), cell_count)?;
            let keys = key_values
                .iter()
                .map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz)))
                .collect();
            let cell_starts =
                read_u32_buffer(device, queue, segments.cell_starts_buffer(), cell_count)?;
            let point_indices =
                read_u32_buffer(device, queue, segments.point_indices_buffer(), point_count)?;
            (keys, cell_starts, point_indices)
        };
    let cell_counts = derive_cell_counts(&cell_starts, cell_count, point_count);

    Ok((keys, cell_starts, cell_counts, point_indices))
}

fn derive_cell_counts(cell_starts: &[u32], cell_count: usize, point_count: usize) -> Vec<u32> {
    (0..cell_count)
        .map(|cell| {
            let start = cell_starts[cell] as usize;
            let end =
                if cell + 1 == cell_count { point_count } else { cell_starts[cell + 1] as usize };
            (end - start) as u32
        })
        .collect()
}

pub(crate) fn finalize_segments_from_readback(
    keys: Vec<(i64, i64, i64)>,
    mut point_indices: Vec<u32>,
    cell_starts: Vec<u32>,
    cell_counts: Vec<u32>,
) -> VoxelSegments {
    for cell_index in 0..keys.len() {
        let start = cell_starts[cell_index] as usize;
        let end = start + cell_counts[cell_index] as usize;
        point_indices[start..end].sort_unstable();
    }

    VoxelSegments { keys, point_indices, cell_starts, cell_counts }
}

fn read_keys(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    keys_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<VoxelKeyOutput>> {
    let byte_len = len * std::mem::size_of::<VoxelKeyOutput>();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-keys-staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-keys-read-encoder"),
    });
    encoder.copy_buffer_to_buffer(keys_buffer, 0, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    read_staging_struct(device, &staging, len)
}

fn read_u32_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u32>> {
    let byte_len = len * std::mem::size_of::<u32>();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-u32-staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-u32-read-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    read_staging_u32(device, &staging, len)
}

fn read_staging_u32(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<u32>> {
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
    let values = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}

fn read_staging_struct<T: Pod>(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    len: usize,
) -> SpatialResult<Vec<T>> {
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
    let values = bytemuck::cast_slice::<u8, T>(&data)[..len].to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(values)
}

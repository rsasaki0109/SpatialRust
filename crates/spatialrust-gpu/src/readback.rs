use spatialrust_core::{SpatialError, SpatialResult};

pub(crate) fn read_staging_f32(
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

pub(crate) fn split_channel_blocks(
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

pub(crate) fn split_xyz_blocks(mut flat: Vec<f32>, cells: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let out_z = flat.split_off(cells * 2);
    let out_y = flat.split_off(cells);
    let out_x = flat;
    (out_x, out_y, out_z)
}

pub(crate) fn split_xyz_and_attribute_blocks(
    flat: Vec<f32>,
    attribute_count: usize,
    cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let blocks = split_channel_blocks(flat, 3 + attribute_count, cells);
    let mut iter = blocks.into_iter();
    let out_x = iter.next().unwrap_or_default();
    let out_y = iter.next().unwrap_or_default();
    let out_z = iter.next().unwrap_or_default();
    let attributes = iter.collect();
    (out_x, out_y, out_z, attributes)
}

pub(crate) fn read_staging_f32_and_u8(
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

pub(crate) fn split_u8_channel_blocks(
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

pub(crate) fn pad_u8_for_gpu_storage(values: &[u8]) -> Vec<u8> {
    let padded_len = values.len().div_ceil(4) * 4;
    let mut padded = values.to_vec();
    padded.resize(padded_len, 0);
    padded
}

const U32_OUTPUT_BYTES_PER_CELL: usize = std::mem::size_of::<u32>();

pub(crate) fn u8_output_staging_bytes(cells: usize, channels: usize) -> usize {
    cells * channels * U32_OUTPUT_BYTES_PER_CELL
}

pub(crate) fn unpack_u8_outputs_from_u32_staging(
    raw: Vec<u8>,
    cells: usize,
    channels: usize,
) -> Vec<u8> {
    let word_count = cells * channels;
    let words: &[u32] = bytemuck::cast_slice(&raw[..word_count * U32_OUTPUT_BYTES_PER_CELL]);
    words.iter().map(|word| (*word & 0xFF) as u8).collect()
}

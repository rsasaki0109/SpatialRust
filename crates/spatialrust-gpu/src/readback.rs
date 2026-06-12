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

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

struct StagingUnmap<'a>(&'a wgpu::Buffer);

impl Drop for StagingUnmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

pub(crate) fn read_staging_xyz(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    cells: usize,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
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

    // Declared before the view so it also unmaps after the view drops on unwind.
    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let values = &values[..cells * 3];
    Ok(crate::readback_planes::copy_owned_xyz_planes(values, cells))
}

pub(crate) fn read_staging_f32_channels(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    channels: usize,
    cells: usize,
) -> SpatialResult<Vec<Vec<f32>>> {
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

    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let values = &values[..cells * channels];
    Ok(crate::readback_planes::copy_owned_f32_planes(values, cells, channels))
}

pub(crate) fn read_staging_xyz_and_packed_attributes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    cells: usize,
    attribute_count: usize,
    u8_attribute_count: usize,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
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

    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let f32_bytes = cells * (3 + attribute_count) * std::mem::size_of::<f32>();
    let values: &[f32] = bytemuck::cast_slice(&data[..f32_bytes]);
    let u8_bytes = packed_u8_output_staging_bytes(cells, u8_attribute_count);
    let words: &[u32] = bytemuck::cast_slice(&data[f32_bytes..f32_bytes + u8_bytes]);
    let attributes =
        crate::readback_planes::copy_owned_f32_planes(&values[cells * 3..], cells, attribute_count);
    let u8_attributes =
        crate::readback_planes::copy_owned_packed_u8_planes(words, cells, u8_attribute_count);
    let (x, y, z) = crate::readback_planes::copy_owned_xyz_planes(&values[..cells * 3], cells);
    Ok((x, y, z, attributes, u8_attributes))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn read_staging_xyz_and_attributes(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    cells: usize,
    attribute_count: usize,
    u8_attribute_count: usize,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>)> {
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

    let _unmap = StagingUnmap(staging_buffer);
    let data = slice.get_mapped_range();
    let f32_bytes = cells * (3 + attribute_count) * std::mem::size_of::<f32>();
    let values: &[f32] = bytemuck::cast_slice(&data[..f32_bytes]);
    let copy_plane = |channel| values[channel * cells..(channel + 1) * cells].to_vec();
    let u8_bytes = u8_output_staging_bytes(cells, u8_attribute_count);
    let words: &[u32] = bytemuck::cast_slice(&data[f32_bytes..f32_bytes + u8_bytes]);
    let attributes = (0..attribute_count).map(|channel| copy_plane(3 + channel)).collect();
    let u8_attributes = (0..u8_attribute_count)
        .map(|channel| {
            words[channel * cells..(channel + 1) * cells]
                .iter()
                .map(|word| (*word & 0xFF) as u8)
                .collect()
        })
        .collect();
    Ok((copy_plane(0), copy_plane(1), copy_plane(2), attributes, u8_attributes))
}

pub(crate) fn pad_u8_for_gpu_storage(values: &[u8]) -> Vec<u8> {
    let padded_len = values.len().div_ceil(4) * 4;
    let mut padded = values.to_vec();
    padded.resize(padded_len, 0);
    padded
}

pub(crate) fn u8_storage_upload_contents(values: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if values.len() % 4 == 0 {
        std::borrow::Cow::Borrowed(values)
    } else {
        std::borrow::Cow::Owned(pad_u8_for_gpu_storage(values))
    }
}

const U32_OUTPUT_BYTES_PER_CELL: usize = std::mem::size_of::<u32>();

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn u8_output_staging_bytes(cells: usize, channels: usize) -> usize {
    cells * channels * U32_OUTPUT_BYTES_PER_CELL
}

pub(crate) fn packed_u8_output_staging_bytes(cells: usize, channels: usize) -> usize {
    cells.div_ceil(4) * channels * U32_OUTPUT_BYTES_PER_CELL
}

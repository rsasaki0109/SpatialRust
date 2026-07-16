use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_segments::VoxelSegments;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CompactParams {
    point_count: u32,
    scan_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

/// Compacts sorted voxel entries on the GPU using a prefix scan over segment boundaries.
#[allow(dead_code)] // public API for callers that need CPU-side `VoxelSegments`
pub fn compact_voxel_segments_from_sorted_gpu(
    runtime: &WgpuRuntime,
    entries_buffer: &wgpu::Buffer,
    point_count: u32,
) -> SpatialResult<VoxelSegments> {
    let gpu_segments = compact_voxel_segments_gpu_buffers(runtime, entries_buffer, point_count)?;
    gpu_segments.to_voxel_segments(runtime)
}

/// Compacts sorted voxel entries and keeps segment metadata on the GPU.
pub fn compact_voxel_segments_gpu_buffers(
    runtime: &WgpuRuntime,
    entries_buffer: &wgpu::Buffer,
    point_count: u32,
) -> SpatialResult<GpuVoxelSegments> {
    if point_count == 0 {
        return Ok(GpuVoxelSegments::new(
            0,
            0,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
        ));
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let buffer_len = point_count as u64;

    let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-flags"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let inclusive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-inclusive"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scan_scratch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-scan-scratch"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-keys"),
        size: buffer_len * std::mem::size_of::<VoxelKeyOutput>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let starts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-starts"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-counts"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-indices"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();
    let layout = &pipelines.voxel_compact.bind_group_layout;

    let mark_params = create_params_buffer(device, point_count, 0);
    let mark_resources = CompactBindResources {
        params: &mark_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: &inclusive_buffer,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: &scan_scratch_buffer,
    };
    let mark_bind_group = create_compact_bind_group(device, layout, &mark_resources);

    let dispatch = point_count.div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-batched-encoder"),
    });
    encoder.clear_buffer(&counts_buffer, 0, None);
    encoder.clear_buffer(&flags_buffer, 0, None);
    encoder.clear_buffer(&inclusive_buffer, 0, None);
    encoder.clear_buffer(&scan_scratch_buffer, 0, None);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-mark-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_compact.mark);
        pass.set_bind_group(0, &mark_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    let init_params = create_params_buffer(device, point_count, 0);
    let init_resources = CompactBindResources {
        params: &init_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: &inclusive_buffer,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: &scan_scratch_buffer,
    };
    let init_bind_group = create_compact_bind_group(device, layout, &init_resources);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-init-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_compact.init);
        pass.set_bind_group(0, &init_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    let mut prefix_read = &inclusive_buffer;
    let mut prefix_write = &scan_scratch_buffer;
    let mut stride = 1u32;
    while stride < point_count {
        let scan_params = create_params_buffer(device, point_count, stride);
        let scan_resources = CompactBindResources {
            params: &scan_params,
            entries: entries_buffer,
            flags: &flags_buffer,
            prefix: prefix_read,
            keys: &keys_buffer,
            starts: &starts_buffer,
            counts: &counts_buffer,
            indices: &indices_buffer,
            scan_out: prefix_write,
        };
        let scan_bind_group = create_compact_bind_group(device, layout, &scan_resources);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel-compact-scan-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.voxel_compact.scan);
            pass.set_bind_group(0, &scan_bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        std::mem::swap(&mut prefix_read, &mut prefix_write);
        stride *= 2;
    }
    queue.submit(Some(encoder.finish()));

    let write_params = create_params_buffer(device, point_count, 0);
    let write_resources = CompactBindResources {
        params: &write_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: prefix_read,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: prefix_write,
    };
    let write_bind_group = create_compact_bind_group(device, layout, &write_resources);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-write-encoder"),
    });
    encoder.clear_buffer(&counts_buffer, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-write-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_compact.write);
        pass.set_bind_group(0, &write_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    let num_cells = read_last_inclusive_count(device, queue, prefix_read, point_count)?;
    if num_cells > point_count as usize {
        return Err(SpatialError::InvalidArgument(format!(
            "GPU compact produced {num_cells} cells for {point_count} points"
        )));
    }

    Ok(GpuVoxelSegments::new(
        num_cells as u32,
        point_count,
        keys_buffer,
        indices_buffer,
        starts_buffer,
    ))
}

struct CompactBindResources<'a> {
    params: &'a wgpu::Buffer,
    entries: &'a wgpu::Buffer,
    flags: &'a wgpu::Buffer,
    prefix: &'a wgpu::Buffer,
    keys: &'a wgpu::Buffer,
    starts: &'a wgpu::Buffer,
    counts: &'a wgpu::Buffer,
    indices: &'a wgpu::Buffer,
    scan_out: &'a wgpu::Buffer,
}

fn create_compact_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &CompactBindResources<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-compact-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: resources.params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: resources.entries.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: resources.flags.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: resources.prefix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: resources.keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: resources.starts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: resources.counts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: resources.indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: resources.scan_out.as_entire_binding() },
        ],
    })
}

pub(crate) fn read_segment_metadata(
    runtime: &WgpuRuntime,
    segments: &GpuVoxelSegments,
    cell_count: usize,
) -> SpatialResult<(Vec<(i64, i64, i64)>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let device = runtime.device();
    let queue = runtime.queue();
    let point_count = segments.point_count() as usize;

    let key_values = read_keys(device, queue, segments.keys_buffer(), cell_count)?;
    let keys = key_values
        .iter()
        .map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz)))
        .collect();
    let cell_starts = read_u32_buffer(device, queue, segments.cell_starts_buffer(), cell_count)?;
    let point_indices =
        read_u32_buffer(device, queue, segments.point_indices_buffer(), point_count)?;
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

fn empty_storage_buffer(runtime: &WgpuRuntime) -> SpatialResult<wgpu::Buffer> {
    Ok(runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-empty"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

fn create_params_buffer(device: &wgpu::Device, point_count: u32, scan_stride: u32) -> wgpu::Buffer {
    let params = CompactParams { point_count, scan_stride, _pad0: 0, _pad1: 0 };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-compact-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn read_last_inclusive_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    inclusive_buffer: &wgpu::Buffer,
    point_count: u32,
) -> SpatialResult<usize> {
    let offset = ((point_count as u64).saturating_sub(1)) * std::mem::size_of::<u32>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-inclusive-staging"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-compact-inclusive-read-encoder"),
    });
    encoder.copy_buffer_to_buffer(inclusive_buffer, offset, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    let values = read_staging_u32(device, &staging, 1)?;
    Ok(values[0] as usize)
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

#[cfg(test)]
mod tests {
    #[test]
    fn inclusive_scan_reference_matches_gpu_algorithm() {
        let flags = vec![1_u32, 0, 1, 0, 1];
        let mut inclusive = flags;
        let point_count = inclusive.len() as u32;
        let mut stride = 1_u32;
        while stride < point_count {
            let source = inclusive.clone();
            for i in stride..point_count {
                inclusive[i as usize] = source[i as usize] + source[(i - stride) as usize];
            }
            stride *= 2;
        }
        assert_eq!(inclusive, vec![1, 1, 2, 2, 3]);
    }
}

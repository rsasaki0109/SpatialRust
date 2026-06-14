use bytemuck::{Pod, Zeroable};
use spatialrust_core::SpatialResult;
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_compact::compact_voxel_segments_gpu_buffers;
use crate::kernels::voxel_keys::VoxelKeyOutput;
use crate::kernels::voxel_segments::VoxelSegments;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SortParams {
    padded_count: u32,
    pair_distance: u32,
    block_width: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct VoxelSortEntry {
    ix: i32,
    iy: i32,
    iz: i32,
    point_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BuildEntriesParams {
    point_count: u32,
    padded_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Sorts per-point voxel keys on the GPU and compacts them into segments.
pub fn build_voxel_segments_gpu(
    runtime: &WgpuRuntime,
    keys: &[(i64, i64, i64)],
) -> SpatialResult<VoxelSegments> {
    let gpu_segments = build_voxel_segments_gpu_from_keys(runtime, keys)?;
    gpu_segments.to_voxel_segments(runtime)
}

/// Builds GPU-resident voxel segments from CPU-side keys.
pub fn build_voxel_segments_gpu_from_keys(
    runtime: &WgpuRuntime,
    keys: &[(i64, i64, i64)],
) -> SpatialResult<GpuVoxelSegments> {
    if keys.is_empty() {
        return empty_gpu_segments(runtime);
    }

    let point_count = keys.len();
    let padded_count = point_count.next_power_of_two();
    let mut key_outputs = vec![VoxelKeyOutput::default(); point_count];
    for (index, (ix, iy, iz)) in keys.iter().copied().enumerate() {
        key_outputs[index] = VoxelKeyOutput {
            ix: ix.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iy: iy.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            iz: iz.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            _pad: 0,
        };
    }

    let device = runtime.device();
    let keys_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-sort-keys-input"),
        contents: bytemuck::cast_slice(&key_outputs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    build_voxel_segments_gpu_from_keys_buffer(
        runtime,
        &keys_buffer,
        point_count as u32,
        padded_count as u32,
    )
}

/// Builds GPU-resident voxel segments from a GPU keys buffer.
pub fn build_voxel_segments_gpu_from_keys_buffer(
    runtime: &WgpuRuntime,
    keys_buffer: &wgpu::Buffer,
    point_count: u32,
    padded_count: u32,
) -> SpatialResult<GpuVoxelSegments> {
    if point_count == 0 {
        return empty_gpu_segments(runtime);
    }

    let entries_buffer =
        build_sort_entries_from_keys_gpu(runtime, keys_buffer, point_count, padded_count)?;
    let sorted_buffer = sort_entries_gpu(runtime, entries_buffer, padded_count)?;
    let compact_entries =
        filter_valid_sorted_entries(runtime, &sorted_buffer, padded_count, point_count)?;
    compact_voxel_segments_gpu_buffers(runtime, &compact_entries, point_count)
}

fn build_sort_entries_from_keys_gpu(
    runtime: &WgpuRuntime,
    keys_buffer: &wgpu::Buffer,
    point_count: u32,
    padded_count: u32,
) -> SpatialResult<wgpu::Buffer> {
    let device = runtime.device();
    let queue = runtime.queue();

    let entries_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-entries-build"),
        size: (padded_count as usize * std::mem::size_of::<VoxelSortEntry>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = BuildEntriesParams { point_count, padded_count, _pad0: 0, _pad1: 0 };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-sort-build-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pipelines = runtime.pipelines();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-sort-build-bind-group"),
        layout: &pipelines.voxel_sort_build.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: keys_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: entries_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-sort-build-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-sort-build-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_sort_build.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(padded_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(entries_buffer)
}

fn empty_gpu_segments(runtime: &WgpuRuntime) -> SpatialResult<GpuVoxelSegments> {
    let device = runtime.device();
    let make_empty = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-sort-empty"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    Ok(GpuVoxelSegments::new(0, 0, make_empty(), make_empty(), make_empty()))
}

fn filter_valid_sorted_entries(
    runtime: &WgpuRuntime,
    entries_buffer: &wgpu::Buffer,
    padded_count: u32,
    point_count: u32,
) -> SpatialResult<wgpu::Buffer> {
    let device = runtime.device();
    let queue = runtime.queue();
    let buffer_len = padded_count as u64;
    let output_len = (point_count as usize * std::mem::size_of::<VoxelSortEntry>()) as u64;

    let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-flags"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let inclusive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-inclusive"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scan_scratch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-scan-scratch"),
        size: buffer_len * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-output"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    queue.write_buffer(
        &flags_buffer,
        0,
        &vec![0u8; (buffer_len * std::mem::size_of::<u32>() as u64) as usize],
    );
    queue.write_buffer(
        &inclusive_buffer,
        0,
        &vec![0u8; (buffer_len * std::mem::size_of::<u32>() as u64) as usize],
    );
    queue.write_buffer(
        &scan_scratch_buffer,
        0,
        &vec![0u8; (buffer_len * std::mem::size_of::<u32>() as u64) as usize],
    );

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-params"),
        size: std::mem::size_of::<FilterParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipelines = runtime.pipelines();
    let layout = &pipelines.voxel_sort_filter.bind_group_layout;

    let mark_bind_group = create_filter_bind_group(
        device,
        layout,
        &params_buffer,
        entries_buffer,
        &flags_buffer,
        &scan_scratch_buffer,
        &inclusive_buffer,
        &output_buffer,
    );
    let init_bind_group = create_filter_bind_group(
        device,
        layout,
        &params_buffer,
        entries_buffer,
        &flags_buffer,
        &scan_scratch_buffer,
        &inclusive_buffer,
        &output_buffer,
    );

    let dispatch_padded = padded_count.div_ceil(WORKGROUP_SIZE);

    write_filter_params(queue, &params_buffer, point_count, padded_count, 0);
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-sort-filter-mark-encoder"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-sort-filter-mark-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_sort_filter.mark);
        pass.set_bind_group(0, &mark_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_padded, 1, 1);
        drop(pass);
        queue.submit(Some(encoder.finish()));
    }

    write_filter_params(queue, &params_buffer, point_count, padded_count, 0);
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-sort-filter-init-encoder"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-sort-filter-init-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_sort_filter.init);
        pass.set_bind_group(0, &init_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_padded, 1, 1);
        drop(pass);
        queue.submit(Some(encoder.finish()));
    }

    let mut scan_read = &inclusive_buffer;
    let mut scan_write = &scan_scratch_buffer;
    let mut stride = 1u32;
    while stride < padded_count {
        write_filter_params(queue, &params_buffer, point_count, padded_count, stride);
        let scan_bind_group = create_filter_bind_group(
            device,
            layout,
            &params_buffer,
            entries_buffer,
            &flags_buffer,
            scan_read,
            scan_write,
            &output_buffer,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-sort-filter-scan-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel-sort-filter-scan-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.voxel_sort_filter.scan);
            pass.set_bind_group(0, &scan_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_padded, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        std::mem::swap(&mut scan_read, &mut scan_write);
        stride *= 2;
    }

    let valid_count = read_filter_valid_count(device, queue, scan_read, padded_count)?;
    if valid_count != point_count as usize {
        return Err(spatialrust_core::SpatialError::InvalidArgument(format!(
            "expected {point_count} sorted voxel entries, found {valid_count}"
        )));
    }

    write_filter_params(queue, &params_buffer, point_count, padded_count, 0);
    let scatter_bind_group = create_filter_bind_group(
        device,
        layout,
        &params_buffer,
        entries_buffer,
        &flags_buffer,
        scan_read,
        scan_write,
        &output_buffer,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-sort-filter-scatter-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-sort-filter-scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_sort_filter.scatter);
        pass.set_bind_group(0, &scatter_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_padded, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(output_buffer)
}

fn create_filter_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params_buffer: &wgpu::Buffer,
    entries_buffer: &wgpu::Buffer,
    flags_buffer: &wgpu::Buffer,
    scan_in: &wgpu::Buffer,
    scan_out: &wgpu::Buffer,
    output_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-sort-filter-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: entries_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: flags_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scan_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scan_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FilterParams {
    point_count: u32,
    padded_count: u32,
    scan_stride: u32,
    _pad: u32,
}

fn write_filter_params(
    queue: &wgpu::Queue,
    params_buffer: &wgpu::Buffer,
    point_count: u32,
    padded_count: u32,
    scan_stride: u32,
) {
    let params = FilterParams { point_count, padded_count, scan_stride, _pad: 0 };
    queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));
}

fn read_filter_valid_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    inclusive_buffer: &wgpu::Buffer,
    padded_count: u32,
) -> SpatialResult<usize> {
    let offset = ((padded_count as u64).saturating_sub(1)) * std::mem::size_of::<u32>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-filter-count-staging"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-sort-filter-count-encoder"),
    });
    encoder.copy_buffer_to_buffer(inclusive_buffer, offset, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| {
            spatialrust_core::SpatialError::InvalidArgument(
                "failed to receive wgpu map result".to_owned(),
            )
        })?
        .map_err(|error| {
            spatialrust_core::SpatialError::InvalidArgument(format!(
                "failed to map wgpu buffer: {error}"
            ))
        })?;

    let data = slice.get_mapped_range();
    let count = bytemuck::cast_slice::<u8, u32>(&data)[0] as usize;
    drop(data);
    staging.unmap();
    Ok(count)
}

fn sort_entries_gpu(
    runtime: &WgpuRuntime,
    entries_buffer: wgpu::Buffer,
    padded_count: u32,
) -> SpatialResult<wgpu::Buffer> {
    let device = runtime.device();
    let queue = runtime.queue();
    let pipelines = runtime.pipelines();

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-params"),
        size: std::mem::size_of::<SortParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-sort-bind-group"),
        layout: &pipelines.voxel_sort.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: entries_buffer.as_entire_binding() },
        ],
    });

    let mut k = 2u32;
    while k <= padded_count {
        let mut j = k / 2;
        while j >= 1 {
            let params = SortParams { padded_count, pair_distance: j, block_width: k, _pad: 0 };
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-sort-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voxel-sort-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.voxel_sort.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(padded_count.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            j /= 2;
        }
        k *= 2;
    }

    Ok(entries_buffer)
}

#[cfg(test)]
mod tests {
    use super::build_voxel_segments_gpu;
    use crate::kernels::voxel_segments::build_voxel_segments;
    use crate::runtime::WgpuRuntime;

    #[test]
    fn gpu_segment_build_matches_cpu_reference() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let keys = vec![(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 1, 0)];
        let cpu = build_voxel_segments(&keys);
        let gpu = build_voxel_segments_gpu(&runtime, &keys).expect("gpu segments");
        assert_eq!(cpu, gpu);
    }
}

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
    compact_voxel_segments_gpu_buffers_with_profiler(runtime, entries_buffer, point_count, None)
}

pub(crate) fn compact_voxel_segments_gpu_buffers_with_profiler(
    runtime: &WgpuRuntime,
    entries_buffer: &wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuVoxelSegments> {
    if point_count == 0 {
        return Ok(GpuVoxelSegments::new(
            0,
            0,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
            empty_storage_buffer(runtime)?,
            0,
        ));
    }

    let device = runtime.device();
    let buffer_len = point_count as u64;

    let mut temporaries = crate::scratch::ScratchBuffers::new(runtime);
    let flags_buffer = temporaries.acquire("voxel-compact-flags", buffer_len * 4);
    let inclusive_buffer = temporaries.acquire("voxel-compact-inclusive", buffer_len * 4);
    let scan_scratch_buffer = temporaries.acquire("voxel-compact-unused-scan-output", 4);
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
    // Counts are derived from starts on readback; retain only the unused ABI binding.
    let counts_buffer = temporaries.acquire("voxel-compact-counts", 4);
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

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-mark-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "voxel-compact-mark",
            ),
        });
        pass.set_pipeline(&pipelines.voxel_compact.mark);
        pass.set_bind_group(0, &mark_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    super::voxel_scan::encode_inclusive_scan(
        runtime,
        &mut encoder,
        &flags_buffer,
        &inclusive_buffer,
        point_count,
        &mut profiler,
        &mut temporaries,
    );

    let write_params = create_params_buffer(device, point_count, 0);
    let write_resources = CompactBindResources {
        params: &write_params,
        entries: entries_buffer,
        flags: &flags_buffer,
        prefix: &inclusive_buffer,
        keys: &keys_buffer,
        starts: &starts_buffer,
        counts: &counts_buffer,
        indices: &indices_buffer,
        scan_out: &scan_scratch_buffer,
    };
    let write_bind_group = create_compact_bind_group(device, layout, &write_resources);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-compact-write-pass"),
            timestamp_writes: crate::profiling::timestamp_writes(
                &mut profiler,
                "voxel-compact-write",
            ),
        });
        pass.set_pipeline(&pipelines.voxel_compact.write);
        pass.set_bind_group(0, &write_bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
    }

    // Copy the scalar before releasing the prefix lease. Another thread may
    // immediately acquire it after submission, but queue order protects this copy.
    let count_staging =
        encode_last_inclusive_count(device, &mut encoder, &inclusive_buffer, point_count);
    temporaries.submit_and_recycle(encoder);
    let num_cells = read_staging_u32(device, &count_staging, 1)?[0] as usize;
    let readback_bytes = count_staging.size();
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
        readback_bytes,
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

fn encode_last_inclusive_count(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    inclusive_buffer: &wgpu::Buffer,
    point_count: u32,
) -> wgpu::Buffer {
    let offset = ((point_count as u64).saturating_sub(1)) * std::mem::size_of::<u32>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-compact-inclusive-staging"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(inclusive_buffer, offset, &staging, 0, staging.size());
    staging
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
    use super::*;
    use crate::kernels::{voxel_scan, voxel_segments::build_voxel_segments};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestEntry {
        ix: i32,
        iy: i32,
        iz: i32,
        point_index: u32,
    }

    #[test]
    fn hierarchical_scan_matches_serial_prefix_at_block_boundaries() {
        let runtime = WgpuRuntime::shared().unwrap();
        for count in [0, 1, 2, 255, 256, 257, 511, 65_535, 65_536, 65_537, 1_000_003] {
            let flags: Vec<u32> = (0..count).map(|i| u32::from((i * 37 % 113) < 47)).collect();
            let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan-test-flags"),
                contents: if flags.is_empty() { &[0; 4] } else { bytemuck::cast_slice(&flags) },
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("scan-test-output"),
                size: (count.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            let mut temporaries = crate::scratch::ScratchBuffers::new(&runtime);
            voxel_scan::encode_inclusive_scan(
                &runtime,
                &mut encoder,
                &input,
                &output,
                count as u32,
                &mut None,
                &mut temporaries,
            );
            temporaries.submit_and_recycle(encoder);
            let mut sum = 0;
            let expected: Vec<_> = flags
                .iter()
                .map(|value| {
                    sum += value;
                    sum
                })
                .collect();
            let actual = if count == 0 {
                Vec::new()
            } else {
                read_u32_buffer(runtime.device(), runtime.queue(), &output, count).unwrap()
            };
            assert_eq!(actual, expected, "element count {count}");
        }
    }

    #[test]
    fn compaction_preserves_runs_crossing_blocks_and_hierarchy_levels() {
        let runtime = WgpuRuntime::shared().unwrap();
        for (count, run) in [(1, 1), (257, 1), (65_537, 1), (65_537, 257), (65_537, 65_537)] {
            let entries: Vec<_> = (0..count)
                .map(|i| TestEntry {
                    ix: (i / run) as i32 - 100,
                    iy: -3,
                    iz: i32::MAX,
                    point_index: (count - i - 1) as u32,
                })
                .collect();
            let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("compact-test-sorted-entries"),
                contents: bytemuck::cast_slice(&entries),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let segments =
                compact_voxel_segments_gpu_buffers(&runtime, &input, count as u32).unwrap();
            assert_eq!(segments.construction_readback_bytes(), 4);
            let keys: Vec<_> = entries
                .iter()
                .map(|entry| (i64::from(entry.ix), i64::from(entry.iy), i64::from(entry.iz)))
                .collect();
            let mut expected = build_voxel_segments(&keys);
            expected.point_indices = entries.iter().map(|entry| entry.point_index).collect();
            let raw = read_u32_buffer(
                runtime.device(),
                runtime.queue(),
                segments.point_indices_buffer(),
                count,
            )
            .unwrap();
            assert!(raw == expected.point_indices, "device index order count={count}, run={run}");
            // The existing host conversion sorts indices within each cell.
            for (&start, &length) in expected.cell_starts.iter().zip(&expected.cell_counts) {
                expected.point_indices[start as usize..(start + length) as usize].sort_unstable();
            }
            let actual = segments.to_voxel_segments(&runtime).unwrap();
            assert!(
                actual.keys == expected.keys,
                "key mismatch count={count}, run={run}: {:?}",
                actual.keys.iter().zip(&expected.keys).position(|(a, b)| a != b)
            );
            for (name, a, b) in [
                ("starts", &actual.cell_starts, &expected.cell_starts),
                ("counts", &actual.cell_counts, &expected.cell_counts),
                ("indices", &actual.point_indices, &expected.point_indices),
            ] {
                let first = a.iter().zip(b).position(|(a, b)| a != b);
                assert!(a == b, "{name} mismatch count={count}, run={run}: lengths {} vs {}, first {:?}, values {:?}", a.len(), b.len(), first, first.map(|i| (a[i], b[i])));
            }
        }
    }

    fn patterned_keys(count: usize, seed: i64) -> Vec<(i64, i64, i64)> {
        (0..count)
            .map(|i| {
                let i = i as i64;
                ((i * 37 + seed) % 113 - 56, (i * 13 + seed) % 17 - 8, -seed)
            })
            .collect()
    }

    #[test]
    fn scratch_reuse_overwrites_stale_contents_and_preserves_live_outputs() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let original_keys = patterned_keys(257, 0);
        let original =
            super::super::voxel_sort::build_voxel_segments_gpu_from_keys(&runtime, &original_keys)
                .unwrap();
        for count in [0, 1, 2, 255, 256, 257, 65_537] {
            let keys = patterned_keys(count, 1);
            let actual =
                super::super::voxel_sort::build_voxel_segments_gpu_from_keys(&runtime, &keys)
                    .unwrap();
            assert_eq!(actual.construction_readback_bytes(), if count == 0 { 0 } else { 4 });
            assert!(actual.to_voxel_segments(&runtime).unwrap() == build_voxel_segments(&keys));
        }
        let warmed = runtime.scratch_buffer_pool_stats();
        for seed in 2..6 {
            let keys = patterned_keys(65_537, seed);
            let actual =
                super::super::voxel_sort::build_voxel_segments_gpu_from_keys(&runtime, &keys)
                    .unwrap();
            assert!(actual.to_voxel_segments(&runtime).unwrap() == build_voxel_segments(&keys));
        }
        let repeated = runtime.scratch_buffer_pool_stats();
        assert_eq!(repeated.allocation_count, warmed.allocation_count);
        assert!(repeated.reuse_count > warmed.reuse_count);
        assert!(repeated.cached_bytes <= repeated.budget_bytes);
        for budget in [4, 0] {
            runtime.set_scratch_buffer_pool_budget(budget);
            let keys = patterned_keys(257, 7);
            let actual =
                super::super::voxel_sort::build_voxel_segments_gpu_from_keys(&runtime, &keys)
                    .unwrap();
            assert!(actual.to_voxel_segments(&runtime).unwrap() == build_voxel_segments(&keys));
            assert!(runtime.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        runtime.clear_scratch_buffer_pool();
        assert!(
            original.to_voxel_segments(&runtime).unwrap() == build_voxel_segments(&original_keys)
        );
    }

    #[test]
    fn concurrent_segment_builds_keep_scratch_exclusive_through_submission() {
        let runtime = WgpuRuntime::new_headless().unwrap();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let runtime = &runtime;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for iteration in 0..3 {
                        let keys = patterned_keys(513, worker * 10 + iteration);
                        let segments =
                            super::super::voxel_sort::build_voxel_segments_gpu_from_keys(
                                runtime, &keys,
                            )
                            .unwrap();
                        assert_eq!(segments.construction_readback_bytes(), 4);
                        assert!(
                            segments.to_voxel_segments(runtime).unwrap()
                                == build_voxel_segments(&keys)
                        );
                    }
                });
            }
        });
        let stats = runtime.scratch_buffer_pool_stats();
        assert!(stats.reuse_count > 0);
        assert!(stats.cached_bytes <= stats.budget_bytes);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_unused_counts_tests.rs"]
mod unused_counts;

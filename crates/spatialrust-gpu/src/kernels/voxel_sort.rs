use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::kernels::gpu_segments::GpuVoxelSegments;
use crate::kernels::voxel_keys::VoxelKeyOutput;
use crate::kernels::voxel_radix::{sort_entries_radix_gpu, sort_keys_radix_gpu};
use crate::kernels::voxel_segments::VoxelSegments;
use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[path = "voxel_entry_preparation.rs"]
mod preparation;

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

/// Stably radix-sorts per-point voxel keys on the GPU and compacts them into segments.
///
/// Coordinates are clamped to the signed i32 range supported by GPU voxel keys.
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
    let point_count = u32::try_from(keys.len()).map_err(|_| {
        SpatialError::InvalidArgument("voxel keys exceed the GPU point limit".to_owned())
    })?;
    validate_sort_size(runtime, point_count)?;
    let entries = prepare_cpu_sort_entries(keys);
    let entries_buffer = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-sort-entries-input"),
        contents: bytemuck::cast_slice(&entries),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let sorted = sort_entries_radix_gpu(runtime, entries_buffer, point_count, None);
    super::voxel_compact::compact_voxel_segments_gpu_buffers_with_profiler(
        runtime,
        &sorted,
        point_count,
        None,
    )
}

// The public caller validates the u32 count and device limits before preparation.
fn prepare_cpu_sort_entries(keys: &[(i64, i64, i64)]) -> Vec<VoxelSortEntry> {
    preparation::prepare(keys)
}

/// Builds GPU-resident voxel segments from a GPU keys buffer using stable radix sorting.
///
/// `padded_count` is retained for compatibility and must be at least `point_count`.
/// Only `point_count` entries are allocated and processed; no power-of-two padding
/// or post-sort filtering is needed. Sorting performs no host readback. Segment
/// compaction retains its existing scalar cell-count synchronization.
pub fn build_voxel_segments_gpu_from_keys_buffer(
    runtime: &WgpuRuntime,
    keys_buffer: &wgpu::Buffer,
    point_count: u32,
    padded_count: u32,
) -> SpatialResult<GpuVoxelSegments> {
    build_voxel_segments_gpu_from_keys_buffer_with_profiler(
        runtime,
        keys_buffer,
        point_count,
        padded_count,
        None,
    )
}

pub(crate) fn build_voxel_segments_gpu_from_keys_buffer_with_profiler(
    runtime: &WgpuRuntime,
    keys_buffer: &wgpu::Buffer,
    point_count: u32,
    padded_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuVoxelSegments> {
    if point_count == 0 {
        return empty_gpu_segments(runtime);
    }
    validate_sort_size(runtime, point_count)?;
    if padded_count < point_count {
        return Err(SpatialError::InvalidArgument(
            "padded voxel count must be at least the point count".to_owned(),
        ));
    }
    let key_bytes = u64::from(point_count) * std::mem::size_of::<VoxelKeyOutput>() as u64;
    if keys_buffer.size() < key_bytes
        || keys_buffer.size() > u64::from(runtime.device().limits().max_storage_buffer_binding_size)
        || !keys_buffer.usage().contains(wgpu::BufferUsages::STORAGE)
    {
        return Err(SpatialError::InvalidArgument(
            "voxel keys require a storage buffer containing every point".to_owned(),
        ));
    }

    let sorted = if point_count == 1 {
        build_sort_entries_from_keys_gpu(runtime, keys_buffer, point_count, profiler.as_deref_mut())
    } else {
        sort_keys_radix_gpu(runtime, keys_buffer, point_count, profiler.as_deref_mut())
    };
    super::voxel_compact::compact_voxel_segments_gpu_buffers_with_profiler(
        runtime,
        &sorted,
        point_count,
        profiler,
    )
}

fn validate_sort_size(runtime: &WgpuRuntime, point_count: u32) -> SpatialResult<()> {
    let limits = runtime.device().limits();
    let entry_bytes = u64::from(point_count) * std::mem::size_of::<VoxelSortEntry>() as u64;
    if entry_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || entry_bytes > limits.max_buffer_size
        || point_count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
    {
        return Err(SpatialError::InvalidArgument(
            "voxel sort exceeds GPU buffer or dispatch limits".to_owned(),
        ));
    }
    Ok(())
}

fn build_sort_entries_from_keys_gpu(
    runtime: &WgpuRuntime,
    keys_buffer: &wgpu::Buffer,
    point_count: u32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> wgpu::Buffer {
    let device = runtime.device();
    let entries_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-sort-entries-build"),
        size: u64::from(point_count) * std::mem::size_of::<VoxelSortEntry>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = BuildEntriesParams { point_count, padded_count: point_count, _pad0: 0, _pad1: 0 };
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
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, "voxel-sort-build"),
        });
        pass.set_pipeline(&pipelines.voxel_sort_build.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(point_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    entries_buffer
}

fn empty_gpu_segments(runtime: &WgpuRuntime) -> SpatialResult<GpuVoxelSegments> {
    let make_empty = || {
        runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-sort-empty"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    Ok(GpuVoxelSegments::new(0, 0, make_empty(), make_empty(), make_empty(), 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::voxel_segments::build_voxel_segments;

    #[test]
    fn gpu_segment_build_matches_cpu_reference() {
        let runtime = WgpuRuntime::shared().expect("wgpu runtime");
        let keys = vec![(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 1, 0)];
        assert_eq!(build_voxel_segments(&keys), build_voxel_segments_gpu(&runtime, &keys).unwrap());
    }

    #[test]
    fn radix_matches_cpu_for_signed_extremes_and_saturated_keys() {
        let runtime = WgpuRuntime::shared().unwrap();
        let values =
            [i64::MIN, i32::MIN as i64, -65536, -1, 0, 1, 65536, i32::MAX as i64, i64::MAX];
        let mut keys = Vec::new();
        for &x in &values {
            for &y in &values {
                for &z in &values {
                    keys.push((x, y, z));
                }
            }
        }
        keys.reverse();
        let clamped: Vec<_> = keys
            .iter()
            .map(|&(x, y, z)| {
                let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64);
                (clamp(x), clamp(y), clamp(z))
            })
            .collect();
        assert_eq!(
            build_voxel_segments_gpu(&runtime, &keys).unwrap(),
            build_voxel_segments(&clamped)
        );
    }

    #[test]
    fn radix_matches_cpu_for_full_width_keys_and_reversed_input() {
        let runtime = WgpuRuntime::shared().unwrap();
        let mut state = 0x12345678u32;
        let mut word = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            i64::from(state as i32)
        };
        let mut keys = Vec::new();
        for i in 0..8193 {
            let key = (word(), word(), word());
            keys.push(key);
            if i % 3 == 0 {
                keys.push(key);
            }
        }
        assert_eq!(build_voxel_segments_gpu(&runtime, &keys).unwrap(), build_voxel_segments(&keys));
        keys.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(build_voxel_segments_gpu(&runtime, &keys).unwrap(), build_voxel_segments(&keys));
    }

    #[test]
    fn radix_handles_empty_singleton_and_workgroup_boundaries() {
        let runtime = WgpuRuntime::shared().unwrap();
        for count in [0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 65537] {
            let keys: Vec<_> = (0..count)
                .map(|i| {
                    // Small mixed-sign keys repeat across workgroups and prefix chunks.
                    (
                        ((i * 37) % 17) as i64 - 8,
                        ((i * 13) % 11) as i64 - 5,
                        ((i * 7) % 5) as i64 - 2,
                    )
                })
                .collect();
            let segments = build_voxel_segments_gpu_from_keys(&runtime, &keys).unwrap();
            assert_eq!(segments.construction_readback_bytes(), if count == 0 { 0 } else { 4 });
            assert_eq!(
                segments.to_voxel_segments(&runtime).unwrap(),
                build_voxel_segments(&keys),
                "point count {count}"
            );
        }
    }

    #[test]
    fn radix_preserves_duplicate_key_indices_on_device() {
        let runtime = WgpuRuntime::shared().unwrap();
        for mixed in [false, true] {
            let keys: Vec<_> = (0..769)
                .map(|i| VoxelKeyOutput {
                    ix: if mixed { (i * 37) % 17 - 8 } else { -1 },
                    iy: 7,
                    iz: i32::MAX,
                    _pad: 0,
                })
                .collect();
            let mut expected: Vec<u32> = (0..769).collect();
            expected.sort_by_key(|&i| {
                let key = keys[i as usize];
                (key.ix, key.iy, key.iz)
            });
            let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("radix-stability-keys"),
                contents: bytemuck::cast_slice(&keys),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let segments =
                build_voxel_segments_gpu_from_keys_buffer(&runtime, &input, 769, 1024).unwrap();
            // Inspect device indices directly: host segment finalization sorts each
            // cell and could otherwise conceal an unstable GPU permutation.
            let byte_len = 769 * 4;
            let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix-stability-readback"),
                size: byte_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                segments.point_indices_buffer(),
                0,
                &staging,
                0,
                byte_len,
            );
            runtime.queue().submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            runtime.wait_idle();
            receiver.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), expected);
            drop(mapped);
            staging.unmap();
        }
    }

    #[test]
    fn rejects_invalid_buffer_counts_and_device_limits() {
        let runtime = WgpuRuntime::shared().unwrap();
        let storage = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid-radix-input"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(build_voxel_segments_gpu_from_keys_buffer(&runtime, &storage, 2, 2).is_err());
        assert!(build_voxel_segments_gpu_from_keys_buffer(&runtime, &storage, 1, 0).is_err());
        assert!(build_voxel_segments_gpu_from_keys_buffer(&runtime, &storage, u32::MAX, u32::MAX)
            .is_err());
        let non_storage = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid-radix-usage"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        assert!(build_voxel_segments_gpu_from_keys_buffer(&runtime, &non_storage, 1, 1).is_err());
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_cpu_entries_tests.rs"]
mod cpu_entries;

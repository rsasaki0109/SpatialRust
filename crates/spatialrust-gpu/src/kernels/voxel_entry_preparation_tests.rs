use super::*;
use crate::kernels::voxel_radix::sort_entries_radix_gpu;
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys,
    build_voxel_segments_gpu_from_keys_buffer, validate_sort_size,
};
use crate::{GpuStageProfiler, GpuVoxelSegments, VoxelSegments, WgpuPowerPreference, WgpuRuntime};
use wgpu::util::DeviceExt;

#[path = "voxel_entry_preparation_reference.rs"]
mod old;

fn keys(n: usize, pattern: usize) -> Vec<(i64, i64, i64)> {
    let mut seed = 0x3141_5926u32;
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            match pattern {
                0 if i % 7 == 0 => (i64::MIN, i64::MAX, i64::from(seed as i32)),
                0 => (
                    i64::from(seed as i32),
                    i64::from(seed.rotate_left(11) as i32),
                    i64::from(seed.rotate_left(23) as i32),
                ),
                1 => (i64::MIN + i as i64, i64::MAX - i as i64, -7),
                _ => (((n - i) % 257) as i64 - 128, (i % 17) as i64 - 8, (i % 3) as i64 - 1),
            }
        })
        .collect()
}

fn clamped(keys: &[(i64, i64, i64)]) -> Vec<(i64, i64, i64)> {
    keys.iter()
        .map(|&(x, y, z)| {
            let clamp = |v: i64| v.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
            (clamp(x), clamp(y), clamp(z))
        })
        .collect()
}

fn expected(keys: &[(i64, i64, i64)]) -> VoxelSegments {
    crate::kernels::voxel_segments::build_voxel_segments(&clamped(keys))
}

fn upload(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    words
}

fn verify(r: &WgpuRuntime, output: &GpuVoxelSegments, expected: &VoxelSegments) -> usize {
    let n = expected.point_indices.len();
    let cells = expected.keys.len();
    assert_eq!(output.to_voxel_segments(r).unwrap(), *expected);
    assert_eq!(output.point_count() as usize, n);
    assert_eq!(output.cell_count() as usize, cells);
    assert_eq!(output.construction_readback_bytes(), if n == 0 { 0 } else { 4 });
    if n == 0 {
        return 0;
    }
    assert_eq!(read_words(r, output.point_indices_buffer()), expected.point_indices);
    assert_eq!(&read_words(r, output.cell_starts_buffer())[..cells], expected.cell_starts);
    let key_words: Vec<u32> =
        expected.keys.iter().flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, 0]).collect();
    assert_eq!(&read_words(r, output.keys_buffer())[..cells * 4], key_words);
    n + cells * 5
}

#[test]
fn voxel_preparation_serial_and_bounded_records_match_parent_oracle_and_capacity() {
    assert_eq!(std::mem::size_of::<VoxelSortEntry>(), 16);
    assert_eq!(std::mem::align_of::<VoxelSortEntry>(), 4);
    let values = [
        i64::MIN,
        i64::from(i32::MIN) - 1,
        i64::from(i32::MIN),
        -1,
        0,
        1,
        i64::from(i32::MAX),
        i64::from(i32::MAX) + 1,
        i64::MAX,
    ];
    let mut extremes = Vec::new();
    for &x in &values {
        for &y in &values {
            for &z in &values {
                extremes.push((x, y, z));
            }
        }
    }
    let mut fixtures = vec![extremes];
    for n in [
        0, 1, 257, 65_537, 131_071, 131_072, 131_073, 524_287, 524_288, 524_289, 1_000_003,
        1_048_577, 2_000_003,
    ] {
        for pattern in 0..3 {
            fixtures.push(keys(n, pattern));
        }
    }
    let mut checked = 0;
    for data in fixtures {
        let previous = old::serial(&data);
        let oracle = clamped(&data);
        for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
            let current = prepare_with_workers(&data, available);
            assert_eq!(current.len(), previous.len());
            assert_eq!(current.capacity(), previous.capacity());
            assert_eq!(
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&current),
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&previous)
            );
            for (i, (e, &(x, y, z))) in current.iter().zip(&oracle).enumerate() {
                assert_eq!(
                    [e.ix, e.iy, e.iz, e.point_index as i32],
                    [x as i32, y as i32, z as i32, i as i32]
                );
            }
            checked += current.len() * 4;
        }
        assert_eq!(
            bytemuck::cast_slice::<VoxelSortEntry, u32>(&prepare(&data)),
            bytemuck::cast_slice::<VoxelSortEntry, u32>(&previous)
        );
    }
    println!("VOXEL_ENTRY_PREPARATION_HOST checked {checked} words across original/saturation oracles, all bounded worker counts, exact 16-byte layout/capacity, threshold edges and partial bands.");
}

#[test]
fn voxel_preparation_worker_policy_and_disjoint_tail_guards_keep_global_indices() {
    for (len, available, expected) in [
        (0, 12, 1),
        (524_287, 12, 1),
        (524_288, 0, 1),
        (524_288, 1, 1),
        (524_288, 2, 2),
        (524_288, 3, 3),
        (524_288, 4, 4),
        (524_288, 12, 4),
        (1_000_003, usize::MAX, 4),
        (usize::MAX, usize::MAX, 4),
    ] {
        assert_eq!(workers(len, available), expected);
    }
    let poison =
        VoxelSortEntry { ix: 1_234_567, iy: -1_234_567, iz: 76_543, point_index: u32::MAX };
    let mut checked = 0;
    for n in [0, 1, 257, 131_071, 131_072, 131_073, 524_289, 1_000_003] {
        let data = keys(n, 0);
        let reference = old::serial(&data);
        for available in [1, 2, 3, 4] {
            let count = available.min(n.max(1));
            let chunk = n.max(1).div_ceil(count);
            let mut guarded = vec![poison; n + 34];
            for (band, rows) in guarded[17..17 + n].chunks_mut(chunk).enumerate() {
                let base = band * chunk;
                fill(&data[base..base + rows.len()], rows, base);
            }
            assert_eq!(
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&guarded[17..17 + n]),
                bytemuck::cast_slice::<VoxelSortEntry, u32>(&reference)
            );
            for e in guarded[..17].iter().chain(&guarded[17 + n..]) {
                assert_eq!(
                    [e.ix, e.iy, e.iz, e.point_index as i32],
                    [poison.ix, poison.iy, poison.iz, -1]
                );
            }
            checked += (n + 34) * 4;
        }
    }
    let data = keys(257, 2);
    let mut rows = vec![poison; data.len()];
    let base = u32::MAX as usize - data.len();
    fill(&data, &mut rows, base);
    assert!(rows.iter().enumerate().all(|(i, e)| e.point_index == (base + i) as u32));
    println!("VOXEL_ENTRY_PREPARATION_GUARDS checked {checked} bounded/direct-band values, dirty output guards, global high-u32 indices, caller-inclusive four-worker policy and all tail shapes.");
}

#[test]
fn voxel_preparation_complete_cpu_key_apis_keep_exact_owned_segments_and_gpu_control() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut retained = Vec::new();
    let mut checked = 0;
    let mut pairs = 0;
    for n in [0, 1, 257, 524_287, 524_288, 524_289, 1_000_003, 2_000_003] {
        for pattern in 0..3 {
            let mut data = keys(n, pattern);
            let reference = expected(&data);
            let before = old::previous_from_keys(&r, &data).unwrap();
            let after = build_voxel_segments_gpu_from_keys(&r, &data).unwrap();
            checked += verify(&r, &before, &reference) + verify(&r, &after, &reference);
            pairs += 1;
            if pattern == 0 && [524_289, 2_000_003].contains(&n) {
                data.fill((i64::MAX, i64::MIN, 0));
                retained.push((before, reference.clone()));
                retained.push((after, reference));
            }
        }
    }
    let data = keys(524_289, 0);
    assert_eq!(build_voxel_segments_gpu(&r, &data).unwrap(), old::previous(&r, &data).unwrap());
    let source_words: Vec<[i32; 4]> = data
        .iter()
        .map(|&(x, y, z)| {
            [
                x.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                y.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                z.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
                0,
            ]
        })
        .collect();
    let source = upload(&r, bytemuck::cast_slice(&source_words));
    verify(
        &r,
        &build_voxel_segments_gpu_from_keys_buffer(
            &r,
            &source,
            data.len() as u32,
            (data.len() as u32).next_power_of_two(),
        )
        .unwrap(),
        &expected(&data),
    );
    assert_eq!(read_words(&r, &source), bytemuck::cast_slice::<[i32; 4], u32>(&source_words));
    r.clear_scratch_buffer_pool();
    let _ = build_voxel_segments_gpu_from_keys(&r, &keys(257, 1)).unwrap();
    for (output, reference) in retained {
        checked += verify(&r, &output, &reference);
    }
    println!("VOXEL_ENTRY_PREPARATION_FULL checked {checked} raw index/key/start values across {pairs} complete parent/CPU API fixture pairs through two million points, explicit transfers, unchanged GPU input, retained ownership and pool reuse.");
}

fn profiled(
    r: &WgpuRuntime,
    data: &[(i64, i64, i64)],
    current: bool,
    profiler: &mut GpuStageProfiler,
) -> GpuVoxelSegments {
    if data.is_empty() {
        return super::super::empty_gpu_segments(r).unwrap();
    }
    let n = data.len() as u32;
    validate_sort_size(r, n).unwrap();
    let prepared = if current { prepare(data) } else { old::serial(data) };
    let entries = upload(r, bytemuck::cast_slice(&prepared));
    let sorted = sort_entries_radix_gpu(r, entries, n, Some(&mut *profiler));
    crate::kernels::voxel_compact::compact_voxel_segments_gpu_buffers_with_profiler(
        r,
        &sorted,
        n,
        Some(profiler),
    )
    .unwrap()
}

#[test]
fn voxel_preparation_keeps_pool_budgets_pass_profiles_preflight_and_errors() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for n in [0, 1, 2, 524_288, 524_289] {
        let data = keys(n, 2);
        let reference = expected(&data);
        let mut fresh = Vec::new();
        for current in [false, true] {
            r.clear_scratch_buffer_pool();
            let allocations = r.scratch_buffer_pool_stats().allocation_count;
            let output = if current {
                build_voxel_segments_gpu_from_keys(&r, &data)
            } else {
                old::previous_from_keys(&r, &data)
            }
            .unwrap();
            let stats = r.scratch_buffer_pool_stats();
            fresh.push((
                stats.cached_bytes,
                stats.cached_resources,
                stats.allocation_count - allocations,
            ));
            verify(&r, &output, &reference);
        }
        assert_eq!(fresh[0], fresh[1]);
        let warm = r.scratch_buffer_pool_stats().allocation_count;
        let _ = build_voxel_segments_gpu_from_keys(&r, &data).unwrap();
        assert_eq!(warm, r.scratch_buffer_pool_stats().allocation_count);
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            verify(&r, &build_voxel_segments_gpu_from_keys(&r, &data).unwrap(), &reference);
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
        if r.timestamp_queries_enabled() {
            let stages = if n == 0 {
                0
            } else {
                1 + crate::kernels::voxel_scan::scan_pass_count(n as u32)
                    + if n > 1 { 72 } else { 0 }
            };
            let mut before = GpuStageProfiler::new(&r, stages.max(1)).unwrap();
            let mut after = GpuStageProfiler::new(&r, stages.max(1)).unwrap();
            verify(&r, &profiled(&r, &data, false, &mut before), &reference);
            verify(&r, &profiled(&r, &data, true, &mut after), &reference);
            let before = before.collect(&r).unwrap();
            let after = after.collect(&r).unwrap();
            assert_eq!(before.stages.len(), stages as usize);
            assert_eq!(after.stages.len(), stages as usize);
            assert_eq!(before.device_to_host_bytes, after.device_to_host_bytes);
            assert_eq!(
                before.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                after.stages.iter().map(|s| s.stage).collect::<Vec<_>>()
            );
        }
    }
    assert!(validate_sort_size(&r, u32::MAX).is_err());
    println!("VOXEL_ENTRY_PREPARATION_RESOURCES verified identical scratch/cache requests, warm/zero-budget behavior, pass/query counts, scalar transfers and pre-allocation validation.");
}

#[test]
fn voxel_preparation_concurrent_bounded_calls_keep_outputs_private_after_host_mutation() {
    let r = WgpuRuntime::new_headless().unwrap();
    let outputs = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                scope.spawn(move || {
                    let mut data = keys(if worker == 3 { 1_000_003 } else { 524_289 }, worker % 3);
                    let reference = expected(&data);
                    let before = prepare_with_workers(&data, worker + 1);
                    assert_eq!(
                        bytemuck::cast_slice::<VoxelSortEntry, u32>(&before),
                        bytemuck::cast_slice::<VoxelSortEntry, u32>(&old::serial(&data))
                    );
                    let result = build_voxel_segments_gpu_from_keys(r, &data).unwrap();
                    data.fill((i64::MIN, i64::MAX, 0));
                    let changed = build_voxel_segments_gpu_from_keys(r, &data).unwrap();
                    verify(r, &changed, &expected(&data));
                    (result, reference)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect::<Vec<_>>()
    });
    r.clear_scratch_buffer_pool();
    for (output, reference) in outputs {
        verify(&r, &output, &reference);
    }
    println!("VOXEL_ENTRY_PREPARATION_CONCURRENT verified four simultaneous mixed-size GPU calls, forced 1/2/3/4 CPU worker shapes, host mutation and retained private outputs after pool clear.");
}

#[path = "voxel_entry_preparation_benchmark.rs"]
mod benchmark;

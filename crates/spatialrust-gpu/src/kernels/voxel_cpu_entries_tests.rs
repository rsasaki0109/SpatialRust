use super::*;
use crate::{GpuStageProfiler, WgpuPowerPreference};

#[path = "voxel_cpu_entries_reference.rs"]
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
fn cpu_voxel_entries_keep_exact_clamp_layout_capacity_and_increasing_indices() {
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
    for n in [0, 1, 2, 255, 256, 257, 65_537, 524_289, 1_000_003] {
        for pattern in 0..3 {
            fixtures.push(keys(n, pattern));
        }
    }
    let mut checked = 0;
    for data in fixtures {
        let previous = old::prepare(&data);
        let current = prepare_cpu_sort_entries(&data);
        assert_eq!(previous.len(), current.len());
        assert_eq!(previous.capacity(), current.capacity());
        let expected = clamped(&data);
        for (i, ((before, after), &(x, y, z))) in
            previous.iter().zip(&current).zip(&expected).enumerate()
        {
            assert_eq!([before.ix, before.iy, before.iz], [x as i32, y as i32, z as i32]);
            assert_eq!([after.ix, after.iy, after.iz], [before.ix, before.iy, before.iz]);
            assert_eq!(before._pad, 0);
            assert_eq!(after.point_index, i as u32);
            checked += 8;
        }
        assert_eq!(bytemuck::cast_slice::<VoxelSortEntry, u32>(&current).len(), data.len() * 4);
    }
    println!("VOXEL_CPU_ENTRIES_HOST checked {checked} parent/clamp/index words, all signed saturation combinations, exact 16-byte layout and unchanged host capacity.");
}

#[test]
fn cpu_voxel_entries_complete_apis_keep_raw_order_and_retained_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut retained = Vec::new();
    let mut checked = 0;
    let mut pairs = 0;
    for n in [0, 1, 2, 255, 256, 257, 65_537, 524_288, 524_289, 1_000_003] {
        for pattern in 0..3 {
            let mut data = keys(n, pattern);
            let reference = expected(&data);
            let before = old::previous_from_keys(&r, &data).unwrap();
            let after = build_voxel_segments_gpu_from_keys(&r, &data).unwrap();
            checked += verify(&r, &before, &reference) + verify(&r, &after, &reference);
            if n > 0 {
                data[0] = (i64::MAX, i64::MIN, i64::MAX);
            }
            retained.push((before, reference.clone()));
            retained.push((after, reference));
            pairs += 1;
        }
    }
    let data = keys(65_537, 0);
    let prepared = prepare_cpu_sort_entries(&data);
    let entries = upload(&r, bytemuck::cast_slice(&prepared));
    assert_eq!(entries.size(), data.len() as u64 * 16);
    assert_eq!(read_words(&r, &entries), bytemuck::cast_slice::<VoxelSortEntry, u32>(&prepared));
    let sorted = sort_entries_radix_gpu(&r, entries, data.len() as u32, None);
    let mut ordered: Vec<_> =
        prepared.iter().map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index]).collect();
    ordered.sort_unstable_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32, e[3]));
    assert_eq!(read_words(&r, &sorted), bytemuck::cast_slice::<[u32; 4], u32>(&ordered));
    r.clear_scratch_buffer_pool();
    let _ = build_voxel_segments_gpu_from_keys(&r, &keys(257, 1)).unwrap();
    for (output, reference) in retained {
        checked += verify(&r, &output, &reference);
    }
    assert_eq!(build_voxel_segments_gpu(&r, &data).unwrap(), old::previous(&r, &data).unwrap());
    println!("VOXEL_CPU_ENTRIES_FULL checked {checked} raw index/key/start values across {pairs} parent/CPU API pairs, uploaded entry bytes, stable prebuilt sorting and retained outputs after host mutation/pool clear/reuse.");
}

fn profiled(
    r: &WgpuRuntime,
    data: &[(i64, i64, i64)],
    current: bool,
    profiler: &mut GpuStageProfiler,
) -> GpuVoxelSegments {
    if data.is_empty() {
        return empty_gpu_segments(r).unwrap();
    }
    let n = data.len() as u32;
    validate_sort_size(r, n).unwrap();
    if current {
        let entries = upload(r, bytemuck::cast_slice(&prepare_cpu_sort_entries(data)));
        let sorted = sort_entries_radix_gpu(r, entries, n, Some(&mut *profiler));
        crate::kernels::voxel_compact::compact_voxel_segments_gpu_buffers_with_profiler(
            r,
            &sorted,
            n,
            Some(profiler),
        )
        .unwrap()
    } else {
        let source = upload(r, bytemuck::cast_slice(&old::prepare(data)));
        build_voxel_segments_gpu_from_keys_buffer_with_profiler(r, &source, n, n, Some(profiler))
            .unwrap()
    }
}

#[test]
fn cpu_voxel_entries_keep_scratch_budgets_profiles_gpu_input_route_and_preflight() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for n in [0, 1, 2, 257, 524_289] {
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
            let mut before =
                GpuStageProfiler::new(&r, (stages + u32::from(n == 1)).max(1)).unwrap();
            let mut after = GpuStageProfiler::new(&r, stages.max(1)).unwrap();
            verify(&r, &profiled(&r, &data, false, &mut before), &reference);
            verify(&r, &profiled(&r, &data, true, &mut after), &reference);
            let before = before.collect(&r).unwrap();
            let after = after.collect(&r).unwrap();
            assert_eq!(after.stages.len(), stages as usize);
            assert_eq!(before.stages.len(), after.stages.len() + usize::from(n == 1));
            assert_eq!(
                before.device_to_host_bytes,
                after.device_to_host_bytes + if n == 1 { 16 } else { 0 }
            );
            assert_eq!(
                before
                    .stages
                    .iter()
                    .map(|s| s.stage)
                    .filter(|&s| s != "voxel-sort-build")
                    .collect::<Vec<_>>(),
                after.stages.iter().map(|s| s.stage).collect::<Vec<_>>()
            );
        }
        if n > 0 {
            let values = old::prepare(&data);
            let source = upload(&r, bytemuck::cast_slice(&values));
            let output = build_voxel_segments_gpu_from_keys_buffer(
                &r,
                &source,
                n as u32,
                (n as u32).next_power_of_two(),
            )
            .unwrap();
            verify(&r, &output, &reference);
            assert_eq!(
                read_words(&r, &source),
                bytemuck::cast_slice::<VoxelKeyOutput, u32>(&values)
            );
        }
    }
    assert!(validate_sort_size(&r, u32::MAX).is_err());
    println!("VOXEL_CPU_ENTRIES_RESOURCES verified unchanged scratch requests/cache bytes/warm reuse/zero budgets, singleton removes one instrumented primitive stage/query pair, GPU-key input stays immutable and validation precedes preparation.");
}

#[test]
fn cpu_voxel_entries_concurrent_calls_keep_host_inputs_and_outputs_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    let outputs = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                scope.spawn(move || {
                    let mut data = keys(if worker == 3 { 524_289 } else { 65_537 }, worker % 3);
                    let reference = expected(&data);
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
    println!("VOXEL_CPU_ENTRIES_CONCURRENT verified four mixed small/large-prefix workers, changed host inputs, independent scratch and retained outputs.");
}

#[path = "voxel_cpu_entries_benchmark.rs"]
mod benchmark;

use super::*;
use crate::kernels::voxel_radix::sort_entries_radix_gpu_with_digits;
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys,
    build_voxel_segments_gpu_from_keys_buffer_with_profiler, prepare_cpu_sort_entries,
};
use crate::{GpuStageProfiler, GpuVoxelSegments, VoxelSegments, WgpuPowerPreference, WgpuRuntime};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

#[path = "voxel_radix_digits_sort_reference.rs"]
mod old_radix;
#[path = "voxel_radix_digits_reference.rs"]
mod parent;

pub(super) fn fixture(n: usize, pattern: usize) -> Vec<(i64, i64, i64)> {
    let mut seed = 0x3141_5926u32;
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            match pattern {
                0 => (7, -19, 23),
                1 => (((n - 1 - i) % 16) as i64, 5, 7),
                2 => (((i * 37) % 256) as i64, ((i * 13) % 64) as i64, ((i * 7) % 16) as i64),
                3 => (
                    -8192 + (i * 7 % 1024) as i64,
                    -4096 + (i * 3 % 256) as i64,
                    -16 + (i % 8) as i64,
                ),
                4 => ((i % 257) as i64 - 128, (i % 17) as i64 - 8, (i % 3) as i64 - 1),
                5 => (
                    i64::from(seed as i32),
                    i64::from(seed.rotate_left(11) as i32),
                    i64::from(seed.rotate_left(23) as i32),
                ),
                6 => (i64::MIN + i as i64, i64::MAX - i as i64, -7),
                7 if i % 7 == 0 => (i64::MIN, i64::MAX, i64::from(seed as i32)),
                7 => (
                    i64::from(seed as i32),
                    i64::from(seed.rotate_left(11) as i32),
                    i64::from(seed.rotate_left(23) as i32),
                ),
                8 => (i64::from(seed & 0x00f0_00f0), i64::from(seed & 0x000f_0000), 5),
                _ => (if i + 1 == n { 0 } else { 16 }, 3, 4),
            }
        })
        .collect()
}

pub(super) fn expected(data: &[(i64, i64, i64)]) -> VoxelSegments {
    let clamp = |v: i64| v.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
    let mut groups = BTreeMap::<_, Vec<u32>>::new();
    for (index, &(x, y, z)) in data.iter().enumerate() {
        groups.entry((clamp(x), clamp(y), clamp(z))).or_default().push(index as u32);
    }
    let mut result = VoxelSegments {
        keys: Vec::new(),
        point_indices: Vec::new(),
        cell_starts: Vec::new(),
        cell_counts: Vec::new(),
    };
    for (key, indices) in groups {
        result.keys.push(key);
        result.cell_starts.push(result.point_indices.len() as u32);
        result.cell_counts.push(indices.len() as u32);
        result.point_indices.extend(indices);
    }
    result
}

pub(super) fn upload(r: &WgpuRuntime, entries: &[VoxelSortEntry]) -> wgpu::Buffer {
    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("digit-fixture"),
        contents: if entries.is_empty() { &[0; 16] } else { bytemuck::cast_slice(entries) },
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

pub(super) fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
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

pub(super) fn verify(r: &WgpuRuntime, result: &GpuVoxelSegments, oracle: &VoxelSegments) -> usize {
    assert_eq!(result.point_count() as usize, oracle.point_indices.len());
    assert_eq!(result.cell_count() as usize, oracle.keys.len());
    assert_eq!(
        result.construction_readback_bytes(),
        if oracle.point_indices.is_empty() { 0 } else { 4 }
    );
    assert_eq!(result.to_voxel_segments(r).unwrap(), *oracle);
    if oracle.point_indices.is_empty() {
        return 0;
    }
    assert_eq!(read_words(r, result.point_indices_buffer()), oracle.point_indices);
    assert_eq!(
        &read_words(r, result.cell_starts_buffer())[..oracle.keys.len()],
        oracle.cell_starts
    );
    let keys: Vec<_> =
        oracle.keys.iter().flat_map(|&(x, y, z)| [x as u32, y as u32, z as u32, 0]).collect();
    assert_eq!(&read_words(r, result.keys_buffer())[..keys.len()], keys);
    oracle.point_indices.len() + oracle.keys.len() * 5
}

pub(super) fn required(entries: &[VoxelSortEntry]) -> u32 {
    let mut mask = 0;
    for axis in 0..3 {
        let values: Vec<_> = entries.iter().map(|e| [e.ix, e.iy, e.iz][axis]).collect();
        let groups: [Vec<_>; 2] = std::array::from_fn(|sign| {
            values.iter().copied().filter(|v| usize::from(*v < 0) == sign).collect()
        });
        for nibble in 0..8 {
            let varies = groups.iter().any(|g| {
                g.first().is_some_and(|first| {
                    g.iter().any(|v| {
                        ((*v as u32 >> (4 * nibble)) & 15) != ((*first as u32 >> (4 * nibble)) & 15)
                    })
                })
            });
            if varies || (nibble == 7 && groups.iter().all(|g| !g.is_empty())) {
                mask |= 1 << ((2 - axis) * 8 + nibble);
            }
        }
    }
    mask
}

#[test]
fn voxel_digits_plan_covers_varying_nibbles_even_parity_worker_and_probe_edges() {
    let mut cases = 0;
    for axis in 0..3 {
        for nibble in 0..8 {
            for value in 0..16u32 {
                let mut changed = [0i32; 3];
                changed[axis] = (value << (nibble * 4)) as i32;
                let data = [
                    (0, 0, 0),
                    (i64::from(changed[0]), i64::from(changed[1]), i64::from(changed[2])),
                    (0, 0, 0),
                ];
                let entries = prepare_cpu_sort_entries(&data);
                let need = required(&entries);
                let selected = plan(&entries);
                assert_eq!(selected, even(need));
                let mut sorted: Vec<_> = entries
                    .iter()
                    .map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index])
                    .collect();
                for digit in 0..24 {
                    if selected & (1 << digit) != 0 {
                        let axis = 2 - digit / 8;
                        let shift = digit % 8 * 4;
                        sorted.sort_by_key(|e| ((e[axis] ^ 0x8000_0000) >> shift) & 15);
                    }
                }
                let mut oracle: Vec<_> = entries
                    .iter()
                    .map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index])
                    .collect();
                oracle.sort_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32));
                assert_eq!(sorted, oracle);
                cases += 1;
            }
        }
    }
    for n in [0, 1, 2, 255, 256, 257, 524_287, 524_288, 524_289, 1_000_003] {
        for pattern in [0, 2, 5, 6, 8, 9] {
            let entries = prepare_cpu_sort_entries(&fixture(n, pattern));
            let need = required(&entries);
            let production = plan(&entries);
            for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
                let mask = plan_with_workers(&entries, available);
                assert_eq!(mask, production);
                assert_eq!(mask & need, need);
                assert_eq!(mask & !ALL_DIGITS, 0);
                assert_eq!(mask.count_ones() % 2, 0);
                assert_eq!(mask.count_ones(), need.count_ones() + (need.count_ones() % 2));
                assert!(workers(n, available) <= 4);
                cases += 1;
            }
        }
    }
    assert_eq!(workers(524_287, usize::MAX), 1);
    assert_eq!(workers(524_288, usize::MAX), 4);
    println!("VOXEL_DIGITS_PLAN verified {cases} masks/LSD orders, every axis/nibble/value, sign bits, 256-point probe and worker boundaries, late-only changes, saturation and even owned-output parity.");
}

#[test]
fn voxel_digits_complete_parent_cpu_and_raw_sort_outputs_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    let mut checked = 0;
    for n in [0, 1, 2, 15, 255, 256, 257, 65_537, 524_287, 524_288, 524_289, 1_000_003] {
        for pattern in 0..10 {
            let data = fixture(n, pattern);
            let entries = prepare_cpu_sort_entries(&data);
            let mask = plan(&entries);
            let oracle = expected(&data);
            let previous = parent::build_voxel_segments_gpu_from_keys(&r, &data).unwrap();
            let current = build_voxel_segments_gpu_from_keys(&r, &data).unwrap();
            checked += verify(&r, &previous, &oracle) + verify(&r, &current, &oracle);
            if n > 0 {
                let input = upload(&r, &entries);
                let owned = input.clone();
                let output = sort_entries_radix_gpu_with_digits(&r, input, n as u32, mask, None);
                assert_eq!(output, owned);
                let mut sorted: Vec<_> = entries
                    .iter()
                    .map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index])
                    .collect();
                sorted.sort_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32));
                assert_eq!(read_words(&r, &output), bytemuck::cast_slice::<[u32; 4], u32>(&sorted));
                checked += n * 4;
            }
            if n <= 257 {
                assert_eq!(parent::build_voxel_segments_gpu(&r, &data).unwrap(), oracle);
                assert_eq!(build_voxel_segments_gpu(&r, &data).unwrap(), oracle);
            }
            cases += 1;
        }
    }
    println!("VOXEL_DIGITS_OUTPUT verified {checked} raw parent/current/CPU stable words across {cases} complete fixture pairs; all clamped coordinates, signs, constant/sparse/full digits, workgroup/prefix/probe boundaries and original owned output allocation.");
}

#[test]
fn voxel_digits_profiles_resources_admission_and_gpu_key_route_remain_exact() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for n in [1, 2, 257, 524_289] {
        for pattern in [0, 1, 2, 5, 8, 9] {
            let data = fixture(n, pattern);
            let entries = prepare_cpu_sort_entries(&data);
            let mask = plan(&entries);
            let count = if n > 1 { mask.count_ones() * 3 } else { 0 };
            for budget in [crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES, 4, 0] {
                r.set_scratch_buffer_pool_budget(budget);
                r.clear_scratch_buffer_pool();
                let before = r.scratch_buffer_pool_stats();
                let input = upload(&r, &entries);
                let owned = input.clone();
                let mut profiler = GpuStageProfiler::new(&r, count.max(1)).unwrap();
                let output = sort_entries_radix_gpu_with_digits(
                    &r,
                    input,
                    n as u32,
                    mask,
                    Some(&mut profiler),
                );
                assert_eq!(output, owned);
                let after = r.scratch_buffer_pool_stats();
                let requests = after.allocation_count - before.allocation_count + after.reuse_count
                    - before.reuse_count;
                assert_eq!(requests, if count == 0 { 0 } else { 4 });
                let report = profiler.collect(&r).unwrap();
                assert_eq!(report.stages.len(), count as usize);
                assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
                for stages in report.stages.chunks_exact(3) {
                    assert_eq!(
                        stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                        ["voxel-radix-histogram", "voxel-radix-prefix", "voxel-radix-scatter"]
                    );
                }
                assert!(after.cached_bytes <= budget);
            }
        }
        let data = fixture(n, 0);
        let entries = prepare_cpu_sort_entries(&data);
        let input = upload(&r, &entries);
        let capacity =
            1 + crate::kernels::voxel_scan::scan_pass_count(n as u32) + if n == 1 { 1 } else { 72 };
        let mut reports = Vec::new();
        for current in [false, true] {
            let mut profiler = GpuStageProfiler::new(&r, capacity).unwrap();
            let output = if current {
                build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &r,
                    &input,
                    n as u32,
                    n as u32,
                    Some(&mut profiler),
                )
            } else {
                parent::build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                    &r,
                    &input,
                    n as u32,
                    n as u32,
                    Some(&mut profiler),
                )
            }
            .unwrap();
            verify(&r, &output, &expected(&data));
            reports.push(profiler.collect(&r).unwrap());
        }
        assert_eq!(
            reports[0].stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            reports[1].stages.iter().map(|s| s.stage).collect::<Vec<_>>()
        );
        assert_eq!(reports[0].device_to_host_bytes, reports[1].device_to_host_bytes);
        assert_eq!(read_words(&r, &input), bytemuck::cast_slice::<VoxelSortEntry, u32>(&entries));
        for (count, padded) in [(u32::MAX, u32::MAX), (n as u32, 0), (n as u32 + 1, n as u32 + 1)] {
            let previous =
                parent::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
                    .err()
                    .unwrap()
                    .to_string();
            let current =
                crate::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
                    .err()
                    .unwrap()
                    .to_string();
            assert_eq!(previous, current);
        }
    }
    println!("VOXEL_DIGITS_RESOURCES verified exact three-pass/query groups per selected digit, zero radix requests for constant keys, four scratch requests otherwise, default/tiny/zero budgets, untouched GPU-key passes/input and original admission errors.");
}

#[test]
fn voxel_digits_concurrent_results_survive_input_mutation_clear_and_changed_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let mut data: Vec<_> = [0, 2, 5, 9].into_iter().map(|p| fixture(65_537, p)).collect();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = data
            .iter()
            .map(|input| {
                let r = &r;
                let gate = &gate;
                scope.spawn(move || {
                    let oracle = expected(input);
                    gate.wait();
                    (build_voxel_segments_gpu_from_keys(r, input).unwrap(), oracle)
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    for input in &mut data {
        input.fill((i64::MAX, i64::MIN, 0));
    }
    r.clear_scratch_buffer_pool();
    for pattern in [1, 6, 8, 3] {
        let changed = fixture(65_537, pattern);
        let output = build_voxel_segments_gpu_from_keys(&r, &changed).unwrap();
        verify(&r, &output, &expected(&changed));
    }
    for (output, oracle) in outputs {
        verify(&r, &output, &oracle);
    }
    println!("VOXEL_DIGITS_CONCURRENT verified simultaneous zero/sparse/full/late-variable calls, immutable shared inputs, independently owned segments after host mutation, pool clears and changed-input reuse.");
}

#[path = "voxel_radix_digits_benchmark.rs"]
mod benchmark;

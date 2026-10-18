use super::tests::{expected, fixture, read_words, upload, verify};
use super::*;
use crate::kernels::voxel_radix::sort_entries_radix_gpu_with_digits;
use crate::kernels::voxel_sort::{
    build_voxel_segments_gpu, build_voxel_segments_gpu_from_keys, prepare_cpu_sort_entries,
};
use crate::{GpuStageProfiler, WgpuPowerPreference, WgpuRuntime};

#[path = "voxel_sign_digits_reference.rs"]
mod old_digits;
#[path = "voxel_sign_digits_build_reference.rs"]
mod parent;

fn signed_fixture(n: usize, pattern: usize) -> Vec<(i64, i64, i64)> {
    match pattern {
        0 => return fixture(n, 4),
        6 => return fixture(n, 5),
        7 => return fixture(n, 0),
        8 => return fixture(n, 2),
        _ => {}
    }
    let mut seed = 0x2718_2818u32;
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let values: [i64; 3] = std::array::from_fn(|axis| {
                let bits = seed.rotate_left(axis as u32 * 11);
                match pattern {
                    1 => {
                        i64::from(bits & 255)
                            + if bits >> 31 == 0 { 0x5678_9000 } else { -0x1234_5000 }
                    }
                    2 => {
                        if axis == 0 {
                            if i + 1 == n {
                                -19
                            } else {
                                7
                            }
                        } else {
                            0
                        }
                    }
                    3 => {
                        let negative = ((i / n.div_ceil(4).max(1)) + axis) % 2 == 1;
                        i64::from(bits & 15) + if negative { -4096 } else { 8192 }
                    }
                    4 => {
                        if (i + axis) % 2 == 0 {
                            i64::MIN
                        } else {
                            i64::MAX
                        }
                    }
                    5 => {
                        i64::from(bits & 0x00f0_00f0)
                            - if bits >> 31 == 1 { 0x1000_0000 } else { 0 }
                    }
                    10..=18 => {
                        let varying = 2 * (pattern - 10) + 3;
                        let nibbles = varying / 3 + usize::from(axis < varying % 3);
                        let half = 1u32 << (4 * nibbles - 1);
                        i64::from(bits & (half - 1))
                            - if bits >> 31 == 1 { i64::from(half) } else { 0 }
                    }
                    _ => panic!("unknown sign fixture"),
                }
            });
            (values[0], values[1], values[2])
        })
        .collect()
}

fn check_order(entries: &[VoxelSortEntry], selected: u32) {
    let mut sorted: Vec<_> =
        entries.iter().map(|e| [e.ix as u32, e.iy as u32, e.iz as u32, e.point_index]).collect();
    let mut oracle = sorted.clone();
    for digit in 0..24 {
        if selected & (1 << digit) != 0 {
            let axis = 2 - digit / 8;
            let shift = digit % 8 * 4;
            sorted.sort_by_key(|e| ((e[axis] ^ 0x8000_0000) >> shift) & 15);
        }
    }
    oracle.sort_by_key(|e| (e[0] as i32, e[1] as i32, e[2] as i32));
    assert_eq!(sorted, oracle);
}

#[test]
fn voxel_sign_digits_signed_lsd_class_presence_and_worker_merges_are_exact() {
    let mut cases = 0;
    let mut seed = 0x1618_0339u32;
    for width in 1..=32 {
        for round in 0..64 {
            let data: Vec<_> = (0..(round + 1))
                .map(|_| {
                    let v: [i64; 3] = std::array::from_fn(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        i64::from((seed as i32) >> (32 - width))
                    });
                    (v[0], v[1], v[2])
                })
                .collect();
            let entries = prepare_cpu_sort_entries(&data);
            let selected = plan(&entries);
            assert_eq!(selected, even(super::tests::required(&entries)));
            check_order(&entries, selected);
            cases += 1;
        }
    }
    for n in [0, 1, 2, 255, 256, 257, 524_287, 524_288, 524_289, 1_000_003] {
        for pattern in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 18] {
            let entries = prepare_cpu_sort_entries(&signed_fixture(n, pattern));
            let selected = plan(&entries);
            let required = super::tests::required(&entries);
            assert_eq!(selected, even(required));
            for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
                assert_eq!(plan_with_workers(&entries, available), selected);
                assert!(workers(n, available) <= 4);
                cases += 1;
            }
            if n <= 257 {
                check_order(&entries, selected);
            }
        }
    }
    for pattern in 10..=18 {
        let entries = prepare_cpu_sort_entries(&signed_fixture(65_537, pattern));
        assert_eq!(plan(&entries).count_ones(), (6 + 2 * (pattern - 10)) as u32);
    }
    assert_eq!(std::mem::size_of::<Summary>(), 48);
    println!("VOXEL_SIGN_DIGITS_PLAN verified {cases} independent mask/order/worker cases; every signed width, absent classes, late opposite signs, isolated worker signs, saturation, 6..22 actual digits and 48-byte summaries.");
}

#[test]
fn voxel_sign_digits_complete_parent_cpu_and_owned_raw_results_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    let mut checked = 0;
    let mut scenarios = Vec::new();
    for n in [0, 1, 2, 257, 65_537, 524_289, 1_000_003] {
        for pattern in [0, 1, 2, 3, 4, 5, 6, 8] {
            scenarios.push((n, pattern));
        }
    }
    for n in [257, 65_537] {
        for pattern in 10..=18 {
            scenarios.push((n, pattern));
        }
    }
    for (n, pattern) in scenarios {
        let data = signed_fixture(n, pattern);
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
    println!("VOXEL_SIGN_DIGITS_OUTPUT verified {checked} raw parent/current/CPU words across {cases} complete fixture pairs; stable signed lexicographic order, saturation, ties, source ownership and 6..22 actual digits.");
}

#[test]
fn voxel_sign_digits_profiles_pools_and_device_key_contracts_are_exact() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    for n in [2, 257, 65_537, 524_289] {
        for pattern in [0, 1, 2, 4, 6, 8, 10, 14, 18] {
            let entries = prepare_cpu_sort_entries(&signed_fixture(n, pattern));
            let mask = plan(&entries);
            let count = 3 * mask.count_ones();
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
                assert_eq!(
                    after.allocation_count - before.allocation_count + after.reuse_count
                        - before.reuse_count,
                    if count == 0 { 0 } else { 4 }
                );
                assert!(after.cached_bytes <= budget);
                let report = profiler.collect(&r).unwrap();
                assert_eq!(report.stages.len(), count as usize);
                assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
                for stages in report.stages.chunks_exact(3) {
                    assert_eq!(
                        stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                        ["voxel-radix-histogram", "voxel-radix-prefix", "voxel-radix-scatter"]
                    );
                }
            }
        }
    }
    let data = signed_fixture(257, 4);
    let input = upload(&r, &prepare_cpu_sort_entries(&data));
    let before = read_words(&r, &input);
    let mut reports = Vec::new();
    for current in [false, true] {
        let mut profiler = GpuStageProfiler::new(&r, 128).unwrap();
        let result = if current {
            crate::kernels::voxel_sort::build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                &r,
                &input,
                257,
                257,
                Some(&mut profiler),
            )
        } else {
            parent::build_voxel_segments_gpu_from_keys_buffer_with_profiler(
                &r,
                &input,
                257,
                257,
                Some(&mut profiler),
            )
        }
        .unwrap();
        verify(&r, &result, &expected(&data));
        reports.push(profiler.collect(&r).unwrap());
    }
    assert_eq!(
        reports[0].stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        reports[1].stages.iter().map(|s| s.stage).collect::<Vec<_>>()
    );
    assert_eq!(reports[0].device_to_host_bytes, reports[1].device_to_host_bytes);
    assert_eq!(read_words(&r, &input), before);
    for (count, padded) in [(u32::MAX, u32::MAX), (257, 0), (258, 258)] {
        let previous = parent::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
            .err()
            .unwrap()
            .to_string();
        let current = crate::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(previous, current);
    }
    println!("VOXEL_SIGN_DIGITS_RESOURCES verified three stages and 48 query bytes per digit, original owned output, default/tiny/zero pool budgets, four nonzero-sort requests and unchanged GPU-key passes/input.");
}

#[test]
fn voxel_sign_digits_concurrent_results_survive_mutation_clear_and_changed_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let mut inputs: Vec<_> = [0, 2, 4, 6].into_iter().map(|p| signed_fixture(65_537, p)).collect();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = inputs
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
    for input in &mut inputs {
        input.fill((i64::MAX, i64::MIN, 0));
    }
    r.clear_scratch_buffer_pool();
    for pattern in [1, 3, 8, 18] {
        let data = signed_fixture(65_537, pattern);
        verify(&r, &build_voxel_segments_gpu_from_keys(&r, &data).unwrap(), &expected(&data));
    }
    for (output, oracle) in outputs {
        verify(&r, &output, &oracle);
    }
    println!("VOXEL_SIGN_DIGITS_CONCURRENT verified independent mixed-sign/late-sign/saturated/full-width results under simultaneous builds, host mutation, pool clear and changed reuse.");
}

#[path = "voxel_sign_digits_benchmark.rs"]
mod benchmark;

#[path = "voxel_digit_probe_tests.rs"]
mod probe;

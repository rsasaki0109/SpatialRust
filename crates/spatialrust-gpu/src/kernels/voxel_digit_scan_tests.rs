use super::*;
use crate::kernels::voxel_radix::sort_entries_radix_gpu_with_digits;

#[path = "voxel_digit_scan_reference.rs"]
mod old_digits;
#[path = "voxel_digit_scan_build_reference.rs"]
mod parent;

fn scan_fixture(n: usize, pattern: usize) -> Vec<(i64, i64, i64)> {
    if pattern <= 28 {
        return super::probe_fixture(n, pattern);
    }
    let mut rows = super::probe_fixture(n, if pattern == 32 { 8 } else { 5 });
    if n != 0 {
        let at = match pattern {
            29 | 32 => n / 2,
            30 => 257.min(n - 1),
            31 => (3 * n.div_ceil(4)).min(n - 1),
            _ => panic!("unknown scan fixture"),
        };
        if pattern == 32 {
            rows[at] = (-1, -2, -3);
            rows[n - 1] = (i64::MIN, i64::MAX, i64::MIN);
        } else {
            rows[at].0 ^= 0x0000_0f00;
        }
    }
    rows
}

#[test]
fn voxel_scan_masks_match_complete_parent_class_oracle_and_worker_edges() {
    let mut cases = 0;
    for n in [
        0, 1, 2, 255, 256, 257, 16_383, 16_384, 16_385, 65_537, 524_287, 524_288, 524_289,
        1_000_003,
    ] {
        for p in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32]
        {
            let entries = prepare_cpu_sort_entries(&scan_fixture(n, p));
            let selected = plan(&entries);
            assert_eq!(selected, old_digits::plan(&entries));
            assert_eq!(selected, even(super::super::tests::required(&entries)));
            for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
                assert_eq!(selected, plan_with_workers(&entries, available));
                if !entries.is_empty() {
                    let prefix = summarize(&entries[..entries.len().min(PROBE_POINTS)]);
                    assert_eq!(selected, refine_with_workers(&entries, available, digits(prefix)));
                }
                assert!(workers(n, available) <= 4);
                cases += 1;
            }
            if n <= 257 {
                check_order(&entries, selected);
            }
        }
    }
    let mut seed = 0xa5a5_1234u32;
    for width in 1..=32 {
        for round in 0..32 {
            let data: Vec<_> = (0..(257 + round))
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
            assert_eq!(selected, old_digits::plan(&entries));
            assert_eq!(selected, even(super::super::tests::required(&entries)));
            check_order(&entries, selected);
            cases += 1;
        }
    }
    // Both signs may share constant middle digits; the global summary can be
    // exact even though the original one-sign admission cannot apply.
    for p in [5, 21, 22] {
        let entries = prepare_cpu_sort_entries(&scan_fixture(65_537, p));
        let summary = summarize(&entries[..256]);
        assert!(summary.iter().any(|classes| classes.iter().all(|c| c.seen())));
        let global = summary.map(|c| (c[0].any | c[1].any) ^ (c[0].all & c[1].all));
        assert_eq!(varying_digits(global), digits(summary));
        assert_eq!(even(digits(summary)), plan(&entries));
    }
    println!("VOXEL_SCAN_PLAN verified {cases} parent/oracle/worker masks and stable signed LSD orders; empty classes, every signed width, exact raw-bit membership, 256-point probe and four-worker/block boundaries, saturated probes and early/middle/late fallback.");
}

#[test]
fn voxel_scan_complete_outputs_profiles_admission_and_concurrent_ownership_match() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let mut checked = 0;
    let mut cases = 0;
    for n in [0, 1, 2, 257, 16_385, 65_537, 524_289, 1_000_003] {
        for p in [5, 8, 23, 24, 25, 29, 30, 31, 32] {
            let data = scan_fixture(n, p);
            let oracle = expected(&data);
            let entries = prepare_cpu_sort_entries(&data);
            let mask = plan(&entries);
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
    for p in [5, 23, 24, 29, 32] {
        let entries = prepare_cpu_sort_entries(&scan_fixture(65_537, p));
        let mask = plan(&entries);
        for budget in [crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES, 4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            r.clear_scratch_buffer_pool();
            let before = r.scratch_buffer_pool_stats();
            let mut profiler = GpuStageProfiler::new(&r, 72).unwrap();
            let input = upload(&r, &entries);
            let owned = input.clone();
            let output =
                sort_entries_radix_gpu_with_digits(&r, input, 65_537, mask, Some(&mut profiler));
            assert_eq!(output, owned);
            let after = r.scratch_buffer_pool_stats();
            assert_eq!(
                after.allocation_count - before.allocation_count + after.reuse_count
                    - before.reuse_count,
                4
            );
            assert!(after.cached_bytes <= budget);
            let report = profiler.collect(&r).unwrap();
            assert_eq!(report.stages.len(), mask.count_ones() as usize * 3);
            assert_eq!(report.device_to_host_bytes, u64::from(mask.count_ones()) * 48);
        }
    }
    let data = scan_fixture(257, 5);
    let entries = prepare_cpu_sort_entries(&data);
    let input = upload(&r, &entries);
    let before = read_words(&r, &input);
    let mut reports = Vec::new();
    for current in [false, true] {
        let mut profiler = GpuStageProfiler::new(&r, 128).unwrap();
        let out = if current {
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
        verify(&r, &out, &expected(&data));
        reports.push(profiler.collect(&r).unwrap());
    }
    assert_eq!(
        reports[0].stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        reports[1].stages.iter().map(|s| s.stage).collect::<Vec<_>>()
    );
    assert_eq!(reports[0].device_to_host_bytes, reports[1].device_to_host_bytes);
    assert_eq!(read_words(&r, &input), before);
    for (count, padded) in [(u32::MAX, u32::MAX), (257, 0), (258, 258)] {
        let old = parent::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
            .err()
            .unwrap()
            .to_string();
        let new = crate::build_voxel_segments_gpu_from_keys_buffer(&r, &input, count, padded)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(old, new);
    }
    let mut data: Vec<_> = [23, 24, 29, 32].into_iter().map(|p| scan_fixture(65_537, p)).collect();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = data
            .iter()
            .map(|d| {
                let r = &r;
                let gate = &gate;
                scope.spawn(move || {
                    let oracle = expected(d);
                    gate.wait();
                    (build_voxel_segments_gpu_from_keys(r, d).unwrap(), oracle)
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    for d in &mut data {
        d.fill((i64::MAX, i64::MIN, 0));
    }
    r.clear_scratch_buffer_pool();
    let changed = scan_fixture(65_537, 21);
    verify(&r, &build_voxel_segments_gpu_from_keys(&r, &changed).unwrap(), &expected(&changed));
    for (out, oracle) in outputs {
        verify(&r, &out, &oracle);
    }
    println!("VOXEL_SCAN_OUTPUT verified {checked} parent/current/CPU raw words across {cases} complete fixture pairs; unchanged masks/passes/queries/requests, ownership, budgets, device-key input/admission, concurrent builds, mutation and changed reuse.");
}

#[path = "voxel_digit_scan_benchmark.rs"]
mod benchmark;

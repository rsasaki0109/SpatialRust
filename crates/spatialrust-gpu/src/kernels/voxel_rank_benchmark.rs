use super::*;

#[test]
#[ignore = "explicit native rank-cache comparison with completed sort timings"]
fn paired_previous_word_rank_sorts() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    let compare_shortcut =
        std::env::var("SPATIALRUST_RADIX_RANK_COMPARE").as_deref() == Ok("cache-only");
    let cache_only = include_str!("../shaders/voxel_radix.wgsl");
    let shortcut = source_pipelines(
        &runtime,
        true,
        include_str!("../../tests/fixtures/voxel_radix_lane_shortcut.wgsl"),
    );
    let small = if compare_shortcut {
        source_pipelines(&runtime, false, cache_only)
    } else {
        previous_pipelines(&runtime, false)
    };
    let large = if compare_shortcut {
        source_pipelines(&runtime, true, cache_only)
    } else {
        previous_pipelines(&runtime, true)
    };
    eprintln!(
        "comparison={}",
        if compare_shortcut {
            "cache-only versus cache-with-shortcut"
        } else {
            "parent versus current"
        }
    );
    eprintln!("backend={}; device_type={}; 3 warmups + 20 alternating pairs; full sort API/setup/encoding/submit/device wait, excluding upload/pipeline cold setup/query collection/output checks/recycle", runtime.adapter_info().backend, runtime.adapter_info().device_type);
    let sizes: &[usize] = if compare_shortcut {
        &[524289, 1000003, 2000003]
    } else {
        &[65537, 524288, 524289, 1000003, 2000003]
    };
    for &count in sizes {
        for pattern in 0..3 {
            let input = inputs(count, pattern);
            let expected = sorted(&input);
            let old = if count <= 524288 { &small } else { &large };
            let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut stages: [Vec<[f64; 3]>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..23 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let mut profiler = GpuStageProfiler::new(&runtime, 72).unwrap();
                    let entries = buffer(&runtime, bytemuck::cast_slice(&input));
                    runtime.wait_idle();
                    let start = std::time::Instant::now();
                    let output = if mode == 0 {
                        previous_sort(
                            &runtime,
                            entries,
                            count as u32,
                            Some(&mut profiler),
                            [&old[0], &old[1], &old[2]],
                        )
                    } else if compare_shortcut {
                        previous_sort(
                            &runtime,
                            entries,
                            count as u32,
                            Some(&mut profiler),
                            [&shortcut[0], &shortcut[1], &shortcut[2]],
                        )
                    } else {
                        sort_entries_radix_gpu(&runtime, entries, count as u32, Some(&mut profiler))
                    };
                    runtime.wait_idle();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let report = profiler.collect(&runtime).unwrap();
                    assert_eq!(report.stages.len(), 72);
                    assert_eq!(report.device_to_host_bytes, 1152);
                    assert_eq!(
                        read_words(&runtime, &output),
                        bytemuck::cast_slice::<Entry, u32>(&expected)
                    );
                    if round >= 3 {
                        wall[mode].push(elapsed);
                        let mut total = [0.0; 3];
                        for (i, stage) in report.stages.iter().enumerate() {
                            total[i % 3] += stage.elapsed_ns / 1000.0;
                        }
                        stages[mode].push(total);
                    }
                    // Keep ownership/drop outside the completion timer.
                    drop(output);
                }
            }
            eprintln!("WORD_RANK {{\"count\":{count},\"pattern\":{pattern},\"previous_ms\":{:?},\"current_ms\":{:?},\"previous_stage_us\":{:?},\"current_stage_us\":{:?}}}", wall[0], wall[1], stages[0], stages[1]);
        }
    }
}

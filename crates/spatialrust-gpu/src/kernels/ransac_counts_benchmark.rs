use super::*;
use std::{hint::black_box, time::Instant};

fn measure(
    r: &WgpuRuntime,
    cache: &RansacPlanePipelines,
    prior: &old::ParentPipelines,
    parent: &wgpu::ComputePipeline,
    n: u32,
    h: u32,
) {
    let (axes, hypotheses) = fixture(n, h);
    let p = plan(n as usize, h as usize, &r.device().limits()).unwrap();
    let threshold = 0.25;
    let expected =
        old::previous(prior, r, &axes[0], &axes[1], &axes[2], &hypotheses, threshold).unwrap();
    let expected_words: &[u32] = bytemuck::cast_slice(&expected);
    let mut calls: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let current = (round + offset) % 2 == 1;
            let start = Instant::now();
            if current {
                let output = score_ransac_plane_hypotheses_gpu(
                    black_box(r),
                    black_box(&axes[0]),
                    black_box(&axes[1]),
                    black_box(&axes[2]),
                    black_box(&hypotheses),
                    threshold,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(bytemuck::cast_slice::<_, u32>(&output), expected_words);
                check_axes(&axes, threshold, &output);
                if round >= 3 {
                    calls[1].push(elapsed);
                }
            } else {
                let output = old::previous(
                    prior,
                    black_box(r),
                    black_box(&axes[0]),
                    black_box(&axes[1]),
                    black_box(&axes[2]),
                    black_box(&hypotheses),
                    threshold,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(bytemuck::cast_slice::<_, u32>(&output), expected_words);
                if round >= 3 {
                    calls[0].push(elapsed);
                }
            }
        }
    }
    let current = cache.pipeline(r.device(), p.parallel);
    let previous = if p.parallel { parent } else { current };
    let pipelines = [previous, current];
    let (outputs, groups) = bindings(r, cache, &p, &axes, &hypotheses, threshold);
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = crate::GpuStageProfiler::new(r, 2).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        for offset in 0..2 {
            let route = (round + offset) % 2;
            let mut slot = Some(&mut profiler);
            let writes = crate::profiling::timestamp_writes(
                &mut slot,
                if route == 0 { "ransac_counts_parent" } else { "ransac_counts_current" },
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: writes,
            });
            pass.set_pipeline(pipelines[route]);
            pass.set_bind_group(0, &groups[route], &[]);
            pass.dispatch_workgroups(p.groups, 1, 1);
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, 32);
        assert_eq!(report.stages.len(), 2);
        if round >= 3 {
            for stage in report.stages {
                samples[usize::from(stage.stage == "ransac_counts_current")]
                    .push(stage.elapsed_ns / 1_000_000.);
            }
        }
    }
    for output in outputs {
        let scores = read_scores(r, &output, p.output_bytes);
        assert_eq!(bytemuck::cast_slice::<_, u32>(&scores), expected_words);
        check_axes(&axes, threshold, &scores);
    }
    println!("RANSAC_COUNTS_RECORD {{\"points\":{n},\"hypotheses\":{h},\"parallel\":{},\"pass_old_ms\":{:?},\"pass_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",p.parallel,samples[0],samples[1],calls[0],calls[1]);
}

#[test]
#[ignore = "paired integer RANSAC reduction and complete upload/score/readback API; run alone in release mode"]
fn benchmark_complete_ransac_counts_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let cache = RansacPlanePipelines::new(r.device());
    let parent = parent_pipeline(&r, &cache);
    let prior = old::ParentPipelines::new(r.device());
    println!("RANSAC_COUNTS_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=gpu_pass_and_full_api checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (n, h) in [
        (100, 1),
        (1023, 17),
        (1024, 1),
        (1024, 17),
        (1024, 513),
        (8193, 17),
        (8193, 513),
        (65537, 17),
        (65537, 257),
        (1000003, 17),
    ] {
        measure(&r, &cache, &prior, &parent, n, h);
    }
}

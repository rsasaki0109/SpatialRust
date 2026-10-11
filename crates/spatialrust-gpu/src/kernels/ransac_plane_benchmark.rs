//! Explicit native paired pass/completion diagnostics; never a CI speed assertion.
use super::*;
use crate::{GpuStageProfiler, WgpuPowerPreference};

#[test]
#[ignore = "native paired RANSAC benchmark; requires explicit timestamp invocation"]
fn paired_previous_current_ransac_scores() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    let cache = RansacPlanePipelines::new(runtime.device());
    let previous = previous_pipeline(&runtime, &cache);
    eprintln!("adapter={:?}; 3 warmups + 20 alternating paired rounds; GPU pass excludes upload/setup/readback; completed interval is submit+wait only",runtime.adapter_info());
    for (n, h) in [
        (100, 1),
        (100, 16),
        (100, 513),
        (1023, 1),
        (1023, 16),
        (1023, 513),
        (1024, 1),
        (1024, 16),
        (1024, 513),
        (8193, 1),
        (8193, 16),
        (8193, 513),
        (65537, 1),
        (65537, 16),
        (65537, 513),
        (1000003, 1),
        (1000003, 16),
    ] {
        let (axes, hypotheses) = fixture(n, h);
        let p = plan(n as usize, h as usize, &runtime.device().limits()).unwrap();
        let current = cache.pipeline(runtime.device(), p.parallel);
        let (outputs, groups) = bindings(&runtime, &cache, &p, &axes, &hypotheses, 0.25);
        let variants = [&previous, current];
        let labels = ["previous_ransac", "current_ransac"];
        let dispatch = [h.div_ceil(256), p.groups];
        let mut stage: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        let mut completed: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..23 {
            let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
            for offset in 0..2 {
                let i = (round + offset) % 2;
                let mut encoder = runtime.device().create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(labels[i]),
                        timestamp_writes: crate::profiling::timestamp_writes(
                            &mut Some(&mut profiler),
                            labels[i],
                        ),
                    });
                    pass.set_pipeline(variants[i]);
                    pass.set_bind_group(0, &groups[i], &[]);
                    pass.dispatch_workgroups(dispatch[i], 1, 1);
                }
                let commands = encoder.finish();
                let start = std::time::Instant::now();
                runtime.queue().submit(Some(commands));
                runtime.wait_idle();
                if round >= 3 {
                    completed[i].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
            let report = profiler.collect(&runtime).unwrap();
            assert_eq!(report.device_to_host_bytes, 32);
            if round >= 3 {
                for sample in report.stages {
                    stage[usize::from(sample.stage == labels[1])].push(sample.elapsed_ns / 1000.);
                }
            }
        }
        let a = read_scores(&runtime, &outputs[0], p.output_bytes);
        let b = read_scores(&runtime, &outputs[1], p.output_bytes);
        assert_eq!(bytemuck::cast_slice::<_, u32>(&a), bytemuck::cast_slice::<_, u32>(&b));
        check_axes(&axes, 0.25, &b);
        eprintln!("PAIRED {{\"points\":{n},\"hypotheses\":{h},\"point_parallel\":{},\"previous_us\":{:?},\"current_us\":{:?},\"previous_completed_ms\":{:?},\"current_completed_ms\":{:?}}}",p.parallel,stage[0],stage[1],completed[0],completed[1]);
    }
}

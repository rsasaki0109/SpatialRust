use super::{
    replica_tests::{fixture, FROZEN, REPLICATED, RUNS},
    *,
};

#[test]
#[ignore = "explicit native completed histogram comparison"]
fn paired_frozen_replicated_histograms() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    assert!(runtime.timestamp_queries_enabled());
    let frozen = experiment::pipelines(runtime.device(), FROZEN);
    let variant = std::env::var("SPATIALRUST_HISTOGRAM_VARIANT").unwrap_or_default();
    let runs = variant == "runs";
    let replicated = experiment::pipelines(runtime.device(), if runs { RUNS } else { REPLICATED });
    eprintln!("prototype={}", if runs { "per-lane runs" } else { "eight replicas" });
    eprintln!("backend={}; device_type={}; 4 warmups + 20 alternating pairs; two-pass resident histogram through device idle; pipeline setup/upload/profiler allocation/collection/bin readback/checks/recycle excluded", runtime.adapter_info().backend, runtime.adapter_info().device_type);
    let shapes: &[(usize, usize)] = if variant == "public-large" {
        &[
            (2048, 2048),
            (4096, 1024),
            (1024, 4096),
            (8192, 513),
            (513, 8192),
            (8192, 1024),
            (4096, 4096),
        ]
    } else if variant == "replicas-large" {
        &[(2048, 2048), (4096, 1024), (1024, 4096), (8192, 513), (513, 8192), (4096, 4096)]
    } else {
        &[(17, 19), (255, 257), (640, 480), (1024, 1024), (1920, 1080), (3840, 2160)]
    };
    for &(width, height) in shapes {
        for pattern in 0..3 {
            let (image, expected) = fixture(width, height, pattern);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut partial: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut gpu: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..24 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
                    let start = std::time::Instant::now();
                    let public = variant.starts_with("public") && mode == 1;
                    if !public {
                        profiler.validate(&runtime, 2).unwrap();
                    }
                    let result = if variant.starts_with("public") {
                        if mode == 1 {
                            histogram_u8_gpu_profiled(&runtime, &source, 0, &mut profiler)
                        } else {
                            experiment::forced_histogram(
                                &runtime,
                                &source,
                                0,
                                Some(&mut profiler),
                                &frozen,
                            )
                        }
                    } else {
                        experiment::forced_histogram(
                            &runtime,
                            &source,
                            0,
                            Some(&mut profiler),
                            if mode == 0 { &frozen } else { &replicated },
                        )
                    };
                    if !public {
                        profiler.finish_recording(&result);
                    }
                    let mut output = result.unwrap();
                    runtime.wait_idle();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let report = profiler.collect(&runtime).unwrap();
                    assert_eq!(report.device_to_host_bytes, 32);
                    assert_eq!(
                        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                        ["histogram_u8_partials_gpu", "histogram_u8_reduce_gpu"]
                    );
                    assert_eq!(output.receipt().device_to_host_bytes(), 0);
                    assert_eq!(output.readback_u32(&runtime).unwrap(), expected);
                    if round >= 4 {
                        wall[mode].push(elapsed);
                        partial[mode].push(report.stages[0].elapsed_ns / 1000.0);
                        gpu[mode].push(report.stages.iter().map(|s| s.elapsed_ns / 1000.0).sum());
                    }
                    output.recycle(&runtime);
                }
            }
            eprintln!("HISTOGRAM {{\"width\":{width},\"height\":{height},\"pattern\":{pattern},\"previous_ms\":{:?},\"current_ms\":{:?},\"previous_partial_us\":{:?},\"current_partial_us\":{:?},\"previous_gpu_us\":{:?},\"current_gpu_us\":{:?}}}", wall[0], wall[1], partial[0], partial[1], gpu[0], gpu[1]);
            source.recycle(&runtime);
        }
    }
}

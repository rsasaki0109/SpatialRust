use super::*;
use crate::WgpuPowerPreference;

#[test]
#[ignore = "explicit native shared-tile Sobel paired diagnostic"]
fn paired_previous_experimental_current_sobel() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let cache =
        runtime.image_spatial_pipelines.get_or_init(|| create_spatial_pipelines(runtime.device()));
    let previous = previous_pipeline(&runtime, cache);
    let tiled = experimental_pipeline(&runtime, cache);
    eprintln!("backend={}; device_type={}; 4 warmups + 20 rotating rounds; shared host executor; completed resident timing includes validation/allocation/encoding/submit/wait, excludes upload/query allocation/query collection/readback/recycling", runtime.adapter_info().backend, runtime.adapter_info().device_type);
    for (w, h) in [
        (17, 19),
        (128, 128),
        (255, 257),
        (256, 256),
        (257, 259),
        (640, 480),
        (1024, 1024),
        (1920, 1080),
        (3840, 2160),
        (17, 4097),
        (4097, 17),
        (4096, 2048),
        (1023, 1025),
        (1025, 1025),
        (128, 8192),
        (8192, 128),
        (129, 8192),
        (8192, 129),
    ] {
        for pattern in [0, 2] {
            let image = fixture(w, h, pattern);
            let expected = oracle(&image);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let mut stages: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            let mut completed: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for round in 0..24 {
                for offset in 0..3 {
                    let i = (round + offset) % 3;
                    let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
                    let start = std::time::Instant::now();
                    let mut output = match i {
                        0 => run_pipeline(&runtime, &source, &previous, 1, Some(&mut profiler)),
                        1 => run_pipeline(&runtime, &source, &tiled, 1, Some(&mut profiler)),
                        _ => sobel_gpu_with_profiler(&runtime, &source, Some(&mut profiler)),
                    }
                    .unwrap();
                    runtime.wait_idle();
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let report = profiler.collect(&runtime).unwrap();
                    assert_eq!(report.stages.len(), 1);
                    assert_eq!(report.device_to_host_bytes, 16);
                    assert_eq!(output.receipt().stages(), ["upload_u8_texture", "sobel_gpu"]);
                    assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * 4) as u64);
                    assert_eq!(output.receipt().device_to_host_bytes(), 0);
                    let actual = output.readback_u8::<1>(&runtime).unwrap();
                    assert_eq!(actual.as_slice(), expected);
                    assert_eq!(actual.metadata(), image.metadata());
                    if round >= 4 {
                        stages[i].push(report.stages[0].elapsed_ns / 1000.0);
                        completed[i].push(ms);
                    }
                    output.recycle(&runtime);
                }
            }
            eprintln!("PAIRED {{\"width\":{w},\"height\":{h},\"hot\":{},\"previous_us\":{:?},\"experimental_us\":{:?},\"current_us\":{:?},\"previous_completed_ms\":{:?},\"experimental_completed_ms\":{:?},\"current_completed_ms\":{:?}}}", pattern==2, stages[0], stages[1], stages[2], completed[0], completed[1], completed[2]);
            source.recycle(&runtime);
        }
    }
}

use super::*;

#[test]
#[ignore = "explicit native resident GPU gray diagnostic"]
fn paired_frozen_current_and_large_gray() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let cache = runtime.image_gray_pipeline.get_or_init(|| create_gray_pipeline(runtime.device()));
    let frozen = create_pipeline(
        runtime.device(),
        &cache.bind_group_layout,
        include_str!("../../../tests/fixtures/gray_previous.wgsl"),
    );
    eprintln!("backend={}; device_type={}; 2 warmups + 20 alternating rounds; shared host executor; timings include validation/allocation/encoding/submit/wait, exclude upload/pipeline setup/profiler creation/query collection/readback/checks/recycle",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    for (w, h) in
        [(17, 19), (128, 128), (640, 480), (1920, 1080), (3840, 2160), (4096, 4095), (4096, 4096)]
    {
        let image = fixture(w, h, 0);
        let expected = oracle(image.as_slice());
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let plan = gray_plan(w as u32, h as u32, &runtime.device().limits()).unwrap();
        // The frozen linear kernel has no valid dispatch for the large case.
        let modes = if plan.groups_y == 1 { 2 } else { 1 };
        let mut stage: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..22 {
            for offset in 0..modes {
                let mode = (round + offset) % modes;
                let current = modes == 1 || mode == 1;
                let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
                let start = std::time::Instant::now();
                let output = if current {
                    rgb_to_gray_gpu_with_profiler(&runtime, &source, Some(&mut profiler)).unwrap()
                } else {
                    source.validate_runtime(&runtime).unwrap();
                    gray_plan(w as u32, h as u32, &runtime.device().limits()).unwrap();
                    profiler.validate(&runtime, 1).unwrap();
                    gray_dispatch(&runtime, &source, plan, cache, &frozen, Some(&mut profiler))
                        .unwrap()
                };
                runtime.wait_idle();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let report = profiler.collect(&runtime).unwrap();
                assert_eq!(report.stages.len(), 1);
                assert_eq!(report.device_to_host_bytes, 16);
                assert_eq!(output.receipt().stages(), ["upload_u8_texture", "rgb_to_gray_gpu"]);
                assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * 4) as u64);
                assert_eq!(output.receipt().device_to_host_bytes(), 0);
                assert_rgba(&runtime, &output, &expected);
                assert_eq!(output.metadata().color_space, ColorSpace::Gray);
                assert_eq!(output.metadata().color_range, image.metadata().color_range);
                if round >= 2 {
                    let i = usize::from(current);
                    stage[i].push(report.stages[0].elapsed_ns / 1000.0);
                    wall[i].push(ms);
                }
                output.recycle(&runtime);
            }
        }
        eprintln!("GRAY {{\"width\":{w},\"height\":{h},\"groups_x\":{},\"groups_y\":{},\"previous_us\":{:?},\"current_us\":{:?},\"previous_ms\":{:?},\"current_ms\":{:?}}}",plan.groups_x,plan.groups_y,stage[0],stage[1],wall[0],wall[1]);
        source.recycle(&runtime);
    }
}

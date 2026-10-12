use super::*;

fn measure<const C: usize>(
    runtime: &WgpuRuntime,
    cache: &AiPackPipeline,
    previous: &wgpu::ComputePipeline,
    w: usize,
    h: usize,
) {
    let image = fixture::<C>(w, h);
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    let plan = pack_plan(w as u32, h as u32, C as u32, &runtime.device().limits()).unwrap();
    let legacy = force_legacy(plan, w as u32, h as u32);
    let scale = 1.0 / 255.0;
    let mean = [0.485, 0.456, 0.406, 0.5];
    let std = [0.229, 0.224, 0.225, 0.75];
    let mut reference =
        pack_dispatch(runtime, &source, scale, mean, std, legacy, cache, previous, None).unwrap();
    let expected = reference.readback_f32(runtime).unwrap();
    reference.recycle(runtime);
    let mut stage: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..22 {
        for offset in 0..2 {
            let i = (round + offset) % 2;
            let mut profiler = crate::GpuStageProfiler::new(runtime, 1).unwrap();
            let start = std::time::Instant::now();
            let mut output = if i == 0 {
                source.validate_runtime(runtime).unwrap();
                pack_plan(w as u32, h as u32, C as u32, &runtime.device().limits()).unwrap();
                profiler.validate(runtime, 1).unwrap();
                pack_dispatch(
                    runtime,
                    &source,
                    scale,
                    mean,
                    std,
                    legacy,
                    cache,
                    previous,
                    Some(&mut profiler),
                )
                .unwrap()
            } else {
                pack_ai_chw_gpu_with_profiler(
                    runtime,
                    &source,
                    scale,
                    mean,
                    std,
                    Some(&mut profiler),
                )
                .unwrap()
            };
            runtime.wait_idle();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let report = profiler.collect(runtime).unwrap();
            assert_eq!(report.stages.len(), 1);
            assert_eq!(report.device_to_host_bytes, 16);
            assert_eq!(output.shape(), [C as u32, h as u32, w as u32]);
            assert_eq!(output.receipt().device_to_host_bytes(), 0);
            assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * C * 4) as u64);
            assert_words(&output.readback_f32(runtime).unwrap(), &expected);
            output.recycle(runtime);
            if round >= 2 {
                stage[i].push(report.stages[0].elapsed_ns / 1000.0);
                wall[i].push(ms);
            }
        }
    }
    eprintln!("AI {{\"width\":{w},\"height\":{h},\"channels\":{C},\"previous_us\":{:?},\"current_us\":{:?},\"previous_ms\":{:?},\"current_ms\":{:?}}}",stage[0],stage[1],wall[0],wall[1]);
    source.recycle(runtime);
}

#[test]
#[ignore = "explicit native paired resident GPU AI packing diagnostic"]
fn paired_frozen_and_validated_ai_pack() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let cache =
        runtime.image_ai_pack_pipeline.get_or_init(|| create_ai_pack_pipeline(runtime.device()));
    let previous = create_probe_pipeline(
        runtime.device(),
        &cache.layout,
        include_str!("../../tests/fixtures/ai_pack_previous.wgsl"),
        None,
    );
    eprintln!("backend={}; device_type={}; 2 warmups + 20 alternating rounds; shared host executor; includes validation/output/uniform allocation/encoding/submit/wait; excludes upload/pipeline setup/profiler creation/query collection/readback/checks/recycle",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    for (w, h) in [
        (17, 19),
        (128, 128),
        (256, 256),
        (640, 480),
        (1920, 1080),
        (3840, 2160),
        (17, 4097),
        (4097, 17),
    ] {
        measure::<1>(&runtime, cache, &previous, w, h);
        measure::<3>(&runtime, cache, &previous, w, h);
        measure::<4>(&runtime, cache, &previous, w, h);
    }
}

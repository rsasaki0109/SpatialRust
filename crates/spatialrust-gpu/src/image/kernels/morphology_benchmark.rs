//! Explicit paired diagnostics, excluding transfers, setup and query collection.
use super::*;
use crate::WgpuPowerPreference;

#[test]
#[ignore = "explicit native paired morphology diagnostic with timestamp queries"]
fn paired_previous_current_morphology() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    let cache = create_spatial_pipelines(runtime.device());
    let previous = previous_pipeline(&runtime, &cache);
    eprintln!("backend={}; device_type={}; 3 warmups + 20 alternating paired rounds; stage sum excludes transfers/setup; completion includes submit+wait only",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    for (w, h, kw, kh) in [
        (17, 19, 3, 3),
        (17, 19, 9, 9),
        (33, 35, 9, 9),
        (33, 35, 31, 31),
        (128, 128, 9, 9),
        (640, 480, 3, 3),
        (640, 480, 9, 9),
        (640, 480, 31, 31),
        (1920, 1080, 3, 3),
        (1920, 1080, 9, 9),
        (1920, 1080, 31, 31),
        (3840, 2160, 3, 3),
        (3840, 2160, 9, 9),
        (3840, 2160, 31, 31),
        (640, 480, 1, 31),
        (640, 480, 31, 1),
        (640, 480, 3, 27),
        (640, 480, 27, 3),
        (640, 480, 15, 15),
    ] {
        let image = fixture(w, h, 0);
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        for operation in [GpuMorphology::Erode, GpuMorphology::Dilate] {
            let p =
                morphology_plan(w as u32, h as u32, kw, kh, operation, &runtime.device().limits())
                    .unwrap();
            let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&p.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let scratch = p
                .separable
                .then(|| create_texture(&runtime, w as u32, h as u32, "paired-morphology-scratch"));
            let scratch_view = scratch.as_ref().map(|t| t.create_view(&Default::default()));
            let outputs: [_; 2] = std::array::from_fn(|_| {
                create_texture(&runtime, w as u32, h as u32, "paired-morphology")
            });
            let source_view = source.view();
            let views: [_; 2] =
                std::array::from_fn(|i| outputs[i].create_view(&Default::default()));
            let mut stage: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut completed: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..23 {
                let mut profiler =
                    crate::GpuStageProfiler::new(&runtime, if p.separable { 3 } else { 2 })
                        .unwrap();
                for offset in 0..2 {
                    let i = (round + offset) % 2;
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    if i == 0 {
                        encode_previous(
                            &runtime,
                            &cache,
                            &previous,
                            &p,
                            &source_view,
                            &views[i],
                            &uniform,
                            &mut encoder,
                            Some(&mut profiler),
                        );
                    } else {
                        encode_morphology(
                            &runtime,
                            &cache,
                            &p,
                            &source_view,
                            &views[i],
                            scratch_view.as_ref(),
                            &uniform,
                            &mut encoder,
                            Some(&mut profiler),
                        );
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
                assert_eq!(report.device_to_host_bytes, if p.separable { 48 } else { 32 });
                if round >= 3 {
                    let mut sums = [0.; 2];
                    for sample in report.stages {
                        sums[usize::from(sample.stage != "previous_morphology")] +=
                            sample.elapsed_ns / 1000.;
                    }
                    for i in 0..2 {
                        stage[i].push(sums[i]);
                    }
                }
            }
            let values: Vec<_> = outputs
                .into_iter()
                .map(|output| {
                    let mut gpu = GpuImage::from_parts(
                        &runtime,
                        w as u32,
                        h as u32,
                        4,
                        output,
                        source.metadata(),
                        Default::default(),
                    )
                    .unwrap();
                    let values = gpu.readback_u8::<4>(&runtime).unwrap();
                    gpu.recycle(&runtime);
                    values
                })
                .collect();
            assert_eq!(values[0].as_slice(), values[1].as_slice());
            eprintln!("PAIRED {{\"width\":{w},\"height\":{h},\"kw\":{kw},\"kh\":{kh},\"operation\":\"{operation:?}\",\"separable\":{},\"previous_us\":{:?},\"current_us\":{:?},\"previous_completed_ms\":{:?},\"current_completed_ms\":{:?}}}",p.separable,stage[0],stage[1],completed[0],completed[1]);
        }
        source.recycle(&runtime);
    }
}

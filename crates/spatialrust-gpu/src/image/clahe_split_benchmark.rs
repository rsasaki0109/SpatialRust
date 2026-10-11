use super::*;
use crate::WgpuPowerPreference;
use spatialrust_image::Image;

const FROZEN_LOOKUP: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("clahe_clip_scan.wgsl"),
    include_str!("../../tests/fixtures/clahe_lookup_single_histogram.wgsl")
);

#[test]
#[ignore = "native paired completed resident CLAHE diagnostic; explicit invocation only"]
fn paired_direct_split_clahe_completed_operations() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    eprintln!("backend={}; device_type={}; 4 warmups + 20 alternating pairs; resident operation API-to-device-idle; upload, profiler allocation/collection and output readback excluded",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    let pipelines =
        runtime.image_clahe_pipelines.get_or_init(|| ClahePipelines::new(runtime.device()));
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-direct-clahe-lookup"),
        source: wgpu::ShaderSource::Wgsl(FROZEN_LOOKUP.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&pipelines.layouts[1]],
        push_constant_ranges: &[],
    });
    let frozen = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("clahe_lookup"),
        compilation_options: Default::default(),
        cache: None,
    });
    for (width, height, x, y) in [
        (255, 257, 1, 1),
        (256, 256, 1, 1),
        (257, 257, 1, 1),
        (257, 259, 1, 1),
        (257, 259, 3, 5),
        (640, 480, 1, 1),
        (640, 480, 8, 8),
        (3840, 2160, 1, 1),
        (3840, 2160, 2, 2),
        (3840, 2160, 4, 4),
        (3840, 2160, 8, 8),
        (3840, 2160, 32, 32),
    ] {
        for hot in [false, true] {
            let image = Image::<u8, 1>::try_new(
                width,
                height,
                (0..width * height)
                    .map(|i| if hot { 43 } else { ((i * 37 + i / 997) % 129 + 32) as u8 })
                    .collect(),
            )
            .unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            for clip in [0., 2.] {
                let expected =
                    spatialrust_vision::clahe(image.view(), clip, x as usize, y as usize).unwrap();
                let selected =
                    plan(width as u32, height as u32, clip, x, y, &runtime.device().limits())
                        .unwrap();
                let chunks = selected.split.as_ref().map_or(1, |p| p.params.chunks);
                let partial_bytes = selected.split.as_ref().map_or(0, |p| p.partial_bytes);
                let mut completed: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                let mut histogram: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                let mut total: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..24 {
                    for offset in 0..2 {
                        let i = (round + offset) % 2;
                        let mut profiler = GpuStageProfiler::new(&runtime, 4).unwrap();
                        let start = std::time::Instant::now();
                        let mut output = if i == 0 {
                            profiler.validate(&runtime, 3).unwrap();
                            source.validate_runtime(&runtime).unwrap();
                            let mut p = plan(
                                width as u32,
                                height as u32,
                                clip,
                                x,
                                y,
                                &runtime.device().limits(),
                            )
                            .unwrap();
                            p.split = None;
                            execute_clahe(
                                &runtime,
                                &source,
                                &p,
                                pipelines,
                                &frozen,
                                Some(&mut profiler),
                            )
                            .unwrap()
                        } else {
                            clahe_gpu_profiled(&runtime, &source, clip, x, y, &mut profiler)
                                .unwrap()
                        };
                        runtime.wait_idle();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.;
                        let report = profiler.collect(&runtime).unwrap();
                        assert_eq!(report.stages.len(), if i == 0 || chunks == 1 { 3 } else { 4 });
                        assert_eq!(report.device_to_host_bytes, report.stages.len() as u64 * 16);
                        let histogram_us = report
                            .stages
                            .iter()
                            .filter(|s| {
                                s.stage == "clahe_lookup_gpu" || s.stage == "clahe_histogram_gpu"
                            })
                            .map(|s| s.elapsed_ns / 1000.)
                            .sum::<f64>();
                        let total_us =
                            report.stages.iter().map(|s| s.elapsed_ns / 1000.).sum::<f64>();
                        if round >= 4 {
                            completed[i].push(elapsed);
                            histogram[i].push(histogram_us);
                            total[i].push(total_us);
                        }
                        let actual = output.readback_u8::<1>(&runtime).unwrap();
                        assert!(actual.as_slice()==expected.as_slice(),"{width}x{height} grid{x}x{y} hot{hot} clip{clip} round{round} variant{i}");
                        assert_eq!(actual.metadata(), expected.metadata());
                        output.recycle(&runtime);
                    }
                }
                eprintln!("PAIRED {{\"width\":{width},\"height\":{height},\"tiles_x\":{x},\"tiles_y\":{y},\"hot\":{hot},\"clip\":{clip},\"chunks\":{chunks},\"partial_bytes\":{partial_bytes},\"previous_histogram_us\":{:?},\"current_histogram_us\":{:?},\"previous_total_us\":{:?},\"current_total_us\":{:?},\"previous_completed_ms\":{:?},\"current_completed_ms\":{:?}}}",histogram[0],histogram[1],total[0],total[1],completed[0],completed[1]);
            }
            source.recycle(&runtime);
        }
    }
}

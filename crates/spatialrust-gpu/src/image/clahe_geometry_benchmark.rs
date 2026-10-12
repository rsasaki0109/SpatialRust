use super::*;
use crate::WgpuPowerPreference;
use spatialrust_image::Image;

pub(super) const PREVIOUS_GEOMETRY: &str = concat!(
    include_str!("rounding_u32.wgsl"),
    include_str!("../../tests/fixtures/clahe_binary64_previous.wgsl"),
    include_str!("clahe_params.wgsl"),
    include_str!("clahe_geometry.wgsl")
);

#[test]
#[ignore = "native completed resident geometry diagnostic; explicit invocation only"]
fn paired_previous_current_clahe_geometry() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(runtime.timestamp_queries_enabled());
    eprintln!("backend={}; device_type={}; 4 warmups + 20 alternating pairs; completed resident shared-executor calls, uploads/query-allocation/query-collection/readback excluded",runtime.adapter_info().backend,runtime.adapter_info().device_type);
    let current =
        runtime.image_clahe_pipelines.get_or_init(|| ClahePipelines::new(runtime.device()));
    let mut previous = ClahePipelines::new(runtime.device());
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-clahe-geometry"),
        source: wgpu::ShaderSource::Wgsl(PREVIOUS_GEOMETRY.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&previous.layouts[0]],
        push_constant_ranges: &[],
    });
    previous.pipelines[0] =
        runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("clahe_geometry"),
            compilation_options: Default::default(),
            cache: None,
        });
    for (width, height, x, y) in [
        (1usize, 257usize, 1u32, 257u32),
        (257, 1, 257, 1),
        (128, 128, 8, 8),
        (256, 256, 8, 8),
        (256, 256, 1, 1),
        (257, 259, 3, 5),
        (640, 480, 8, 8),
        (1024, 1024, 8, 8),
        (1920, 1080, 8, 8),
        (3840, 2160, 8, 8),
        (4096, 2048, 8, 8),
        (4096, 2048, 1, 1),
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
            let expected =
                spatialrust_vision::clahe(image.view(), 2., x as usize, y as usize).unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let mut geometry: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut total: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            let mut completed: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..24 {
                for offset in 0..2 {
                    let i = (round + offset) % 2;
                    let family = if i == 0 { &previous } else { current };
                    let mut profiler = GpuStageProfiler::new(&runtime, 4).unwrap();
                    let start = std::time::Instant::now();
                    source.validate_runtime(&runtime).unwrap();
                    let p = plan(width as u32, height as u32, 2., x, y, &runtime.device().limits())
                        .unwrap();
                    profiler.validate(&runtime, if p.split.is_some() { 4 } else { 3 }).unwrap();
                    let mut output = execute_clahe(
                        &runtime,
                        &source,
                        &p,
                        family,
                        &family.pipelines[1],
                        Some(&mut profiler),
                    )
                    .unwrap();
                    runtime.wait_idle();
                    let ms = start.elapsed().as_secs_f64() * 1000.;
                    let report = profiler.collect(&runtime).unwrap();
                    assert_eq!(report.stages.len(), if p.split.is_some() { 4 } else { 3 });
                    assert_eq!(report.device_to_host_bytes, report.stages.len() as u64 * 16);
                    let geometry_us = report
                        .stages
                        .iter()
                        .find(|s| s.stage == "clahe_geometry_gpu")
                        .unwrap()
                        .elapsed_ns
                        / 1000.;
                    let total_us = report.stages.iter().map(|s| s.elapsed_ns / 1000.).sum::<f64>();
                    if round >= 4 {
                        geometry[i].push(geometry_us);
                        total[i].push(total_us);
                        completed[i].push(ms);
                    }
                    assert_eq!(
                        output.receipt().stages().len(),
                        if p.split.is_some() { 5 } else { 4 }
                    );
                    let actual = output.readback_u8::<1>(&runtime).unwrap();
                    assert_eq!(
                        actual.as_slice(),
                        expected.as_slice(),
                        "{width}x{height}, grid{x}x{y}, hot{hot}, round{round}, variant{i}"
                    );
                    assert_eq!(actual.metadata(), expected.metadata());
                    output.recycle(&runtime);
                }
            }
            eprintln!("PAIRED {{\"width\":{width},\"height\":{height},\"tiles_x\":{x},\"tiles_y\":{y},\"hot\":{hot},\"previous_geometry_us\":{:?},\"current_geometry_us\":{:?},\"previous_total_us\":{:?},\"current_total_us\":{:?},\"previous_completed_ms\":{:?},\"current_completed_ms\":{:?}}}",geometry[0],geometry[1],total[0],total[1],completed[0],completed[1]);
            source.recycle(&runtime);
        }
    }
}

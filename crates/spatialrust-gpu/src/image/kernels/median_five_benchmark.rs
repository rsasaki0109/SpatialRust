use super::*;
use std::time::Instant;
use wgpu::util::DeviceExt;

fn timed_passes<const C: usize>(
    r: &WgpuRuntime,
    source: &GpuImage,
    expected: &spatialrust_image::Image<u8, C>,
    kernel: u32,
    b: GpuImageBorder,
) -> [Vec<f64>; 2] {
    let sources = [old::shader(kernel), shader(kernel)];
    let pipelines: [_; 2] = std::array::from_fn(|i| {
        let module = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("median-network-timestamp"),
            source: wgpu::ShaderSource::Wgsl(sources[i].as_str().into()),
        });
        r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("median"),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let params = Params {
        width: source.width(),
        height: source.height(),
        border: u32::from(b == GpuImageBorder::ConstantZero),
        padding: 0,
    };
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let outputs: [_; 2] = std::array::from_fn(|_| {
        create_texture(r, source.width(), source.height(), "median-timed-output")
    });
    let groups: [_; 2] = std::array::from_fn(|i| {
        r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines[i].get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &outputs[i].create_view(&Default::default()),
                    ),
                },
            ],
        })
    });
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = crate::GpuStageProfiler::new(r, 2).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        for offset in 0..2 {
            let route = (round + offset) % 2;
            let mut slot = Some(&mut profiler);
            let writes = crate::profiling::timestamp_writes(
                &mut slot,
                if route == 0 { "median_previous" } else { "median_current" },
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: writes,
            });
            pass.set_pipeline(&pipelines[route]);
            pass.set_bind_group(0, &groups[route], &[]);
            pass.dispatch_workgroups(
                source.width().div_ceil(WORKGROUP),
                source.height().div_ceil(WORKGROUP),
                1,
            );
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, 32);
        assert_eq!(report.stages.len(), 2);
        if round >= 3 {
            for stage in report.stages {
                samples[usize::from(stage.stage == "median_current")]
                    .push(stage.elapsed_ns / 1_000_000.0);
            }
        }
    }
    for texture in outputs {
        let mut output = GpuImage::from_parts(
            r,
            source.width(),
            source.height(),
            C as u32,
            texture,
            source.metadata(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(&output.readback_u8::<C>(r).unwrap(), expected);
        output.recycle(r);
    }
    samples
}

fn measure<const C: usize>(
    r: &WgpuRuntime,
    w: usize,
    h: usize,
    kernel: u32,
    profile: usize,
    b: GpuImageBorder,
) {
    let image = fixture::<C>(w, h, profile);
    let expected = median_blur(image.view(), kernel as usize, border(b)).unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let old_cache = OnceLock::new();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for family in 0..2 {
            for offset in 0..2 {
                let route = (round + offset) % 2;
                let start = Instant::now();
                let uploaded = if family == 1 {
                    Some(GpuImage::upload_u8(r, image.view()).unwrap())
                } else {
                    None
                };
                let input = uploaded.as_ref().unwrap_or(&source);
                let mut output = if route == 0 {
                    old::previous(r, &old_cache, input, kernel, b)
                } else {
                    median_blur_gpu(r, input, kernel, b)
                }
                .unwrap();
                let elapsed = if family == 0 {
                    r.wait_idle();
                    start.elapsed().as_secs_f64() * 1000.0
                } else {
                    let actual = output.readback_u8::<C>(r).unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    output.recycle(r);
                    uploaded.unwrap().recycle(r);
                    assert_eq!(actual, expected);
                    if round >= 3 {
                        samples[family * 2 + route].push(elapsed);
                    }
                    continue;
                };
                assert_eq!(output.receipt().host_to_device_bytes(), (w * h * 4) as u64);
                assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * 4) as u64);
                assert_eq!(output.receipt().device_to_host_bytes(), 0);
                assert_eq!(output.readback_u8::<C>(r).unwrap(), expected);
                output.recycle(r);
                if round >= 3 {
                    samples[family * 2 + route].push(elapsed);
                }
            }
        }
    }
    let pass = timed_passes(r, &source, &expected, kernel, b);
    source.recycle(r);
    println!("MEDIAN_FIVE_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"kernel\":{kernel},\"profile\":{profile},\"border\":{},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",u32::from(b==GpuImageBorder::ConstantZero),samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "requires isolated completed-call native release measurement"]
fn benchmark_completed_odd_even_median_five_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let info = r.adapter_info();
    println!(
        "MEDIAN_FIVE_ENV backend={} device_type={} available_workers={} texture_budget={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        r.image_texture_pool_stats().budget_bytes
    );
    let b = GpuImageBorder::Replicate;
    measure::<1>(&r, 65, 33, 5, 0, b);
    measure::<3>(&r, 257, 193, 5, 0, b);
    measure::<1>(&r, 1024, 1024, 5, 0, b);
    measure::<3>(&r, 1920, 1080, 5, 0, b);
    measure::<4>(&r, 1920, 1080, 5, 0, b);
    measure::<4>(&r, 1025, 257, 5, 0, b);
    measure::<3>(&r, 4097, 17, 5, 0, b);
    measure::<4>(&r, 1024, 1024, 5, 2, b);
    measure::<3>(&r, 1024, 1024, 5, 4, GpuImageBorder::ConstantZero);
    measure::<4>(&r, 257, 193, 3, 0, b);
    measure::<4>(&r, 1024, 1024, 3, 0, b);
}

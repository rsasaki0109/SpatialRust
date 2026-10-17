use super::*;
use std::{hint::black_box, time::Instant};

fn vertical_passes<const C: usize>(
    r: &WgpuRuntime,
    prior: &old::ParentGaussianPipelines,
    source: &GpuImage,
    options: GpuGaussianBlurOptions,
    expected: &Image<u8, C>,
) -> [Vec<f64>; 2] {
    let p = plan(source.width(), source.height(), C as u32, options, &r.device().limits()).unwrap();
    let cache = GaussianPipelines::new();
    let current = cache.pipelines(r.device(), p.packed);
    let previous = prior.pipelines(r.device(), p.packed);
    let sums = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gaussian-split-timed-sums"),
        size: p.scratch_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let horizontal = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &current[0].get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source.view()),
            },
            wgpu::BindGroupEntry { binding: 2, resource: sums.as_entire_binding() },
        ],
    });
    // The unchanged horizontal stage prepares a shared input outside timestamps.
    let mut encoder = r.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&current[0]);
        pass.set_bind_group(0, &horizontal, &[]);
        pass.dispatch_workgroups(p.params.groups, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    r.wait_idle();
    let outputs: [_; 2] = std::array::from_fn(|_| {
        create_texture(r, source.width(), source.height(), "gaussian-split-timed-output")
    });
    let pipelines = [&previous[1], &current[1]];
    let groups: [_; 2] = std::array::from_fn(|i| {
        r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines[i].get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &outputs[i].create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry { binding: 2, resource: sums.as_entire_binding() },
            ],
        })
    });
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, 2).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        for offset in 0..2 {
            let route = (round + offset) % 2;
            let mut slot = Some(&mut profiler);
            let writes = crate::profiling::timestamp_writes(
                &mut slot,
                if route == 0 { "gaussian_split_previous" } else { "gaussian_split_current" },
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: writes,
            });
            pass.set_pipeline(pipelines[route]);
            pass.set_bind_group(0, &groups[route], &[]);
            pass.dispatch_workgroups(p.params.groups, 1, 1);
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.device_to_host_bytes, 32);
        assert_eq!(report.stages.len(), 2);
        if round >= 3 {
            for stage in report.stages {
                samples[usize::from(stage.stage == "gaussian_split_current")]
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
    prior: &old::ParentGaussianPipelines,
    w: usize,
    h: usize,
    kernel: u32,
    constant: bool,
) {
    let image = fixture::<C>(w, h, false);
    let options = GpuGaussianBlurOptions {
        kernel_width: kernel,
        kernel_height: kernel,
        border: if constant {
            GpuGaussianBorder::Constant([19, 73, 129, 251])
        } else {
            GpuGaussianBorder::Reflect101
        },
        ..Default::default()
    };
    let expected = gaussian_blur_u8(
        image.view(),
        kernel as usize,
        kernel as usize,
        options.sigma_x,
        options.sigma_y,
        border(options.border),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = (round + offset) % 4;
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| GpuImage::upload_u8(black_box(r), black_box(image.view())).unwrap());
            let input = uploaded.as_ref().unwrap_or(&source);
            let mut output = if current {
                gaussian_blur_gpu(black_box(r), black_box(input), black_box(options)).unwrap()
            } else {
                old::previous(prior, black_box(r), black_box(input), black_box(options), None)
                    .unwrap()
            };
            let downloaded = if full {
                Some(output.readback_u8::<C>(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let downloaded = downloaded.unwrap_or_else(|| output.readback_u8::<C>(r).unwrap());
            assert_eq!(downloaded, expected);
            let scratch = if kernel == 7 || C == 1 { C } else { C.div_ceil(2) };
            assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * (scratch + 1) * 4) as u64);
            output.recycle(r);
            if let Some(input) = uploaded {
                input.recycle(r);
            }
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    let pass = vertical_passes(r, prior, &source, options, &expected);
    source.recycle(r);
    println!("GAUSSIAN_SPLIT_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"kernel\":{kernel},\"constant\":{constant},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?},\"pass_old_ms\":{:?},\"pass_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3],pass[0],pass[1]);
}

#[test]
#[ignore = "paired vertical-pass and completed Gaussian calls; run alone in release mode"]
fn benchmark_complete_gaussian_split_calls() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    assert!(r.timestamp_queries_enabled());
    let prior = old::ParentGaussianPipelines::new();
    println!("GAUSSIAN_SPLIT_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=vertical_pass_resident_and_full checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    measure::<4>(&r, &prior, 16, 17, 7, false);
    measure::<3>(&r, &prior, 257, 259, 7, false);
    measure::<1>(&r, &prior, 1920, 1080, 7, false);
    measure::<2>(&r, &prior, 1920, 1080, 7, false);
    measure::<3>(&r, &prior, 1920, 1080, 7, false);
    measure::<4>(&r, &prior, 1920, 1080, 7, false);
    measure::<3>(&r, &prior, 3840, 2160, 7, false);
    measure::<3>(&r, &prior, 1025, 257, 7, true);
    measure::<1>(&r, &prior, 1920, 1080, 5, false);
    measure::<4>(&r, &prior, 1920, 1080, 5, false);
}

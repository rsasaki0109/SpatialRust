use super::*;

#[path = "gaussian_packed_reference.rs"]
mod frozen;

#[test]
fn gaussian_packed_weights_fit_sixteen_bits_and_keep_original_capacity_errors() {
    let limits = wgpu::Limits::default();
    let mut checked = 0;
    for kernel in [3, 5] {
        for sigma in [f64::from_bits(1), 1e-300, 1e-100, 0.05, 0.17, 0.5, 1.2, 7.0, 1e100, f64::MAX]
            .into_iter()
            .chain((1..=4096).map(|i| 2.0f64.powf(i as f64 / 32.0 - 64.0)))
        {
            let w = weights(kernel, sigma, 8);
            let taps: Vec<_> = w.iter().flatten().copied().collect();
            assert!(taps.iter().all(|&v| v <= 256));
            assert_eq!(taps.iter().sum::<u32>(), 256);
            assert_eq!(taps.iter().map(|&v| 255 * v).sum::<u32>(), 65280);
            checked += 1;
        }
    }
    for c in 1..=4 {
        for (kw, kh) in [(3, 3), (3, 5), (5, 3), (5, 5), (7, 3), (3, 7), (7, 7)] {
            let o = GpuGaussianBlurOptions {
                kernel_width: kw,
                kernel_height: kh,
                ..Default::default()
            };
            let p = plan(257, 259, c, o, &limits).unwrap();
            let packed = kw != 7 && kh != 7 && c >= 2;
            assert_eq!(p.packed, packed);
            assert_eq!(p.params.padding, [if packed { c.div_ceil(2) } else { 0 }, 0, 0]);
            assert_eq!(
                p.scratch_bytes,
                257 * 259 * 4 * u64::from(if packed { c.div_ceil(2) } else { c })
            );
            for field in 0..17 {
                let mut bad = limits.clone();
                match field {
                    0 => bad.max_compute_invocations_per_workgroup = 255,
                    1 => bad.max_compute_workgroup_size_x = 255,
                    2 => bad.max_compute_workgroups_per_dimension = 0,
                    3 => bad.max_sampled_textures_per_shader_stage = 0,
                    4 => bad.max_storage_textures_per_shader_stage = 0,
                    5 => bad.max_storage_buffers_per_shader_stage = 0,
                    6 => bad.max_uniform_buffers_per_shader_stage = 0,
                    7 => bad.max_uniform_buffer_binding_size = 127,
                    8 => bad.max_buffer_size = 127,
                    9 => bad.max_storage_buffer_binding_size = 257 * 259 * c * 4 - 1,
                    10 => bad.max_bind_groups = 0,
                    11 => bad.max_bindings_per_bind_group = 2,
                    12 => bad.max_compute_workgroup_size_y = 0,
                    13 => bad.max_compute_workgroup_size_z = 0,
                    14 => bad.max_texture_dimension_2d = 258,
                    15 => bad.max_buffer_size = u64::from(257 * 259 * c * 4 - 1),
                    _ => bad.max_storage_buffer_binding_size = 257 * 259 * c * 4,
                }
                assert_eq!(
                    plan(257, 259, c, o, &bad).map(|_| ()),
                    frozen::baseline_plan_result(257, 259, c, o, &bad)
                );
            }
        }
    }
    eprintln!("GAUSSIAN_PACKED_WEIGHTS verified {checked} nonnegative Q8 mass/bound cases and exact original limit/error outcomes");
}

fn mapped(i: isize, n: usize, b: GpuGaussianBorder) -> Option<usize> {
    if (0..n as isize).contains(&i) {
        return Some(i as usize);
    }
    match b {
        GpuGaussianBorder::Constant(_) => None,
        GpuGaussianBorder::Replicate => Some(i.clamp(0, n as isize - 1) as usize),
        _ if n == 1 => Some(0),
        GpuGaussianBorder::Wrap => Some(i.rem_euclid(n as isize) as usize),
        GpuGaussianBorder::Reflect => {
            let v = i.rem_euclid(2 * n as isize) as usize;
            Some(if v < n { v } else { 2 * n - 1 - v })
        }
        GpuGaussianBorder::Reflect101 => {
            let v = i.rem_euclid(2 * (n - 1) as isize) as usize;
            Some(if v < n { v } else { 2 * (n - 1) - v })
        }
    }
}

fn raw_words(runtime: &WgpuRuntime, b: &wgpu::Buffer) -> Vec<u32> {
    let slice = b.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
    runtime.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    b.unmap();
    result
}

fn raw_horizontal<const C: usize>(
    r: &WgpuRuntime,
    old: &frozen::FrozenGaussianPipelines,
    w: usize,
    h: usize,
    o: GpuGaussianBlurOptions,
    hot: bool,
) -> usize {
    let image = fixture::<C>(w, h, hot);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut p = plan(w as u32, h as u32, C as u32, o, &r.device().limits()).unwrap();
    assert!(p.packed);
    // Deliberately force grid-stride work even on small fixtures.
    p.params.groups = 1;
    let words = w * h * C.div_ceil(2);
    let allocate = |label, len| {
        let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&vec![0xdead_beef_u32; len]));
        buffer
    };
    let packed = allocate("gaussian-packed-guarded", words + 8);
    let scalar = allocate("gaussian-scalar-guarded", w * h * C + 8);
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gaussian-packed-probe"),
        size: packed.size() + scalar.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-packed-probe-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let cache = GaussianPipelines::new();
    let view = source.view();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in
        [(&cache.pipelines(r.device(), true)[0], &packed), (old.horizontal(), &scalar)]
    {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian-packed-probe-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&packed, 0, &staging, 0, packed.size());
    encoder.copy_buffer_to_buffer(&scalar, 0, &staging, packed.size(), scalar.size());
    r.queue().submit(Some(encoder.finish()));
    let result = raw_words(r, &staging);
    let previous = &result[words + 8..];
    assert_eq!(&result[words..words + 8], &[0xdead_beef; 8]);
    assert_eq!(&previous[w * h * C..], &[0xdead_beef; 8]);
    let constant = if let GpuGaussianBorder::Constant(v) = o.border { v } else { [0; 4] };
    for pixel in 0..w * h {
        for c in 0..C {
            let got = (result[pixel * C.div_ceil(2) + c / 2] >> ((c % 2) * 16)) & 65535;
            assert_eq!(got, previous[pixel * C + c]);
            let expected = (0..o.kernel_width as usize)
                .map(|tap| {
                    let v = mapped(
                        (pixel % w) as isize + tap as isize - (o.kernel_width / 2) as isize,
                        w,
                        o.border,
                    )
                    .map_or(constant[c], |sx| image.as_slice()[((pixel / w) * w + sx) * C + c]);
                    u32::from(v) * p.params.weights_x[tap / 4][tap % 4]
                })
                .sum::<u32>();
            assert_eq!(got, expected);
            if hot {
                assert_eq!(got, 65280);
            }
        }
        if C % 2 == 1 {
            assert_eq!(result[(pixel + 1) * C.div_ceil(2) - 1] >> 16, 0);
        }
    }
    source.recycle(r);
    w * h * C * 2
}

#[test]
fn gaussian_packed_horizontal_matches_original_and_cpu_words_guards_and_odd_channels() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::FrozenGaussianPipelines::new(r.device());
    let mut checked = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for kw in [3, 5] {
            for b in [
                GpuGaussianBorder::Replicate,
                GpuGaussianBorder::Reflect,
                GpuGaussianBorder::Reflect101,
                GpuGaussianBorder::Wrap,
                GpuGaussianBorder::Constant([19, 73, 129, 251]),
            ] {
                for sigma in [0.17, 7.0] {
                    let o = GpuGaussianBlurOptions {
                        kernel_width: kw,
                        kernel_height: 5,
                        sigma_x: sigma,
                        border: b,
                        ..Default::default()
                    };
                    checked += raw_horizontal::<2>(&r, &old, w, h, o, false)
                        + raw_horizontal::<3>(&r, &old, w, h, o, false)
                        + raw_horizontal::<4>(&r, &old, w, h, o, false);
                }
            }
        }
    }
    let o = GpuGaussianBlurOptions {
        border: GpuGaussianBorder::Constant([255; 4]),
        ..Default::default()
    };
    checked += raw_horizontal::<2>(&r, &old, 1025, 1025, o, true)
        + raw_horizontal::<3>(&r, &old, 1025, 1025, o, true)
        + raw_horizontal::<4>(&r, &old, 1025, 1025, o, true);
    eprintln!("GAUSSIAN_PACKED_RAW verified {checked} original GPU/independent horizontal words; poison/guards, odd high halves, maximum mass and one-group stride");
}

fn compare_frozen<const C: usize>(
    r: &WgpuRuntime,
    old: &frozen::FrozenGaussianPipelines,
    w: usize,
    h: usize,
    o: GpuGaussianBlurOptions,
) -> usize {
    let image = fixture::<C>(w, h, false);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let expected = gaussian_blur_u8(
        image.view(),
        o.kernel_width as usize,
        o.kernel_height as usize,
        o.sigma_x,
        o.sigma_y,
        border(o.border),
    )
    .unwrap();
    let mut previous = frozen::baseline_gaussian(r, &source, o, None, old).unwrap();
    let mut actual = gaussian_blur_gpu(r, &source, o).unwrap();
    assert_eq!(actual.readback_u8::<C>(r).unwrap(), expected);
    assert_eq!(previous.readback_u8::<C>(r).unwrap(), expected);
    assert_eq!(actual.metadata(), previous.metadata());
    assert_eq!(actual.receipt().stages(), previous.receipt().stages());
    assert_eq!(actual.receipt().host_to_device_bytes(), previous.receipt().host_to_device_bytes());
    assert_eq!(actual.receipt().device_to_host_bytes(), previous.receipt().device_to_host_bytes());
    let scratch = if o.kernel_width != 7 && o.kernel_height != 7 { C.div_ceil(2) } else { C };
    assert_eq!(actual.receipt().gpu_to_gpu_bytes(), (w * h * (scratch + 1) * 4) as u64);
    let mut physical = GpuImage::from_parts(
        r,
        w as u32,
        h as u32,
        4,
        actual.texture().clone(),
        actual.metadata(),
        Default::default(),
    )
    .unwrap();
    let rgba = physical.readback_u8::<4>(r).unwrap();
    for (p, out) in rgba.as_slice().chunks_exact(4).enumerate() {
        assert_eq!(&out[..C], &expected.as_slice()[p * C..(p + 1) * C]);
        assert!(out[C..].iter().all(|&v| v == 0));
    }
    drop(physical);
    source.recycle(r);
    actual.recycle(r);
    previous.recycle(r);
    w * h * C * 3
}

#[test]
fn gaussian_packed_public_calls_match_complete_frozen_gpu_cpu_physical_and_receipts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::FrozenGaussianPipelines::new(r.device());
    let mut checked = 0;
    for (w, h) in [(1, 1), (2, 3), (17, 19), (257, 259)] {
        for (kw, kh) in [(3, 3), (3, 5), (5, 3), (5, 5), (7, 3), (5, 7)] {
            for b in [
                GpuGaussianBorder::Replicate,
                GpuGaussianBorder::Reflect,
                GpuGaussianBorder::Reflect101,
                GpuGaussianBorder::Wrap,
                GpuGaussianBorder::Constant([19, 73, 129, 251]),
            ] {
                let o = GpuGaussianBlurOptions {
                    kernel_width: kw,
                    kernel_height: kh,
                    sigma_x: 0.8,
                    sigma_y: 2.3,
                    border: b,
                };
                checked += compare_frozen::<1>(&r, &old, w, h, o)
                    + compare_frozen::<2>(&r, &old, w, h, o)
                    + compare_frozen::<3>(&r, &old, w, h, o)
                    + compare_frozen::<4>(&r, &old, w, h, o);
            }
        }
    }
    let o = GpuGaussianBlurOptions {
        kernel_width: 3,
        border: GpuGaussianBorder::Constant([255; 4]),
        ..Default::default()
    };
    checked += compare_frozen::<2>(&r, &old, 1025, 1025, o)
        + compare_frozen::<3>(&r, &old, 1025, 1025, o)
        + compare_frozen::<4>(&r, &old, 1025, 1025, o);
    eprintln!("GAUSSIAN_PACKED_PUBLIC verified {checked} complete frozen GPU/CPU/physical bytes; Q8/Q15 and single-channel controls, borders, strips and stride");
}

#[test]
fn gaussian_packed_cold_mixed_pipelines_profile_and_owned_outputs_survive_recycling() {
    let r = std::sync::Arc::new(
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap(),
    );
    let image = fixture::<4>(257, 259, false);
    let source = std::sync::Arc::new(GpuImage::upload_u8(&r, image.view()).unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = [3, 7, 5, 7]
        .into_iter()
        .map(|kernel| {
            let r = r.clone();
            let source = source.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let o = GpuGaussianBlurOptions {
                    kernel_width: kernel,
                    kernel_height: 3,
                    ..Default::default()
                };
                let mut profiler =
                    r.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&r, 2).unwrap());
                gate.wait();
                let output = if let Some(p) = profiler.as_mut() {
                    gaussian_blur_gpu_profiled(&r, &source, o, p).unwrap()
                } else {
                    gaussian_blur_gpu(&r, &source, o).unwrap()
                };
                if let Some(p) = profiler {
                    assert_eq!(
                        p.collect(&r).unwrap().stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                        ["gaussian_horizontal_gpu", "gaussian_blur_gpu"]
                    );
                }
                let cache = r.image_gaussian_pipelines.get().unwrap();
                (output, o, cache.pipelines(r.device(), kernel != 7) as *const _ as usize)
            })
        })
        .collect();
    let mut results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results[0].2, results[2].2);
    assert_eq!(results[1].2, results[3].2);
    assert_ne!(results[0].2, results[1].2);
    std::sync::Arc::try_unwrap(source).ok().unwrap().recycle(&r);
    r.clear_image_texture_pool();
    for (output, o, _) in &mut results {
        let expected = gaussian_blur_u8(
            image.view(),
            o.kernel_width as usize,
            3,
            o.sigma_x,
            o.sigma_y,
            border(o.border),
        )
        .unwrap();
        output.receipt().validate_resident_chain(257 * 259 * 4).unwrap();
        assert_eq!(output.readback_u8::<4>(&r).unwrap(), expected);
    }
}

#[path = "gaussian_packed_benchmark.rs"]
mod benchmark;

#[test]
fn gaussian_packed_rejections_allocate_nothing_and_queued_scratch_reuse_is_ordered() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let other =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let image = fixture::<3>(17, 19, false);
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let o = GpuGaussianBlurOptions::default();
    let texture = r.image_texture_pool_stats();
    let scratch = r.scratch_buffer_pool_stats();
    assert!(gaussian_blur_gpu(&other, &source, o).is_err());
    for options in [
        GpuGaussianBlurOptions { kernel_width: 9, ..o },
        GpuGaussianBlurOptions { sigma_x: f64::NAN, ..o },
        GpuGaussianBlurOptions { sigma_y: 0.0, ..o },
    ] {
        assert!(gaussian_blur_gpu(&r, &source, options).is_err());
    }
    if r.timestamp_queries_enabled() && other.timestamp_queries_enabled() {
        let mut short = GpuStageProfiler::new(&r, 1).unwrap();
        let mut wrong = GpuStageProfiler::new(&other, 2).unwrap();
        assert!(gaussian_blur_gpu_profiled(&r, &source, o, &mut short).is_err());
        assert!(gaussian_blur_gpu_profiled(&r, &source, o, &mut wrong).is_err());
    }
    assert_eq!(r.image_texture_pool_stats(), texture);
    assert_eq!(r.scratch_buffer_pool_stats(), scratch);
    assert!(r.image_gaussian_pipelines.get().is_none());
    let first = gaussian_blur_gpu(&r, &source, o).unwrap();
    let allocations = r.scratch_buffer_pool_stats().allocation_count;
    let mut outputs = vec![(first, image.clone())];
    for value in [0, 255, 19, 73] {
        let image = Image::<u8, 3>::try_new(17, 19, vec![value; 17 * 19 * 3]).unwrap();
        let input = GpuImage::upload_u8(&r, image.view()).unwrap();
        let output = gaussian_blur_gpu(&r, &input, o).unwrap();
        input.recycle(&r);
        outputs.push((output, image));
    }
    assert_eq!(r.scratch_buffer_pool_stats().allocation_count, allocations);
    let cache = r.image_gaussian_pipelines.get().unwrap();
    assert!(cache.packed.get().is_some());
    assert!(cache.pipelines.get().is_none());
    source.recycle(&r);
    r.clear_image_texture_pool();
    r.clear_scratch_buffer_pool();
    for (output, input) in &mut outputs {
        let expected =
            gaussian_blur_u8(input.view(), 5, 5, o.sigma_x, o.sigma_y, border(o.border)).unwrap();
        assert_eq!(output.readback_u8::<3>(&r).unwrap(), expected);
        assert_eq!(output.receipt().gpu_to_gpu_bytes(), 17 * 19 * 12);
    }
}

use super::*;

#[path = "gaussian_rows_reference.rs"]
mod old;

fn row(y: i32, height: usize, border: GpuGaussianBorder) -> Option<usize> {
    let n = height as i32;
    if (0..n).contains(&y) {
        return Some(y as usize);
    }
    match border {
        GpuGaussianBorder::Constant(_) => None,
        GpuGaussianBorder::Replicate => Some(y.clamp(0, n - 1) as usize),
        _ if n == 1 => Some(0),
        GpuGaussianBorder::Wrap => Some(y.rem_euclid(n) as usize),
        GpuGaussianBorder::Reflect => {
            let y = y.rem_euclid(2 * n);
            Some(if y < n { y } else { 2 * n - 1 - y } as usize)
        }
        GpuGaussianBorder::Reflect101 => {
            let y = y.rem_euclid(2 * (n - 1));
            Some(if y < n { y } else { 2 * (n - 1) - y } as usize)
        }
    }
}

fn raw_vertical<const C: usize>(
    r: &WgpuRuntime,
    prior: &old::ParentGaussianPipelines,
    cache: &GaussianPipelines,
    w: usize,
    h: usize,
    options: GpuGaussianBlurOptions,
    hot: bool,
) -> usize {
    let mut p = plan(w as u32, h as u32, C as u32, options, &r.device().limits()).unwrap();

    let mass: u64 = p.params.weights_y.iter().flatten().map(|&w| u64::from(w)).sum();
    assert_eq!(mass, 1u64 << p.params.bits);
    assert!(65535 * mass < (1 << 31));
    assert!(127 * mass < (1 << 22));
    p.params.groups = 1;
    let words = w * h * if p.packed { C.div_ceil(2) } else { C };
    let edges =
        [0, 1, 16383, 16384, 16385, 32767, 32768, 32769, 65535, 65536, 65537, 8_355_840, 8_388_607];
    let mut sums: Vec<u32> = (0..words)
        .map(|i| {
            if hot {
                if p.packed {
                    u32::MAX
                } else {
                    (1u32 << (p.params.bits + 8)) - 1
                }
            } else if p.packed {
                (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
            } else if i % 2 == 0 {
                edges[(i / 2) % edges.len()] & ((1u32 << (p.params.bits + 8)) - 1)
            } else {
                (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453).rotate_left(13)
                    & ((1u32 << (p.params.bits + 8)) - 1)
            }
        })
        .collect();
    // Probe full u16 or 23-bit bounds, including dirty inactive packed halves.
    sums.extend_from_slice(&[0xdead_beef; 8]);
    let input = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-vmirror-raw-input"),
        contents: bytemuck::cast_slice(&sums),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gaussian-vmirror-raw-params"),
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let outputs: [_; 2] = std::array::from_fn(|_| {
        create_texture(r, w as u32, h as u32, "gaussian-vmirror-raw-output")
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in [
        (&prior.pipelines(r.device(), p.packed)[1], &outputs[0]),
        (
            cache.vertical_pipeline(r.device(), &p, &cache.pipelines(r.device(), p.packed)[1]),
            &outputs[1],
        ),
    ] {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian-vmirror-raw-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &output.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    let constant =
        if let GpuGaussianBorder::Constant(pixel) = options.border { pixel } else { [0; 4] };
    let mut expected = vec![0; w * h * 4];
    for pixel in 0..w * h {
        for c in 0..C {
            let total: u64 = (0..options.kernel_height as usize)
                .map(|tap| {
                    let horizontal = row(
                        (pixel / w) as i32 + tap as i32 - (options.kernel_height / 2) as i32,
                        h,
                        options.border,
                    )
                    .map_or(u32::from(constant[c]) << p.params.bits, |sy| {
                        if p.packed {
                            let word = sums[(sy * w + pixel % w) * C.div_ceil(2) + c / 2];
                            (word >> (16 * (c % 2))) & 65535
                        } else {
                            sums[(sy * w + pixel % w) * C + c]
                        }
                    });
                    u64::from(horizontal) * u64::from(p.params.weights_y[tap / 4][tap % 4])
                })
                .sum();
            expected[pixel * 4 + c] =
                ((total + (1u64 << (2 * p.params.bits - 1))) >> (2 * p.params.bits)).min(255) as u8;
        }
    }
    for texture in outputs {
        let mut output = GpuImage::from_parts(
            r,
            w as u32,
            h as u32,
            4,
            texture,
            Default::default(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(output.readback_u8::<4>(r).unwrap().as_slice(), expected);
        output.recycle(r);
    }
    w * h * 8
}

#[test]
fn gaussian_rows_raw_vertical_matches_parent_and_u64_all_layouts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let cache = GaussianPipelines::new();
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for (kw, kh) in [(3, 3), (3, 5), (7, 3), (7, 5), (7, 7)] {
            for border in [
                GpuGaussianBorder::Replicate,
                GpuGaussianBorder::Reflect,
                GpuGaussianBorder::Reflect101,
                GpuGaussianBorder::Wrap,
                GpuGaussianBorder::Constant([19, 73, 129, 251]),
            ] {
                for sigma in [0.17, 7.0] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: kw,
                        kernel_height: kh,
                        sigma_y: sigma,
                        border,
                        ..Default::default()
                    };
                    bytes += raw_vertical::<1>(&r, &prior, &cache, w, h, options, false)
                        + raw_vertical::<2>(&r, &prior, &cache, w, h, options, false)
                        + raw_vertical::<3>(&r, &prior, &cache, w, h, options, false)
                        + raw_vertical::<4>(&r, &prior, &cache, w, h, options, false);
                    cases += 4;
                }
            }
        }
    }
    for kw in [3, 7] {
        let options = GpuGaussianBlurOptions {
            kernel_width: kw,
            border: GpuGaussianBorder::Constant([255; 4]),
            ..Default::default()
        };
        bytes += raw_vertical::<1>(&r, &prior, &cache, 1025, 1025, options, true)
            + raw_vertical::<2>(&r, &prior, &cache, 1025, 1025, options, true)
            + raw_vertical::<3>(&r, &prior, &cache, 1025, 1025, options, true)
            + raw_vertical::<4>(&r, &prior, &cache, 1025, 1025, options, true);
        cases += 4;
    }
    println!("GAUSSIAN_ROWS_RAW verified {bytes} physical RGBA bytes across {cases} paired parent/u64 fixtures, full u16/23-bit inputs, dirty inactive halves, all borders and one-group stride.");
}

fn complete_parent<const C: usize>(r: &WgpuRuntime, prior: &old::ParentGaussianPipelines) -> usize {
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        let image = fixture::<C>(w, h, false);
        let source = GpuImage::upload_u8(r, image.view()).unwrap();
        for kw in [3, 5, 7] {
            for kh in [3, 5, 7] {
                for b in [
                    GpuGaussianBorder::Replicate,
                    GpuGaussianBorder::Reflect,
                    GpuGaussianBorder::Reflect101,
                    GpuGaussianBorder::Wrap,
                    GpuGaussianBorder::Constant([19, 73, 129, 251]),
                ] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: kw,
                        kernel_height: kh,
                        sigma_x: 0.17,
                        sigma_y: 7.,
                        border: b,
                    };
                    let expected = gaussian_blur_u8(
                        image.view(),
                        kw as usize,
                        kh as usize,
                        options.sigma_x,
                        options.sigma_y,
                        border(b),
                    )
                    .unwrap();
                    let mut previous = old::previous(prior, r, &source, options, None).unwrap();
                    let mut current = gaussian_blur_gpu(r, &source, options).unwrap();
                    assert_eq!(previous.receipt(), current.receipt());
                    assert_eq!(previous.metadata(), current.metadata());
                    assert_eq!(previous.readback_u8::<C>(r).unwrap(), expected);
                    assert_eq!(current.readback_u8::<C>(r).unwrap(), expected);
                    assert_eq!(previous.receipt(), current.receipt());
                    previous.recycle(r);
                    current.recycle(r);
                    bytes += w * h * C * 2;
                }
            }
        }
        source.recycle(r);
    }
    bytes
}

#[test]
fn gaussian_rows_complete_parent_cpu_keep_all_channels_borders_and_mixed_axes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let mut bytes = complete_parent::<1>(&r, &prior)
        + complete_parent::<2>(&r, &prior)
        + complete_parent::<3>(&r, &prior)
        + complete_parent::<4>(&r, &prior);
    bytes += large::<2>(&r, &prior, 7, 7, false)
        + large::<3>(&r, &prior, 3, 7, true)
        + large::<4>(&r, &prior, 7, 5, false);
    println!("GAUSSIAN_ROWS_PUBLIC verified {bytes} complete parent/CPU bytes, both scratch layouts, mixed axes, receipts and metadata.");
}

fn large<const C: usize>(
    r: &WgpuRuntime,
    prior: &old::ParentGaussianPipelines,
    kw: u32,
    kh: u32,
    constant: bool,
) -> usize {
    let (w, h) = (1025, 1027);
    let image = fixture::<C>(w, h, false);
    let o = GpuGaussianBlurOptions {
        kernel_width: kw,
        kernel_height: kh,
        border: if constant {
            GpuGaussianBorder::Constant([19, 73, 129, 251])
        } else {
            GpuGaussianBorder::Reflect101
        },
        ..Default::default()
    };
    let expected = gaussian_blur_u8(
        image.view(),
        kw as usize,
        kh as usize,
        o.sigma_x,
        o.sigma_y,
        border(o.border),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut previous = old::previous(prior, r, &source, o, None).unwrap();
    let mut current = gaussian_blur_gpu(r, &source, o).unwrap();
    assert_eq!(previous.receipt(), current.receipt());
    assert_eq!(previous.metadata(), current.metadata());
    assert_eq!(previous.readback_u8::<C>(r).unwrap(), expected);
    assert_eq!(current.readback_u8::<C>(r).unwrap(), expected);
    previous.recycle(r);
    current.recycle(r);
    source.recycle(r);
    w * h * C * 2
}

fn resources<const C: usize>(r: &WgpuRuntime, prior: &old::ParentGaussianPipelines) {
    r.set_scratch_buffer_pool_budget(64 * 1024 * 1024);
    r.set_image_texture_pool_budget(64 * 1024 * 1024);
    let image = fixture::<C>(257, 259, false);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let o = GpuGaussianBlurOptions { kernel_width: 7, kernel_height: 7, ..Default::default() };
    let expected =
        gaussian_blur_u8(image.view(), 7, 7, o.sigma_x, o.sigma_y, border(o.border)).unwrap();
    let mut fresh = Vec::new();
    for current in [false, true] {
        r.clear_scratch_buffer_pool();
        r.clear_image_texture_pool();
        let a = r.scratch_buffer_pool_stats().allocation_count;
        let t = r.image_texture_pool_stats().allocation_count;
        let mut output = if current {
            gaussian_blur_gpu(r, &source, o)
        } else {
            old::previous(prior, r, &source, o, None)
        }
        .unwrap();
        assert_eq!(output.receipt().gpu_to_gpu_bytes(), 257 * 259 * (C as u64 + 1) * 4);
        assert_eq!(output.receipt().device_to_host_bytes(), 0);
        assert_eq!(output.readback_u8::<C>(r).unwrap(), expected);
        output.recycle(r);
        let s = r.scratch_buffer_pool_stats();
        let i = r.image_texture_pool_stats();
        fresh.push((
            s.cached_bytes,
            s.cached_resources,
            s.allocation_count - a,
            i.cached_bytes,
            i.cached_resources,
            i.allocation_count - t,
        ));
    }
    assert_eq!(fresh[0], fresh[1]);
    let warm = (
        r.scratch_buffer_pool_stats().allocation_count,
        r.image_texture_pool_stats().allocation_count,
    );
    let mut output = gaussian_blur_gpu(r, &source, o).unwrap();
    assert_eq!(output.readback_u8::<C>(r).unwrap(), expected);
    output.recycle(r);
    assert_eq!(
        warm,
        (
            r.scratch_buffer_pool_stats().allocation_count,
            r.image_texture_pool_stats().allocation_count
        )
    );
    for budget in [4, 0] {
        r.set_scratch_buffer_pool_budget(budget);
        r.set_image_texture_pool_budget(budget);
        let mut output = gaussian_blur_gpu(r, &source, o).unwrap();
        assert_eq!(output.readback_u8::<C>(r).unwrap(), expected);
        output.recycle(r);
        assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        assert!(r.image_texture_pool_stats().cached_bytes <= budget);
    }
    if r.timestamp_queries_enabled() {
        let mut before = GpuStageProfiler::new(r, 2).unwrap();
        let mut after = GpuStageProfiler::new(r, 2).unwrap();
        let mut previous = old::previous(prior, r, &source, o, Some(&mut before)).unwrap();
        let mut current = gaussian_blur_gpu_profiled(r, &source, o, &mut after).unwrap();
        let before = before.collect(r).unwrap();
        let after = after.collect(r).unwrap();
        assert_eq!(before.device_to_host_bytes, 32);
        assert_eq!(after.device_to_host_bytes, 32);
        assert_eq!(
            before.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            vec!["gaussian_horizontal_gpu", "gaussian_blur_gpu"]
        );
        assert_eq!(
            after.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            vec!["gaussian_horizontal_gpu", "gaussian_blur_gpu"]
        );
        assert_eq!(previous.receipt(), current.receipt());
        assert_eq!(previous.readback_u8::<C>(r).unwrap(), expected);
        assert_eq!(current.readback_u8::<C>(r).unwrap(), expected);
        previous.recycle(r);
        current.recycle(r);
    }
    source.recycle(r);
}

#[test]
fn gaussian_rows_lazy_selection_preserves_controls_resources_profiles_and_preflight() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let prior = old::ParentGaussianPipelines::new();
    let image = fixture::<2>(17, 19, false);
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let o = GpuGaussianBlurOptions { kernel_width: 7, ..Default::default() };
    let scratch = r.scratch_buffer_pool_stats();
    let textures = r.image_texture_pool_stats();
    assert!(
        gaussian_blur_gpu(&r, &source, GpuGaussianBlurOptions { sigma_x: f64::NAN, ..o }).is_err()
    );
    if r.timestamp_queries_enabled() {
        let mut short = GpuStageProfiler::new(&r, 1).unwrap();
        assert!(gaussian_blur_gpu_profiled(&r, &source, o, &mut short).is_err());
    }
    assert_eq!(scratch, r.scratch_buffer_pool_stats());
    assert_eq!(textures, r.image_texture_pool_stats());
    assert!(r.image_gaussian_pipelines.get().is_none());
    let control = gaussian_blur_gpu(&r, &source, GpuGaussianBlurOptions::default()).unwrap();
    r.wait_idle();
    control.recycle(&r);
    let cache = r.image_gaussian_pipelines.get().unwrap();
    assert!(cache.q15_pairs.get().is_none());
    assert!(cache.q15_rows.get().is_none());
    let p = plan(17, 19, 2, GpuGaussianBlurOptions::default(), &r.device().limits()).unwrap();
    let normal = cache.pipelines(r.device(), p.packed);
    assert!(std::ptr::eq(cache.vertical_pipeline(r.device(), &p, &normal[1]), &normal[1]));
    let mono = fixture::<1>(17, 19, false);
    let mono = GpuImage::upload_u8(&r, mono.view()).unwrap();
    let control = gaussian_blur_gpu(&r, &mono, o).unwrap();
    r.wait_idle();
    control.recycle(&r);
    mono.recycle(&r);
    let p = plan(17, 19, 1, o, &r.device().limits()).unwrap();
    let normal = cache.pipelines(r.device(), p.packed);
    assert!(std::ptr::eq(cache.vertical_pipeline(r.device(), &p, &normal[1]), &normal[1]));
    assert!(cache.q15_pairs.get().is_none());
    assert!(cache.q15_rows.get().is_none());
    resources::<2>(&r, &prior);
    assert!(cache.q15_pairs.get().is_some());
    assert!(cache.q15_rows.get().is_none());
    resources::<3>(&r, &prior);
    let shared = cache.q15_rows.get().unwrap() as *const _;
    resources::<4>(&r, &prior);
    assert_eq!(shared, cache.q15_rows.get().unwrap() as *const _);
    r.wait_idle();
    source.recycle(&r);
    println!("GAUSSIAN_ROWS_RESOURCES verified lazy two-/four-lane cache selection, byte-identical Q8/mono shaders and pipeline identity, unchanged scratch/texture requests/warm reuse/zero budgets/two passes/query bytes/receipts and rejection before allocation.");
}

fn worker<const C: usize>(r: &WgpuRuntime, barrier: &std::sync::Barrier) -> (GpuImage, Vec<u8>) {
    let image = fixture::<C>(257, 259, false);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let o = GpuGaussianBlurOptions { kernel_width: 7, kernel_height: 7, ..Default::default() };
    let expected =
        gaussian_blur_u8(image.view(), 7, 7, o.sigma_x, o.sigma_y, border(o.border)).unwrap();
    barrier.wait();
    let output = gaussian_blur_gpu(r, &source, o).unwrap();
    r.wait_idle();
    source.recycle(r);
    (output, expected.as_slice().to_vec())
}

#[test]
fn gaussian_rows_concurrent_first_use_and_retained_outputs_keep_owned_bytes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let barrier = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let a = scope.spawn(|| worker::<2>(&r, &barrier));
        let b = scope.spawn(|| worker::<3>(&r, &barrier));
        let c = scope.spawn(|| worker::<4>(&r, &barrier));
        let d = scope.spawn(|| worker::<1>(&r, &barrier));
        vec![a.join().unwrap(), b.join().unwrap(), c.join().unwrap(), d.join().unwrap()]
    });
    let cache = r.image_gaussian_pipelines.get().unwrap();
    assert!(cache.q15_pairs.get().is_some());
    assert!(cache.q15_rows.get().is_some());
    r.clear_scratch_buffer_pool();
    r.clear_image_texture_pool();
    let input = fixture::<3>(257, 259, true);
    let input = GpuImage::upload_u8(&r, input.view()).unwrap();
    let mut changed = gaussian_blur_gpu(
        &r,
        &input,
        GpuGaussianBlurOptions { kernel_width: 7, kernel_height: 7, ..Default::default() },
    )
    .unwrap();
    assert!(changed.readback_u8::<3>(&r).unwrap().as_slice().iter().all(|&v| v == 255));
    changed.recycle(&r);
    input.recycle(&r);
    for (mut output, expected) in outputs {
        let actual = match output.channels() {
            1 => output.readback_u8::<1>(&r).unwrap().as_slice().to_vec(),
            2 => output.readback_u8::<2>(&r).unwrap().as_slice().to_vec(),
            3 => output.readback_u8::<3>(&r).unwrap().as_slice().to_vec(),
            4 => output.readback_u8::<4>(&r).unwrap().as_slice().to_vec(),
            _ => unreachable!(),
        };
        assert_eq!(actual, expected);
        output.recycle(&r);
    }
    println!("GAUSSIAN_ROWS_CONCURRENT verified four simultaneous cold-cache channel layouts, shared OnceLock initialization, live independent outputs after clear/reuse and changed input.");
}

#[path = "gaussian_rows_benchmark.rs"]
mod benchmark;

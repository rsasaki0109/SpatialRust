use super::*;
#[path = "box_blur_row_prefix_reference.rs"]
mod parent;

fn populated(cache: &BlurPipeline) -> Vec<usize> {
    cache.variants.iter().enumerate().filter_map(|(i, v)| v.get().map(|_| i)).collect()
}

#[test]
fn box_row_prefix_capabilities_shapes_and_integer_bounds() {
    let mut limits = wgpu::Limits::default();
    for memory in [0, 16_383, 16_384, 32_768] {
        limits.max_compute_workgroup_storage_size = memory;
        for w in [0, 1, 511, 512, 513, 4095, 4096, 4097, u32::MAX] {
            for h in [0, 1, 127, 128, 129, u32::MAX] {
                for kw in [1, 257, 259, 4099, max_height(1)] {
                    assert_eq!(
                        row_prefix_enabled(w, h, kw, &limits),
                        (512..=4096).contains(&w) && h >= 128 && kw >= 259 && memory >= 16_384
                    );
                }
            }
        }
    }
    let mut checked = 0;
    for width in 1usize..=4096 {
        let n = width.next_power_of_two();
        let mut values: Vec<u32> =
            (0..n).map(|x| if x < width { ((x * 7919) % 256) as u32 } else { 0 }).collect();
        let expected: Vec<u32> = std::iter::once(0)
            .chain(values.iter().scan(0, |sum, &v| {
                *sum += v;
                Some(*sum)
            }))
            .take(n)
            .collect();
        let mut step = 1;
        while step < n {
            for end in ((2 * step - 1)..n).step_by(2 * step) {
                values[end] += values[end - step];
            }
            step *= 2;
        }
        values[n - 1] = 0;
        step = n / 2;
        while step != 0 {
            for end in ((2 * step - 1)..n).step_by(2 * step) {
                let left = values[end - step];
                let base = values[end];
                values[end - step] = base;
                values[end] = base + left;
            }
            step /= 2;
        }
        assert_eq!(values, expected);
        assert!(values.iter().all(|&v| v <= 4096 * 255));
        checked += n;
    }
    println!("BOX_ROW_PREFIX_PLAN checked {checked} exclusive-prefix values for every width 1..4096; exact capability/shape boundaries, 16KiB storage and bounded u32 tree sums.");
}

fn prefix_raw(
    r: &WgpuRuntime,
    current: &BlurPipeline,
    previous: &parent::ParentBlurPipeline,
    image: &Image<u8, 1>,
    kw: u32,
    border: GpuImageBorder,
    groups: u32,
) -> usize {
    let (w, h) = (image.width(), image.height());
    let rgba = Image::<u8, 4>::try_new(
        w,
        h,
        image.as_slice().iter().flat_map(|&v| [v, 73, 129, 251]).collect(),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, rgba.view()).unwrap();
    let mut p = plan(w as u32, h as u32, kw, 1, border, &r.device().limits()).unwrap();
    p.params.groups = groups;
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let words = w * h;
    let outputs: [_; 2] = std::array::from_fn(|_| {
        r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![0xdead_beefu32; words + 8]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let pipes =
        [&previous.variant(r.device(), 1).pipeline, &current.variant(r.device(), 5).pipeline];
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in pipes.into_iter().zip(&outputs) {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.view()),
                },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    let before = read_words(r, &outputs[0]);
    let after = read_words(r, &outputs[1]);
    assert_eq!(before, after, "{w}x{h},kw={kw},groups={groups},border={border:?}");
    assert_eq!(&after[words..], &[0xdead_beef; 8]);
    for (actual, expected) in after.iter().zip(horizontal_reference(image, kw, border)) {
        assert_eq!(u64::from(*actual), expected);
    }
    source.recycle(r);
    words * 2 + 16
}

#[test]
fn box_row_prefix_raw_parent_u64_words_and_guard_writes_are_exact() {
    let r = WgpuRuntime::new_headless().unwrap();
    let current = BlurPipeline::new();
    let previous = parent::ParentBlurPipeline::new();
    let mut words = 0;
    let mut cases = 0;
    for w in [1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096]
    {
        for h in [1, 3] {
            for hot in [false, true] {
                for kw in [1, 259, 4099, max_height(1)] {
                    for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                        words +=
                            prefix_raw(&r, &current, &previous, &fixture(w, h, hot), kw, border, 1);
                        cases += 1;
                    }
                }
            }
        }
    }
    for groups in [1, 2, 7, 4096] {
        words += prefix_raw(
            &r,
            &current,
            &previous,
            &fixture(513, 129, false),
            511,
            GpuImageBorder::Replicate,
            groups,
        );
        cases += 1;
    }
    println!("BOX_ROW_PREFIX_RAW checked {words} immediate-parent/current/u64/guard words in {cases} fixtures; all scan powers, padding, maximum mass, both borders, source RGBA poison, idle groups and repeated rows per workgroup.");
}

#[test]
fn box_row_prefix_complete_profiles_cache_resources_admission_and_ownership() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = parent::ParentBlurPipeline::new();
    let rejected = GpuImage::upload_u8(&r, fixture(512, 128, false).view()).unwrap();
    let stats = r.scratch_buffer_pool_stats();
    let textures = r.image_texture_pool_stats();
    for (kw, kh) in [(0, 1), (2, 3), (259, 0), (4101, 4101)] {
        let old =
            parent::previous(&previous, &r, &rejected, kw, kh, GpuImageBorder::Replicate, None)
                .err()
                .unwrap()
                .to_string();
        let new = box_blur_gpu(&r, &rejected, kw, kh, GpuImageBorder::Replicate)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(old, new);
    }
    assert!(r.image_blur_pipeline.get().is_none());
    assert_eq!(stats, r.scratch_buffer_pool_stats());
    assert_eq!(textures, r.image_texture_pool_stats());
    rejected.recycle(&r);
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h, kw, kh) in [
        (17, 19, 3, 3),
        (257, 259, 257, 31),
        (511, 129, 259, 9),
        (512, 127, 259, 9),
        (512, 128, 259, 1),
        (513, 129, 259, 31),
        (1025, 129, 511, 9),
        (2049, 129, 4099, 9),
        (4096, 129, 4099, 31),
        (4097, 129, 259, 9),
        (512, 129, 65_537, 1),
        (512, 128, 259, max_height(259)),
    ] {
        for hot in [false, true] {
            for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                let image = fixture(w, h, hot);
                let source = GpuImage::upload_u8(&r, image.view()).unwrap();
                let expected = prefix_oracle(&image, kw, kh, border);
                let mut old =
                    parent::previous(&previous, &r, &source, kw, kh, border, None).unwrap();
                let count = blur_pass_count(w as u32, h as u32, kw, kh);
                let mut profiler = GpuStageProfiler::new(&r, count).unwrap();
                let mut output =
                    box_blur_gpu_with_profiler(&r, &source, kw, kh, border, Some(&mut profiler))
                        .unwrap();
                assert_eq!(old.receipt(), output.receipt());
                assert_eq!(old.metadata(), output.metadata());
                let report = profiler.collect(&r).unwrap();
                assert_eq!(report.stages.len(), count as usize);
                assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
                assert_eq!(old.readback_u8::<1>(&r).unwrap().as_slice(), expected);
                assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
                let mut physical = GpuImage::from_parts(
                    &r,
                    w as u32,
                    h as u32,
                    4,
                    output.texture().clone(),
                    output.metadata(),
                    Default::default(),
                )
                .unwrap();
                for (pixel, &v) in
                    physical.readback_u8::<4>(&r).unwrap().as_slice().chunks_exact(4).zip(&expected)
                {
                    assert_eq!(pixel, [v, 0, 0, 0]);
                }
                drop(physical);
                source.recycle(&r);
                old.recycle(&r);
                output.recycle(&r);
                bytes += w * h * 6;
                cases += 1;
            }
        }
    }
    assert!(r.image_blur_pipeline.get().unwrap().variants.iter().all(|v| v.get().is_some()));
    let source = GpuImage::upload_u8(&r, fixture(512, 128, true).view()).unwrap();
    let identities: Vec<_> = r
        .image_blur_pipeline
        .get()
        .unwrap()
        .variants
        .iter()
        .map(|v| v.get().unwrap() as *const Variant as usize)
        .collect();
    for budget in [crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES, 4, 0] {
        r.set_scratch_buffer_pool_budget(budget);
        r.clear_scratch_buffer_pool();
        for _ in 0..2 {
            let mut out = box_blur_gpu(&r, &source, 259, 3, GpuImageBorder::Replicate).unwrap();
            assert!(out.readback_u8::<1>(&r).unwrap().as_slice().iter().all(|&v| v == 255));
            out.recycle(&r);
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
    }
    assert_eq!(
        identities,
        r.image_blur_pipeline
            .get()
            .unwrap()
            .variants
            .iter()
            .map(|v| v.get().unwrap() as *const Variant as usize)
            .collect::<Vec<_>>()
    );
    source.recycle(&r);
    let fresh = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = [(512, 128, 259), (513, 129, 4099), (257, 259, 257), (17, 19, 3)]
            .into_iter()
            .map(|(w, h, kw)| {
                let r = &fresh;
                let gate = &gate;
                scope.spawn(move || {
                    let image = fixture(w, h, false);
                    let source = GpuImage::upload_u8(r, image.view()).unwrap();
                    gate.wait();
                    let output =
                        box_blur_gpu(r, &source, kw, 3, GpuImageBorder::ConstantZero).unwrap();
                    source.recycle(r);
                    (output, prefix_oracle(&image, kw, 3, GpuImageBorder::ConstantZero))
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    fresh.clear_scratch_buffer_pool();
    fresh.clear_image_texture_pool();
    let source = GpuImage::upload_u8(&fresh, fixture(513, 129, true).view()).unwrap();
    let mut changed = box_blur_gpu(&fresh, &source, 4099, 3, GpuImageBorder::Replicate).unwrap();
    assert!(changed.readback_u8::<1>(&fresh).unwrap().as_slice().iter().all(|&v| v == 255));
    for (mut out, expected) in outputs {
        assert_eq!(out.readback_u8::<1>(&fresh).unwrap().as_slice(), expected);
    }
    println!("BOX_ROW_PREFIX_PUBLIC checked {bytes} complete parent/CPU/RGBA bytes in {cases} fixtures; unchanged receipts/pass/query counts, capability/shape fallback, lazy six-program cache, budgets, errors, concurrent first use and live output ownership.");
}

#[path = "box_blur_row_prefix_benchmark.rs"]
mod benchmark;

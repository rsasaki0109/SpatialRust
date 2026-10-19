use super::*;
#[path = "box_blur_column_prefix_reference.rs"]
mod parent;

fn populated(cache: &BlurPipeline) -> Vec<usize> {
    cache.variants.iter().enumerate().filter_map(|(i, v)| v.get().map(|_| i)).collect()
}

#[test]
fn box_column_prefix_limits_overflow_and_tiled_ownership() {
    let mut limits = wgpu::Limits::default();
    let mut checked = 0;
    for memory in [0, 16_383, 16_384, 32_768] {
        limits.max_compute_workgroup_storage_size = memory;
        for w in [0, 127, 128, 129, u32::MAX] {
            for h in [0, 511, 512, 513, 1024, 4096, 4097, u32::MAX] {
                for kw in [1, 257, 259, 4111, 16_447, 16_449, u32::MAX] {
                    for kh in [1, 257, 259, u32::MAX] {
                        let expected = w >= 128
                            && (512..=4096).contains(&h)
                            && kw >= 259
                            && kh >= 259
                            && memory >= 16_384
                            && u128::from(h) * u128::from(kw) * 255 <= u128::from(u32::MAX);
                        assert_eq!(column_prefix_enabled(w, h, kw, kh, &limits), expected);
                        checked += 1;
                    }
                }
            }
        }
    }
    // Scan exactly the shader's interleaved layout, including all padded heights.
    let mut words = 0;
    for h in 1usize..=4096 {
        let n = h.next_power_of_two();
        let columns = 8.min(4096 / n);
        let max = u32::MAX / 255 / h as u32 * 255;
        let input: Vec<u32> = (0..n * columns)
            .map(|i| if i / columns < h { max - (i % 251) as u32 } else { 0 })
            .collect();
        let mut expected = vec![0u32; n * columns];
        for c in 0..columns {
            let mut total = 0u64;
            for y in 0..n {
                expected[y * columns + c] = u32::try_from(total).unwrap();
                total += u64::from(input[y * columns + c]);
            }
            assert!(total <= u64::from(u32::MAX));
        }
        let mut values = input;
        let mut step = 1;
        while step < n {
            for node in 0..n / (2 * step) * columns {
                let end = ((node / columns + 1) * 2 * step - 1) * columns + node % columns;
                values[end] = values[end].checked_add(values[end - step * columns]).unwrap();
            }
            step *= 2;
        }
        for c in 0..columns {
            values[(n - 1) * columns + c] = 0;
        }
        step = n / 2;
        while step != 0 {
            for node in 0..n / (2 * step) * columns {
                let end = ((node / columns + 1) * 2 * step - 1) * columns + node % columns;
                let left = values[end - step * columns];
                let base = values[end];
                values[end - step * columns] = base;
                values[end] = base.checked_add(left).unwrap();
            }
            step /= 2;
        }
        assert_eq!(values, expected);
        words += values.len();
        for w in [1usize, 7, 8, 9, 127, 128, 129, 4097] {
            for groups in [1usize, 2, 7, 4096] {
                let mut visits = vec![0u8; w];
                for group in 0..groups {
                    let mut base = group * columns;
                    while base < w {
                        for visit in &mut visits[base..(base + columns).min(w)] {
                            *visit += 1;
                        }
                        if w - base <= groups * columns {
                            break;
                        }
                        base += groups * columns;
                    }
                }
                assert!(visits.iter().all(|&v| v == 1));
            }
        }
    }
    println!("BOX_COLUMN_PREFIX_PLAN checked {checked} selection bounds, {words} near-u32-limit interleaved prefixes, all heights 1..4096, tiled column ownership and capped dispatches.");
}

#[allow(clippy::too_many_arguments)]
fn raw_vertical(
    r: &WgpuRuntime,
    current: &BlurPipeline,
    previous: &parent::ParentBlurPipeline,
    w: usize,
    h: usize,
    hot: bool,
    kw: u32,
    kh: u32,
    border: GpuImageBorder,
    groups: u32,
) -> usize {
    let max = kw * 255;
    let mut data: Vec<u32> = (0..w * h)
        .map(|i| {
            if hot {
                max
            } else {
                ((i as u64 * 7919 + (i / w) as u64 * 31) % (u64::from(max) + 1)) as u32
            }
        })
        .collect();
    data.extend([0xdead_beef; 8]);
    let input = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mut p = plan(w as u32, h as u32, kw, kh, border, &r.device().limits()).unwrap();
    p.params.groups = groups;
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let canary =
        Image::<u8, 4>::try_new(w + 2, h + 2, [17, 29, 43, 59].repeat((w + 2) * (h + 2))).unwrap();
    let mut outputs: Vec<_> =
        (0..2).map(|_| GpuImage::upload_u8(r, canary.view()).unwrap()).collect();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, output) in
        [&previous.variant(r.device(), 2).pipeline, &current.variant(r.device(), 6).pipeline]
            .into_iter()
            .zip(&outputs)
    {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output.view()),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    let before = outputs[0].readback_u8::<4>(r).unwrap();
    let after = outputs[1].readback_u8::<4>(r).unwrap();
    assert_eq!(
        before.as_slice(),
        after.as_slice(),
        "{w}x{h},kw={kw},kh={kh},groups={groups},border={border:?}"
    );
    for x in 0..w + 2 {
        let mut prefix = vec![0u64; h + 1];
        if x < w {
            for y in 0..h {
                prefix[y + 1] = prefix[y] + u64::from(data[y * w + x]);
            }
        }
        for y in 0..h + 2 {
            let pixel = &after.as_slice()[(y * (w + 2) + x) * 4..][..4];
            if x >= w || y >= h {
                assert_eq!(pixel, [17, 29, 43, 59]);
                continue;
            }
            let radius = (kh / 2) as usize;
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(h);
            let mut sum = prefix[hi] - prefix[lo];
            if border == GpuImageBorder::Replicate {
                sum += radius.saturating_sub(y) as u64 * u64::from(data[x]);
                sum += radius.saturating_sub(h - 1 - y) as u64 * u64::from(data[(h - 1) * w + x]);
            }
            let area = u64::from(kw) * u64::from(kh);
            assert_eq!(pixel, [((sum + area / 2) / area) as u8, 0, 0, 0]);
        }
    }
    assert_eq!(read_words(r, &input), data);
    for out in outputs {
        out.recycle(r);
    }
    (w + 2) * (h + 2) * 8 + data.len() * 4
}

#[test]
fn box_column_prefix_raw_u32_u64_parent_and_canaries() {
    let r = WgpuRuntime::new_headless().unwrap();
    let current = BlurPipeline::new();
    let previous = parent::ParentBlurPipeline::new();
    let mut bytes = 0;
    let mut cases = 0;
    for h in [1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096]
    {
        for hot in [false, true] {
            for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                for kh in [1, 259, max_height(259)] {
                    bytes += raw_vertical(&r, &current, &previous, 9, h, hot, 259, kh, border, 1);
                    cases += 1;
                }
            }
        }
    }
    for groups in [1, 2, 7, 4096] {
        bytes += raw_vertical(
            &r,
            &current,
            &previous,
            129,
            513,
            false,
            511,
            259,
            GpuImageBorder::Replicate,
            groups,
        );
        cases += 1;
    }
    // The total prefix is close to u32::MAX while every final window is admitted.
    bytes += raw_vertical(
        &r,
        &current,
        &previous,
        3,
        4096,
        true,
        4111,
        259,
        GpuImageBorder::Replicate,
        1,
    );
    cases += 1;
    println!("BOX_COLUMN_PREFIX_RAW checked {bytes} parent/u64/physical RGBA/input-canary bytes in {cases} fixtures; padding, interleaved tails, near-u32 prefixes, both borders and repeated column tiles.");
}

#[test]
fn box_column_prefix_complete_profiles_cache_resources_admission_and_ownership() {
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
        (127, 512, 259, 259),
        (128, 511, 259, 259),
        (128, 512, 259, 257),
        (128, 512, 259, 259),
        (129, 513, 259, 259),
        (513, 513, 511, 511),
        (128, 1024, 16_447, 259),
        (128, 1024, 16_449, 259),
        (128, 4096, 4111, 259),
        (128, 4097, 259, 259),
        (4097, 512, 259, 259),
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
    let source = GpuImage::upload_u8(&r, fixture(128, 512, true).view()).unwrap();
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
            let mut out = box_blur_gpu(&r, &source, 259, 259, GpuImageBorder::Replicate).unwrap();
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
        let jobs: Vec<_> = [(128, 512, 259), (513, 513, 4099), (257, 259, 257), (17, 19, 3)]
            .into_iter()
            .map(|(w, h, kw)| {
                let r = &fresh;
                let gate = &gate;
                scope.spawn(move || {
                    let image = fixture(w, h, false);
                    let source = GpuImage::upload_u8(r, image.view()).unwrap();
                    gate.wait();
                    let output =
                        box_blur_gpu(r, &source, kw, 259, GpuImageBorder::ConstantZero).unwrap();
                    source.recycle(r);
                    (output, prefix_oracle(&image, kw, 259, GpuImageBorder::ConstantZero))
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
    println!("BOX_COLUMN_PREFIX_PUBLIC checked {bytes} complete parent/CPU/RGBA bytes in {cases} fixtures; unchanged receipts/pass/query counts, capability/shape fallback, lazy seven-program cache, budgets, errors, concurrent first use and live output ownership.");
}

#[path = "box_blur_column_prefix_benchmark.rs"]
mod benchmark;

use super::*;

#[path = "box_blur_window_reference.rs"]
mod old;

fn horizontal_reference(image: &Image<u8, 1>, kw: u32, border: GpuImageBorder) -> Vec<u64> {
    let (w, h) = (image.width(), image.height());
    let radius = (kw / 2) as usize;
    let mut result = vec![0; w * h];
    let mut prefix = vec![0_u64; w + 1];
    for y in 0..h {
        for x in 0..w {
            prefix[x + 1] = prefix[x] + u64::from(image.as_slice()[y * w + x]);
        }
        for x in 0..w {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(w);
            let mut sum = prefix[hi] - prefix[lo];
            if border == GpuImageBorder::Replicate {
                sum += radius.saturating_sub(x) as u64 * u64::from(image.as_slice()[y * w]);
                sum += radius.saturating_sub(w - 1 - x) as u64
                    * u64::from(image.as_slice()[(y + 1) * w - 1]);
            }
            result[y * w + x] = sum;
        }
    }
    result
}

fn prefix_oracle(image: &Image<u8, 1>, kw: u32, kh: u32, border: GpuImageBorder) -> Vec<u8> {
    let (w, h) = (image.width(), image.height());
    let radius = (kh / 2) as usize;
    let horizontal = horizontal_reference(image, kw, border);
    let mut result = vec![0; w * h];
    let mut prefix = vec![0_u64; h + 1];
    let area = u64::from(kw) * u64::from(kh);
    for x in 0..w {
        for y in 0..h {
            prefix[y + 1] = prefix[y] + horizontal[y * w + x];
        }
        for y in 0..h {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(h);
            let mut sum = prefix[hi] - prefix[lo];
            if border == GpuImageBorder::Replicate {
                sum += radius.saturating_sub(y) as u64 * horizontal[x];
                sum += radius.saturating_sub(h - 1 - y) as u64 * horizontal[(h - 1) * w + x];
            }
            result[y * w + x] = ((sum + area / 2) / area) as u8;
        }
    }
    result
}

fn read_words(r: &WgpuRuntime, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    r.wait_idle();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

fn raw(
    r: &WgpuRuntime,
    current: &BlurPipeline,
    previous: &old::ParentBlurPipeline,
    image: &Image<u8, 1>,
    kw: u32,
    border: GpuImageBorder,
) -> usize {
    let (w, h) = (image.width(), image.height());
    let rgba: Vec<u8> = image.as_slice().iter().flat_map(|&v| [v, 73, 129, 251]).collect();
    let rgba = Image::<u8, 4>::try_new(w, h, rgba).unwrap();
    let source = GpuImage::upload_u8(r, rgba.view()).unwrap();
    let mut p = plan(w as u32, h as u32, kw, 3, border, &r.device().limits()).unwrap();
    p.params.groups = 1;
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let words = (w * h).div_ceil(2);
    let outputs: [_; 2] = std::array::from_fn(|_| {
        r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![0xdead_beefu32; words + 8]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    let pipes =
        [&previous.variant(r.device(), 3).pipeline, &current.variant(r.device(), 3).pipeline];
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
        pass.dispatch_workgroups(1, 1, 1);
    }
    r.queue().submit(Some(encoder.finish()));
    let expected = horizontal_reference(image, kw, border);
    let parent = read_words(r, &outputs[0]);
    let actual = read_words(r, &outputs[1]);
    assert_eq!(actual, parent, "{w}x{h},kw={kw},border={border:?}");
    assert_eq!(&actual[words..], &[0xdead_beef; 8]);
    for (i, &sum) in expected.iter().enumerate() {
        assert_eq!(
            u64::from((actual[i / 2] >> ((i % 2) * 16)) & 65535),
            sum,
            "{w}x{h},kw={kw},i={i},border={border:?}"
        );
    }
    if w * h % 2 == 1 {
        assert_eq!(actual[words - 1] >> 16, 0);
    }
    source.recycle(r);
    w * h * 2 + 16
}

#[test]
fn box_window_raw_parent_u64_words_cover_edges_crossings_tails_and_stride() {
    let r = WgpuRuntime::new_headless().unwrap();
    let current = BlurPipeline::new();
    let previous = old::ParentBlurPipeline::new();
    let mut words = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (2, 3), (7, 1), (3, 257), (17, 19), (256, 257), (257, 259)] {
        for pattern in 0..3 {
            let image = if pattern == 2 {
                Image::<u8, 1>::try_new(
                    w,
                    h,
                    (0..w * h)
                        .map(|i| if i % w == 0 || i % w == w - 1 { 255 } else { 0 })
                        .collect(),
                )
                .unwrap()
            } else {
                fixture(w, h, pattern == 1)
            };
            for kw in [1, 3, 5, 9, 31, 255, 257] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    words += raw(&r, &current, &previous, &image, kw, border);
                    cases += 1;
                }
            }
        }
    }
    for hot in [false, true] {
        for kw in [9, 31] {
            for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                words += raw(&r, &current, &previous, &fixture(1025, 1025, hot), kw, border);
                cases += 1;
            }
        }
    }
    println!("BOX_WINDOW_RAW verified {words} parent/current/u64 horizontal/guard words across {cases} cases; full u16 mass, nonzero unused source channels, both borders, one-tap/row crossings/odd tails and one-group stride.");
}

fn complete(
    r: &WgpuRuntime,
    previous: &old::ParentBlurPipeline,
    image: &Image<u8, 1>,
    kw: u32,
    kh: u32,
    border: GpuImageBorder,
) -> usize {
    let expected = prefix_oracle(image, kw, kh, border);
    if image.width() <= 17 && image.height() <= 19 && u64::from(kw) * u64::from(kh) <= 1024 {
        assert_eq!(expected, oracle(image, kw, kh, border));
    }
    let input = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut parent = old::previous(previous, r, &input, kw, kh, border, None).unwrap();
    let mut current = box_blur_gpu(r, &input, kw, kh, border).unwrap();
    assert_eq!(parent.receipt(), current.receipt());
    assert_eq!(parent.metadata(), current.metadata());
    assert_eq!(parent.readback_u8::<1>(r).unwrap().as_slice(), expected);
    assert_eq!(current.readback_u8::<1>(r).unwrap().as_slice(), expected);
    assert_eq!(parent.receipt(), current.receipt());
    let mut physical = GpuImage::from_parts(
        r,
        image.width() as u32,
        image.height() as u32,
        4,
        current.texture().clone(),
        current.metadata(),
        Default::default(),
    )
    .unwrap();
    for (pixel, &value) in
        physical.readback_u8::<4>(r).unwrap().as_slice().chunks_exact(4).zip(&expected)
    {
        assert_eq!(pixel, [value, 0, 0, 0]);
    }
    drop(physical);
    input.recycle(r);
    parent.recycle(r);
    current.recycle(r);
    image.width() * image.height() * 6
}

#[test]
fn box_window_complete_parent_prefix_brute_outputs_receipts_and_metadata_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let previous = old::ParentBlurPipeline::new();
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            for (kw, kh) in [
                (1, 1),
                (3, 3),
                (9, 9),
                (1, 81),
                (3, 27),
                (31, 33),
                (255, 3),
                (257, 3),
                (259, 3),
                (257, 257),
            ] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    bytes += complete(&r, &previous, &image, kw, kh, border);
                    cases += 1;
                }
            }
        }
    }
    for (kw, kh, border) in [
        (9, 9, GpuImageBorder::Replicate),
        (31, 33, GpuImageBorder::ConstantZero),
        (1, 81, GpuImageBorder::Replicate),
    ] {
        bytes += complete(&r, &previous, &fixture(1025, 1025, false), kw, kh, border);
        cases += 1;
    }
    bytes += complete(&r, &previous, &fixture(1, 1, true), 4099, 4099, GpuImageBorder::Replicate);
    cases += 1;
    println!("BOX_WINDOW_FULL verified {bytes} complete parent/CPU/physical output bytes across {cases} cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.");
}

#[test]
fn box_window_profiles_resources_budget_reuse_and_rejections_match_parent() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = old::ParentBlurPipeline::new();
    let source = GpuImage::upload_u8(&r, fixture(257, 259, false).view()).unwrap();
    for (kw, kh) in [(3, 3), (9, 9), (1, 81), (257, 3), (259, 3)] {
        let expected = prefix_oracle(&fixture(257, 259, false), kw, kh, GpuImageBorder::Replicate);
        let count = blur_pass_count(257, 259, kw, kh);
        let mut stats = Vec::new();
        let mut reports = Vec::new();
        for current in [false, true] {
            r.clear_scratch_buffer_pool();
            r.clear_image_texture_pool();
            let before = r.scratch_buffer_pool_stats();
            let textures_before = r.image_texture_pool_stats();
            let mut profiler = GpuStageProfiler::new(&r, count).unwrap();
            let mut output = if current {
                box_blur_gpu_with_profiler(
                    &r,
                    &source,
                    kw,
                    kh,
                    GpuImageBorder::Replicate,
                    Some(&mut profiler),
                )
            } else {
                old::previous(
                    &previous,
                    &r,
                    &source,
                    kw,
                    kh,
                    GpuImageBorder::Replicate,
                    Some(&mut profiler),
                )
            }
            .unwrap();
            let after = r.scratch_buffer_pool_stats();
            let texture_after = r.image_texture_pool_stats();
            stats.push((
                after.cached_bytes,
                after.cached_resources,
                after.allocation_count - before.allocation_count,
                texture_after.allocation_count - textures_before.allocation_count,
            ));
            let report = profiler.collect(&r).unwrap();
            assert_eq!(report.device_to_host_bytes, u64::from(count) * 16);
            assert_eq!(report.stages.len(), count as usize);
            reports.push(report.stages.into_iter().map(|s| s.stage).collect::<Vec<_>>());
            assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            output.recycle(&r);
        }
        assert_eq!(stats[0], stats[1]);
        assert_eq!(reports[0], reports[1]);
        let allocations = r.scratch_buffer_pool_stats().allocation_count;
        let textures = r.image_texture_pool_stats().allocation_count;
        let mut warm = box_blur_gpu(&r, &source, kw, kh, GpuImageBorder::Replicate).unwrap();
        assert_eq!(allocations, r.scratch_buffer_pool_stats().allocation_count);
        assert_eq!(textures, r.image_texture_pool_stats().allocation_count);
        assert_eq!(warm.readback_u8::<1>(&r).unwrap().as_slice(), expected);
        warm.recycle(&r);
        for budget in [4, 0] {
            r.set_scratch_buffer_pool_budget(budget);
            let mut output = box_blur_gpu(&r, &source, kw, kh, GpuImageBorder::Replicate).unwrap();
            assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
            output.recycle(&r);
            assert!(r.scratch_buffer_pool_stats().cached_bytes <= budget);
        }
        r.set_scratch_buffer_pool_budget(crate::resource_cache::DEFAULT_GPU_POOL_BUDGET_BYTES);
    }
    let cold = WgpuRuntime::new_headless().unwrap();
    let input = GpuImage::upload_u8(&cold, fixture(17, 19, false).view()).unwrap();
    let before = cold.scratch_buffer_pool_stats();
    let textures = cold.image_texture_pool_stats();
    for (kw, kh) in [(0, 9), (2, 9), (9, 2), (4101, 4101)] {
        let a = old::previous(&previous, &cold, &input, kw, kh, GpuImageBorder::Replicate, None)
            .err()
            .unwrap()
            .to_string();
        let b = box_blur_gpu(&cold, &input, kw, kh, GpuImageBorder::Replicate)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(a, b);
    }
    assert!(cold.image_blur_pipeline.get().is_none());
    assert_eq!(before, cold.scratch_buffer_pool_stats());
    assert_eq!(textures, cold.image_texture_pool_stats());
    assert!(box_blur_gpu(&cold, &source, 9, 9, GpuImageBorder::Replicate).is_err());
    assert!(cold.image_blur_pipeline.get().is_none());
    let invalid =
        GpuImage::upload_u8(&cold, Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap().view())
            .unwrap();
    assert!(box_blur_gpu(&cold, &invalid, 9, 9, GpuImageBorder::Replicate).is_err());
    println!("BOX_WINDOW_RESOURCES verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.");
}

#[test]
fn box_window_concurrent_first_use_retains_outputs_after_clear_and_changed_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = [(1, 81), (9, 9), (257, 3), (259, 3)]
            .into_iter()
            .map(|(kw, kh)| {
                let r = &r;
                let gate = &gate;
                scope.spawn(move || {
                    let image = fixture(257, 259, false);
                    let input = GpuImage::upload_u8(r, image.view()).unwrap();
                    gate.wait();
                    let output =
                        box_blur_gpu(r, &input, kw, kh, GpuImageBorder::ConstantZero).unwrap();
                    input.recycle(r);
                    (output, prefix_oracle(&image, kw, kh, GpuImageBorder::ConstantZero))
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
    });
    r.clear_scratch_buffer_pool();
    r.clear_image_texture_pool();
    let input = GpuImage::upload_u8(&r, fixture(257, 259, true).view()).unwrap();
    let mut changed = box_blur_gpu(&r, &input, 31, 33, GpuImageBorder::Replicate).unwrap();
    assert!(changed.readback_u8::<1>(&r).unwrap().as_slice().iter().all(|&v| v == 255));
    for (mut output, expected) in outputs {
        assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
    }
    let cache = r.image_blur_pipeline.get().unwrap();
    assert!(cache.variants[3].get().is_some() && cache.variants[4].get().is_some());
    assert!(cache.variants[1].get().is_some() && cache.variants[2].get().is_some());
    assert!(cache.variants[0].get().is_none());
    println!("BOX_WINDOW_CONCURRENT verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.");
}

#[path = "box_blur_window_benchmark.rs"]
mod benchmark;

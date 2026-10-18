use super::*;

#[path = "box_blur_wide_vwindow_reference.rs"]
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

fn max_height(kw: u32) -> u32 {
    let mut kh = u32::MAX / 255 / kw;
    if kh % 2 == 0 {
        kh -= 1;
    }
    while u64::from(kw) * u64::from(kh) * 255 + u64::from(kw) * u64::from(kh) / 2
        > u64::from(u32::MAX)
    {
        kh -= 2;
    }
    kh
}

#[test]
fn box_wide_vertical_pair_indices_cover_each_pixel_once_and_keep_signed_delta_bounds() {
    let mut checked = 0;
    for (w, h) in [(1u32, 1u32), (1, 7), (2, 3), (3, 2), (7, 1), (17, 19), (256, 257), (257, 259)] {
        let pixels = w * h;
        let pairs = w * (h / 2 + h % 2);
        let mut visits = vec![0u8; pixels as usize];
        for pair in 0..pairs {
            let x = pair % w;
            let y = pair / w * 2;
            visits[(y * w + x) as usize] += 1;
            if y + 1 < h {
                visits[((y + 1) * w + x) as usize] += 1;
            }
        }
        assert!(visits.iter().all(|&n| n == 1));
        checked += visits.len();
        for stride in [256u32, 1024, 4096 * 256] {
            let mut covered = 0;
            for gid in 0..stride.min(pairs) {
                let mut pair = gid;
                loop {
                    covered += 1;
                    if pairs - pair <= stride {
                        break;
                    }
                    pair = pair.checked_add(stride).unwrap();
                }
            }
            assert_eq!(covered, pairs);
        }
    }
    for (w, h) in [(1, i32::MAX as u32), (i32::MAX as u32, 1), (3, 1_431_655_765), (65_535, 65_537)]
    {
        let pixels = w.checked_mul(h).unwrap();
        let pairs = w.checked_mul(h / 2 + h % 2).unwrap();
        assert!(pairs <= pixels);
        for pair in [0, 1, w - 1, w, pairs - 1].into_iter().filter(|&i| i < pairs) {
            let x = pair % w;
            let y = (pair / w).checked_mul(2).unwrap();
            assert!(y < h);
            assert!(y.checked_mul(w).unwrap().checked_add(x).unwrap() < pixels);
            if y + 1 < h {
                assert!((y + 1).checked_mul(w).unwrap().checked_add(x).unwrap() < pixels);
            }
            checked += 1;
        }
    }
    for kw in [259u32, 511, 4099, 65_537, max_height(1)] {
        let kh = max_height(kw);
        let area = u64::from(kw) * u64::from(kh);
        assert!(area * 255 + area / 2 <= u64::from(u32::MAX));
        let next = u64::from(kw) * u64::from(kh + 2);
        assert!(next * 255 + next / 2 > u64::from(u32::MAX));
        for h in [1u32, 2, 3, 257, 65_537] {
            let radius = kh / 2;
            assert!(u64::from(h - 1) + u64::from(radius) <= i32::MAX as u64);
            if h > 1 {
                let incoming = i64::from(h - 2) + i64::from(radius) + 1;
                assert!(incoming <= i64::from(i32::MAX));
            }
        }
    }
    println!("BOX_WIDE_VERTICAL_PLAN verified {checked} pixel owners and extreme indices, forced grid strides, maximum odd kernels/rounding bounds and signed outgoing/incoming coordinates.");
}

fn raw_shader(source: &str) -> String {
    let binding = "var output_px: texture_storage_2d<rgba8uint, write>;";
    let store = "textureStore(output_px, vec2<i32>(i32(x), i32(y)), vec4<u32>(value, 0u, 0u, 0u));";
    assert_eq!(source.matches(binding).count(), 1);
    assert_eq!(source.matches(store).count(), 1);
    source
        .replace(binding, "var<storage, read_write> sum_output: array<u32>;")
        .replace(store, "sum_output[y * params.width + x] = sum;")
}

fn raw_pipeline(r: &WgpuRuntime, source: &str) -> wgpu::ComputePipeline {
    let shader = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(raw_shader(source).into()),
    });
    r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn vertical_oracle(
    values: &[u32],
    w: usize,
    h: usize,
    kh: u32,
    border: GpuImageBorder,
) -> Vec<u32> {
    let radius = (kh / 2) as usize;
    let mut result = vec![0; w * h];
    let mut prefix = vec![0u64; h + 1];
    for x in 0..w {
        for y in 0..h {
            prefix[y + 1] = prefix[y] + u64::from(values[y * w + x]);
        }
        for y in 0..h {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(h);
            let mut sum = prefix[hi] - prefix[lo];
            if border == GpuImageBorder::Replicate {
                sum += radius.saturating_sub(y) as u64 * u64::from(values[x]);
                sum += radius.saturating_sub(h - 1 - y) as u64 * u64::from(values[(h - 1) * w + x]);
            }
            result[y * w + x] = u32::try_from(sum).unwrap();
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn raw(
    r: &WgpuRuntime,
    pipes: &[wgpu::ComputePipeline; 2],
    w: usize,
    h: usize,
    kw: u32,
    kh: u32,
    pattern: u32,
    border: GpuImageBorder,
) -> usize {
    let max = kw * 255;
    let pixels = w * h;
    let values: Vec<u32> = (0..pixels)
        .map(|i| match pattern {
            1 => max,
            2 => {
                if i / w == 0 || i / w == h - 1 {
                    max
                } else {
                    0
                }
            }
            _ => (i as u32).wrapping_mul(7919).wrapping_add((i / w) as u32 * 31) % (max + 1),
        })
        .collect();
    let mut words = values.clone();
    words.extend([0xdead_beef; 8]);
    let source = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params = BlurParams {
        width: w as u32,
        height: h as u32,
        kernel_width: kw,
        kernel_height: kh,
        border_mode: u32::from(border == GpuImageBorder::ConstantZero),
        groups: 1,
        pixel_count: pixels as u32,
        _pad: 0,
    };
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let outputs: [_; 2] = std::array::from_fn(|_| {
        r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![0xdead_beefu32; pixels + 8]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let mut e = r.device().create_command_encoder(&Default::default());
    for (pipeline, out) in pipes.iter().zip(&outputs) {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            ],
        });
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    r.queue().submit(Some(e.finish()));
    let old = read_words(r, &outputs[0]);
    let new = read_words(r, &outputs[1]);
    let expected = vertical_oracle(&values, w, h, kh, border);
    assert_eq!(old, new, "{w}x{h},kw={kw},kh={kh},pattern={pattern},border={border:?}");
    assert_eq!(&new[..pixels], expected);
    assert_eq!(&new[pixels..], &[0xdead_beef; 8]);
    assert_eq!(read_words(r, &source), words);
    pixels * 3 + 16 + words.len()
}

#[test]
fn box_wide_vertical_raw_u32_sums_match_parent_u64_oracle_guards_and_maximum_rounding() {
    let r = WgpuRuntime::new_headless().unwrap();
    let parent = concat!(
        include_str!("box_blur_params.wgsl"),
        include_str!("box_blur_wide_vwindow_vertical_reference.wgsl")
    );
    let pipes = [raw_pipeline(&r, parent), raw_pipeline(&r, SHADERS[2])];
    let mut words = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (2, 3), (3, 2), (7, 1), (17, 19), (256, 257), (257, 259)] {
        for kw in [259, 511, 4099] {
            for kh in [1, 3, 31, 257] {
                for pattern in 0..3 {
                    for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                        words += raw(&r, &pipes, w, h, kw, kh, pattern, border);
                        cases += 1;
                    }
                }
            }
        }
    }
    for (kw, kh, pattern) in [(259, 31, 0), (4099, 257, 1)] {
        for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
            words += raw(&r, &pipes, 1025, 1025, kw, kh, pattern, border);
            cases += 1;
        }
    }
    for (w, h) in [(1, 3), (2, 3), (17, 2)] {
        for kw in [259, 511, 4099, 65_537, max_height(1)] {
            for pattern in [1, 2] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    words += raw(&r, &pipes, w, h, kw, max_height(kw), pattern, border);
                    cases += 1;
                }
            }
        }
    }
    println!("BOX_WIDE_VERTICAL_RAW verified {words} parent/current/u64 raw u32 sums and source/output guard words across {cases} cases; full-width input words and dirty guards, maximum rounding-safe kernels, both borders and one-group row-pair stride.");
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
fn box_wide_vertical_complete_parent_prefix_brute_outputs_receipts_and_metadata_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let previous = old::ParentBlurPipeline::new();
    let mut bytes = 0;
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (17, 19), (257, 259)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            for (kw, kh) in [
                (1, 1),
                (3, 3),
                (9, 9),
                (257, 3),
                (257, 257),
                (259, 1),
                (259, 3),
                (259, 31),
                (511, 81),
                (4099, 257),
                (65_537, 3),
                (259, max_height(259)),
            ] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    bytes += complete(&r, &previous, &image, kw, kh, border);
                    cases += 1;
                }
            }
        }
    }
    for (kw, kh, border) in [
        (259, 9, GpuImageBorder::Replicate),
        (511, 33, GpuImageBorder::ConstantZero),
        (4099, 1, GpuImageBorder::Replicate),
    ] {
        bytes += complete(&r, &previous, &fixture(1025, 1025, false), kw, kh, border);
        cases += 1;
    }
    bytes += complete(&r, &previous, &fixture(1, 1, true), 4099, 4099, GpuImageBorder::Replicate);
    cases += 1;
    for kw in [259, 511, 4099, 65_537, max_height(1)] {
        for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
            bytes += complete(&r, &previous, &fixture(2, 3, true), kw, max_height(kw), border);
            cases += 1;
        }
    }
    println!("BOX_WIDE_VERTICAL_FULL verified {bytes} complete parent/CPU/physical output bytes across {cases} cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.");
}

#[test]
fn box_wide_vertical_profiles_resources_budget_reuse_and_rejections_match_parent() {
    let r = WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
        .unwrap();
    let previous = old::ParentBlurPipeline::new();
    let source = GpuImage::upload_u8(&r, fixture(257, 259, false).view()).unwrap();
    for (kw, kh) in [(3, 3), (9, 9), (259, 1), (259, 31), (65_537, 3)] {
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
    for (kw, kh) in [
        (0, 9),
        (2, 9),
        (9, 2),
        (4101, 4101),
        (65_537, max_height(65_537) + 2),
        (max_height(1) + 2, 1),
    ] {
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
    println!("BOX_WIDE_VERTICAL_RESOURCES verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.");
}

#[test]
fn box_wide_vertical_concurrent_first_use_retains_outputs_after_clear_and_changed_reuse() {
    let r = WgpuRuntime::new_headless().unwrap();
    let gate = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = [(257, 3), (259, 1), (511, 31), (4099, 81)]
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
    let mut changed = box_blur_gpu(&r, &input, 65_537, 3, GpuImageBorder::Replicate).unwrap();
    assert!(changed.readback_u8::<1>(&r).unwrap().as_slice().iter().all(|&v| v == 255));
    for (mut output, expected) in outputs {
        assert_eq!(output.readback_u8::<1>(&r).unwrap().as_slice(), expected);
    }
    let cache = r.image_blur_pipeline.get().unwrap();
    assert!(cache.variants[3].get().is_some() && cache.variants[4].get().is_some());
    assert!(cache.variants[1].get().is_some() && cache.variants[2].get().is_some());
    assert!(cache.variants[0].get().is_none());
    println!("BOX_WIDE_VERTICAL_CONCURRENT verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.");
}

#[path = "box_blur_wide_vwindow_benchmark.rs"]
mod benchmark;

use super::*;

#[path = "box_blur_packed_reference.rs"]
mod frozen;

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

#[test]
fn packed_box_blur_planner_keeps_old_errors_and_exact_u16_width_boundary() {
    let limits = wgpu::Limits::default();
    let mut cases = 0;
    for (w, h) in [(1, 1), (1, 7), (17, 19), (257, 259), (1024, 1024)] {
        for (kw, kh) in [
            (1, 1),
            (3, 3),
            (9, 9),
            (1, 81),
            (81, 1),
            (255, 3),
            (257, 3),
            (259, 3),
            (257, 257),
            (4099, 4099),
        ] {
            let p = plan(w, h, kw, kh, GpuImageBorder::Replicate, &limits).unwrap();
            assert_eq!(p.packed, p.separable && kw <= 257);
            assert_eq!(
                p.scratch_bytes,
                if p.packed { (u64::from(w) * u64::from(h)).div_ceil(2) * 4 } else { p.bytes }
            );
            assert_eq!(p.bytes, u64::from(w) * u64::from(h) * 4);
            assert_eq!(p.params._pad, 0);
            assert_eq!(blur_pass_count(w, h, kw, kh), 1 + u32::from(p.separable));
            for field in 0..12 {
                let mut l = limits.clone();
                match field {
                    0 => l.max_compute_invocations_per_workgroup = 255,
                    1 => l.max_compute_workgroup_size_x = 255,
                    2 => l.max_compute_workgroups_per_dimension = 0,
                    3 => l.max_sampled_textures_per_shader_stage = 0,
                    4 => l.max_storage_textures_per_shader_stage = 0,
                    5 => l.max_uniform_buffers_per_shader_stage = 0,
                    6 => l.max_uniform_buffer_binding_size = 31,
                    7 => l.max_storage_buffers_per_shader_stage = 0,
                    8 => l.max_buffer_size = u64::from(w) * u64::from(h) * 4 - 1,
                    9 => l.max_storage_buffer_binding_size = w * h * 4 - 1,
                    10 => l.max_texture_dimension_2d = w.max(h) - 1,
                    _ => l.max_storage_buffer_binding_size = w * h * 4,
                }
                assert_eq!(
                    plan(w, h, kw, kh, GpuImageBorder::Replicate, &l).map(|_| ()),
                    frozen::baseline_plan_result(w, h, kw, kh, GpuImageBorder::Replicate, &l)
                );
                cases += 1;
            }
        }
    }
    assert_eq!(255_u32 * 257, 65535);
    assert_eq!(255_u32 * 259, 66045);
    let huge = wgpu::Limits {
        max_texture_dimension_2d: u32::MAX,
        max_buffer_size: u64::MAX,
        max_storage_buffer_binding_size: u32::MAX,
        ..limits
    };
    for (w, h, kw, kh) in [
        (0, 1, 9, 9),
        (1, 0, 9, 9),
        (1, 1, 0, 9),
        (1, 1, 2, 9),
        (1, 1, 4101, 4101),
        (u32::MAX, u32::MAX, 3, 3),
        (i32::MAX as u32, 1, 5, 3),
    ] {
        assert_eq!(
            plan(w, h, kw, kh, GpuImageBorder::Replicate, &huge).map(|_| ()),
            frozen::baseline_plan_result(w, h, kw, kh, GpuImageBorder::Replicate, &huge)
        );
        cases += 1;
    }
    eprintln!("BLUR_PACKED_PLAN verified {cases} original capacity/error outcomes; width 257/259, odd/even requests and original pass policy");
}

fn read_words(r: &WgpuRuntime, b: &wgpu::Buffer) -> Vec<u32> {
    let slice = b.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |x| send.send(x).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    b.unmap();
    result
}

fn probe(
    r: &WgpuRuntime,
    cache: &BlurPipeline,
    old: &frozen::FrozenBlurPipeline,
    image: &Image<u8, 1>,
    kw: u32,
    border: GpuImageBorder,
) -> usize {
    let (w, h) = (image.width(), image.height());
    let pixels = w * h;
    let words = pixels.div_ceil(2);
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut p = plan(w as u32, h as u32, kw, 3, border, &r.device().limits()).unwrap();
    p.params.groups = 1;
    let make = |name, len| {
        let b = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        r.queue().write_buffer(&b, 0, bytemuck::cast_slice(&vec![0xdead_beef_u32; len]));
        b
    };
    let packed = make("blur-packed-guard", words + 8);
    let previous = make("blur-old-guard", pixels + 8);
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur-packed-probe"),
        size: packed.size() + previous.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let view = source.view();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (pipeline, buffer) in
        [(&cache.variant(r.device(), 3).pipeline, &packed), (old.horizontal(r.device()), &previous)]
    {
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry { binding: 2, resource: buffer.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&packed, 0, &staging, 0, packed.size());
    encoder.copy_buffer_to_buffer(&previous, 0, &staging, packed.size(), previous.size());
    r.queue().submit(Some(encoder.finish()));
    let result = read_words(r, &staging);
    let previous = &result[words + 8..];
    let expected = horizontal_reference(image, kw, border);
    assert_eq!(&result[words..words + 8], &[0xdead_beef; 8]);
    assert_eq!(&previous[pixels..], &[0xdead_beef; 8]);
    for i in 0..pixels {
        let got = (result[i / 2] >> ((i % 2) * 16)) & 65535;
        assert_eq!(got, previous[i]);
        assert_eq!(u64::from(got), expected[i]);
    }
    if pixels % 2 == 1 {
        assert_eq!(result[words - 1] >> 16, 0);
    }
    source.recycle(r);
    pixels * 2
}

#[test]
fn packed_box_blur_horizontal_matches_old_cpu_and_dirty_guards_through_65535() {
    let r = WgpuRuntime::new_headless().unwrap();
    let cache = BlurPipeline::new();
    let old = frozen::FrozenBlurPipeline::new();
    let mut count = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (257, 259)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            for kw in [1, 3, 9, 31, 255, 257] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    count += probe(&r, &cache, &old, &image, kw, border);
                }
            }
        }
    }
    for hot in [false, true] {
        count += probe(&r, &cache, &old, &fixture(1025, 1025, hot), 9, GpuImageBorder::Replicate);
    }
    eprintln!("BLUR_PACKED_RAW verified {count} frozen GPU/CPU intermediate words; guarded/poisoned layouts, odd row crossings/tails, maximum width and one-group stride");
}

fn compare(
    r: &WgpuRuntime,
    old: &frozen::FrozenBlurPipeline,
    image: &Image<u8, 1>,
    kw: u32,
    kh: u32,
    border: GpuImageBorder,
) -> usize {
    let (w, h) = (image.width(), image.height());
    let expected = prefix_oracle(image, kw, kh, border);
    if w <= 33 && h <= 35 && u64::from(kw) * u64::from(kh) <= 10000 {
        assert_eq!(expected, oracle(image, kw, kh, border));
    }
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut previous = frozen::baseline_blur(r, &source, kw, kh, border, None, old).unwrap();
    let mut actual = box_blur_gpu(r, &source, kw, kh, border).unwrap();
    assert_eq!(actual.metadata(), previous.metadata());
    assert_eq!(actual.receipt().stages(), previous.receipt().stages());
    assert_eq!(actual.receipt().host_to_device_bytes(), previous.receipt().host_to_device_bytes());
    assert_eq!(actual.receipt().device_to_host_bytes(), 0);
    let p = plan(w as u32, h as u32, kw, kh, border, &r.device().limits()).unwrap();
    assert_eq!(
        actual.receipt().gpu_to_gpu_bytes(),
        p.bytes + if p.separable { p.scratch_bytes } else { 0 }
    );
    actual.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
    assert_eq!(actual.readback_u8::<1>(r).unwrap().as_slice(), expected);
    assert_eq!(previous.readback_u8::<1>(r).unwrap().as_slice(), expected);
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
    for (pixel, &byte) in
        physical.readback_u8::<4>(r).unwrap().as_slice().chunks_exact(4).zip(&expected)
    {
        assert_eq!(pixel, [byte, 0, 0, 0]);
    }
    drop(physical);
    source.recycle(r);
    actual.recycle(r);
    previous.recycle(r);
    w * h * 3
}

#[test]
fn packed_box_blur_complete_calls_match_frozen_prefix_and_brute_oracles() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::FrozenBlurPipeline::new();
    let mut count = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 19), (33, 35), (257, 259)] {
        for hot in [false, true] {
            let image = fixture(w, h, hot);
            for (kw, kh) in [
                (1, 1),
                (3, 3),
                (9, 9),
                (1, 81),
                (81, 1),
                (255, 3),
                (257, 3),
                (259, 3),
                (3, 257),
                (257, 257),
            ] {
                for border in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    count += compare(&r, &old, &image, kw, kh, border);
                }
            }
        }
    }
    for (kw, kh) in [(9, 9), (31, 33)] {
        count += compare(&r, &old, &fixture(1025, 1025, false), kw, kh, GpuImageBorder::Replicate);
    }
    count += compare(&r, &old, &fixture(1, 1, true), 4099, 4099, GpuImageBorder::Replicate);
    eprintln!("BLUR_PACKED_PUBLIC verified {count} complete frozen GPU/CPU/physical bytes; small/wide controls, width 257/259, borders, odd shapes and grid stride");
}

#[test]
fn packed_box_blur_cold_mixed_variants_profile_and_owned_outputs_survive_recycle() {
    let r = std::sync::Arc::new(
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap(),
    );
    let source =
        std::sync::Arc::new(GpuImage::upload_u8(&r, fixture(257, 259, false).view()).unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = [(9, 9), (259, 3), (257, 3), (3, 3)]
        .into_iter()
        .map(|(kw, kh)| {
            let r = r.clone();
            let source = source.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let passes = blur_pass_count(257, 259, kw, kh);
                let mut profiler = r
                    .timestamp_queries_enabled()
                    .then(|| GpuStageProfiler::new(&r, passes).unwrap());
                gate.wait();
                let result = box_blur_gpu_with_profiler(
                    &r,
                    &source,
                    kw,
                    kh,
                    GpuImageBorder::Replicate,
                    profiler.as_mut(),
                );
                if let Some(p) = profiler.as_mut() {
                    p.finish_recording(&result);
                }
                let output = result.unwrap();
                if let Some(p) = profiler {
                    let stages =
                        p.collect(&r).unwrap().stages.iter().map(|s| s.stage).collect::<Vec<_>>();
                    assert_eq!(
                        stages,
                        if passes == 2 {
                            vec!["box_blur_horizontal_gpu", "box_blur_gpu"]
                        } else {
                            vec!["box_blur_gpu"]
                        }
                    );
                }
                let cache = r.image_blur_pipeline.get().unwrap();
                let index = if kw == 3 {
                    0
                } else if kw > 257 {
                    1
                } else {
                    3
                };
                (output, kw, kh, cache.variant(r.device(), index) as *const _ as usize)
            })
        })
        .collect();
    let mut outputs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(outputs[0].3, outputs[2].3);
    assert_ne!(outputs[0].3, outputs[1].3);
    std::sync::Arc::try_unwrap(source).ok().unwrap().recycle(&r);
    r.clear_image_texture_pool();
    r.clear_scratch_buffer_pool();
    let image = fixture(257, 259, false);
    for (output, kw, kh, _) in &mut outputs {
        output.receipt().validate_resident_chain(257 * 259 * 4).unwrap();
        assert_eq!(
            output.readback_u8::<1>(&r).unwrap().as_slice(),
            prefix_oracle(&image, *kw, *kh, GpuImageBorder::Replicate)
        );
    }
}

#[test]
fn packed_box_blur_invalid_calls_keep_cache_cold_and_queued_reuse_is_ordered() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let image = fixture(33, 35, false);
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let stats = r.scratch_buffer_pool_stats();
    let textures = r.image_texture_pool_stats();
    for (kw, kh) in [(0, 9), (2, 9), (9, 2), (4101, 4101)] {
        assert!(box_blur_gpu(&r, &source, kw, kh, GpuImageBorder::Replicate).is_err());
    }
    assert!(box_blur_gpu(&other, &source, 9, 9, GpuImageBorder::Replicate).is_err());
    assert!(r.image_blur_pipeline.get().is_none());
    assert_eq!(stats, r.scratch_buffer_pool_stats());
    assert_eq!(textures, r.image_texture_pool_stats());
    let first = box_blur_gpu(&r, &source, 9, 9, GpuImageBorder::Replicate).unwrap();
    let allocations = r.scratch_buffer_pool_stats().allocation_count;
    let mut outputs = vec![(first, image)];
    for v in [0, 255, 19, 73] {
        let image = Image::<u8, 1>::try_new(33, 35, vec![v; 33 * 35]).unwrap();
        let input = GpuImage::upload_u8(&r, image.view()).unwrap();
        let output = box_blur_gpu(&r, &input, 9, 9, GpuImageBorder::Replicate).unwrap();
        input.recycle(&r);
        outputs.push((output, image));
    }
    assert_eq!(r.scratch_buffer_pool_stats().allocation_count, allocations);
    let cache = r.image_blur_pipeline.get().unwrap();
    assert!(cache.variants[..3].iter().all(|v| v.get().is_none()));
    source.recycle(&r);
    r.clear_image_texture_pool();
    r.clear_scratch_buffer_pool();
    for (output, input) in &mut outputs {
        output.receipt().validate_resident_chain(33 * 35 * 4).unwrap();
        assert_eq!(
            output.readback_u8::<1>(&r).unwrap().as_slice(),
            prefix_oracle(input, 9, 9, GpuImageBorder::Replicate)
        );
    }
}

#[path = "box_blur_packed_benchmark.rs"]
mod benchmark;

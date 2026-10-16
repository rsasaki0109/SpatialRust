use super::*;

#[path = "morphology_packed_reference.rs"]
mod frozen;

fn extrema(values: impl Iterator<Item = u8>, op: GpuMorphology) -> u8 {
    values.fold(if op == GpuMorphology::Erode { 255 } else { 0 }, |a, b| {
        if op == GpuMorphology::Erode {
            a.min(b)
        } else {
            a.max(b)
        }
    })
}

fn horizontal_oracle(image: &Image<u8, 1>, kw: u32, op: GpuMorphology) -> Vec<u8> {
    let (w, h, radius) = (image.width(), image.height(), (kw / 2) as usize);
    (0..w * h)
        .map(|i| {
            let x = i % w;
            extrema(
                (x.saturating_sub(radius)..=(x + radius).min(w - 1))
                    .map(|sx| image.as_slice()[i / w * w + sx]),
                op,
            )
        })
        .collect()
}

fn separable_oracle(image: &Image<u8, 1>, kw: u32, kh: u32, op: GpuMorphology) -> Vec<u8> {
    let (w, h, radius) = (image.width(), image.height(), (kh / 2) as usize);
    let rows = horizontal_oracle(image, kw, op);
    (0..w * h)
        .map(|i| {
            let y = i / w;
            extrema(
                (y.saturating_sub(radius)..=(y + radius).min(h - 1)).map(|sy| rows[sy * w + i % w]),
                op,
            )
        })
        .collect()
}

#[test]
fn packed_morphology_planner_preserves_complete_parent_limits_and_row_bounds() {
    let limits = wgpu::Limits::default();
    let mut checked = 0;
    for (w, h) in [
        (1, 1),
        (1, 4097),
        (2, 1025),
        (3, 4315),
        (3, 4316),
        (17, 19),
        (33, 35),
        (257, 259),
        (1025, 1025),
    ] {
        for (kw, kh) in [(1, 1), (3, 3), (9, 9), (31, 31), (1, 31), (31, 1), (3, 27), (27, 3)] {
            for op in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                for field in 0..15 {
                    let mut l = limits.clone();
                    match field {
                        0 => {}
                        1 => l.max_texture_dimension_2d = w.max(h) - 1,
                        2 => l.max_compute_invocations_per_workgroup = 255,
                        3 => l.max_compute_workgroup_size_x = 15,
                        4 => l.max_compute_workgroup_size_y = 15,
                        5 => l.max_compute_workgroups_per_dimension = 0,
                        6 => l.max_sampled_textures_per_shader_stage = 0,
                        7 => l.max_storage_textures_per_shader_stage = 0,
                        8 => l.max_uniform_buffers_per_shader_stage = 0,
                        9 => l.max_uniform_buffer_binding_size = 31,
                        10 => l.max_compute_workgroup_size_z = 0,
                        11 => l.max_buffer_size = 31,
                        12 => l.max_bind_groups = 0,
                        13 => l.max_bindings_per_bind_group = 2,
                        _ => {
                            l.max_storage_buffer_binding_size = 0;
                            l.max_storage_buffers_per_shader_stage = 0;
                        }
                    }
                    let actual = morphology_plan(w, h, kw, kh, op, &l);
                    let previous = frozen::baseline_plan(w, h, kw, kh, op, &l);
                    match actual {
                        Ok(p) => {
                            assert_eq!(
                                previous.unwrap(),
                                (p.separable, p.bytes, bytemuck::bytes_of(&p.params).to_vec())
                            );
                            assert_eq!(p.packed, p.separable && w > 1);
                            assert_eq!(p.scratch_width, if p.packed { w.div_ceil(4) } else { w });
                            assert_eq!(
                                p.scratch_bytes,
                                u64::from(p.scratch_width) * u64::from(h) * 4
                            );
                            assert!(p.scratch_bytes <= p.bytes);
                            assert_eq!(
                                morphology_pass_count(w, h, kw, kh),
                                1 + u32::from(p.separable)
                            );
                        }
                        Err(e) => assert_eq!(e.to_string(), previous.unwrap_err().to_string()),
                    }
                    checked += 1;
                }
            }
        }
    }
    let mut wide = limits;
    wide.max_texture_dimension_2d = u32::MAX;
    wide.max_compute_workgroups_per_dimension = u32::MAX;
    for (w, h, kw, kh) in [
        (0, 1, 3, 3),
        (1, 0, 3, 3),
        (1, 1, 0, 3),
        (1, 1, 2, 3),
        (1, 1, 3, 0),
        (1, 1, 3, 2),
        (1, 1, 33, 3),
        (1, 1, 3, 33),
        (i32::MAX as u32 - 14, 1, 31, 1),
        (i32::MAX as u32 - 13, 1, 31, 1),
        (u32::MAX, u32::MAX, 31, 31),
    ] {
        let actual = morphology_plan(w, h, kw, kh, GpuMorphology::Erode, &wide)
            .map(|p| (p.separable, p.bytes, bytemuck::bytes_of(&p.params).to_vec()));
        let previous = frozen::baseline_plan(w, h, kw, kh, GpuMorphology::Erode, &wide);
        assert_eq!(format!("{actual:?}"), format!("{previous:?}"));
        checked += 1;
    }
    assert!(!morphology_plan(3, 4315, 9, 9, GpuMorphology::Erode, &wide).unwrap().packed);
    assert!(morphology_plan(3, 4316, 9, 9, GpuMorphology::Erode, &wide).unwrap().packed);
    eprintln!("MORPHOLOGY_PACKED_PLAN verified {checked} frozen limits/errors, original thresholds and exact per-row scratch requests");
}

fn poison(r: &WgpuRuntime, texture: &wgpu::Texture, w: u32, h: u32, byte: u8) {
    r.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &vec![byte; w as usize * h as usize * 4],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_horizontal(
    r: &WgpuRuntime,
    cache: &SpatialPipelines,
    pipeline: &wgpu::ComputePipeline,
    source: &GpuImage,
    output: &wgpu::Texture,
    p: &MorphologyPlan,
    packed: bool,
    encoder: &mut wgpu::CommandEncoder,
) {
    let control = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cache.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: control.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source.view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(
        (if packed { p.params.output_width.div_ceil(4) } else { p.params.output_width })
            .div_ceil(16),
        p.params.output_height.div_ceil(16),
        1,
    );
}

fn raw_texture(r: &WgpuRuntime, texture: wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let mut image =
        GpuImage::from_parts(r, w, h, 4, texture, Default::default(), Default::default()).unwrap();
    image.readback_u8::<4>(r).unwrap().as_slice().to_vec()
}

#[test]
fn packed_morphology_horizontal_matches_original_cpu_with_row_tails_and_texture_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let cache = create_spatial_pipelines(r.device());
    let mut checked = 0;
    for (w, h) in [
        (1, 1),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 9),
        (7, 17),
        (8, 19),
        (9, 33),
        (17, 19),
        (33, 35),
        (257, 259),
        (1025, 1025),
    ] {
        for pattern in 0..4 {
            if w > 257 && pattern != 0 {
                continue;
            }
            let image = fixture(w, h, pattern);
            let source = GpuImage::upload_u8(&r, image.view()).unwrap();
            for kw in [1, 3, 9, 31] {
                if w > 257 && ![3, 31].contains(&kw) {
                    continue;
                }
                for op in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                    let p = morphology_plan(w as u32, h as u32, kw, 31, op, &r.device().limits())
                        .unwrap();
                    let packed_w = (w as u32).div_ceil(4);
                    let original =
                        create_texture(&r, w as u32 + 2, h as u32 + 1, "original-horizontal-guard");
                    let packed =
                        create_texture(&r, packed_w + 2, h as u32 + 1, "packed-horizontal-guard");
                    poison(&r, &original, w as u32 + 2, h as u32 + 1, 0xcd);
                    poison(&r, &packed, packed_w + 2, h as u32 + 1, 0xcd);
                    let mut encoder = r.device().create_command_encoder(&Default::default());
                    encode_horizontal(
                        &r,
                        &cache,
                        &cache.separable_morphology(r.device())[0],
                        &source,
                        &original,
                        &p,
                        false,
                        &mut encoder,
                    );
                    encode_horizontal(
                        &r,
                        &cache,
                        &cache.packed_morphology(r.device())[0],
                        &source,
                        &packed,
                        &p,
                        true,
                        &mut encoder,
                    );
                    r.queue().submit(Some(encoder.finish()));
                    let original = raw_texture(&r, original, w as u32 + 2, h as u32 + 1);
                    let packed = raw_texture(&r, packed, packed_w + 2, h as u32 + 1);
                    let expected = horizontal_oracle(&image, kw, op);
                    for y in 0..h {
                        for x in 0..w {
                            assert_eq!(original[(y * (w + 2) + x) * 4], expected[y * w + x]);
                            assert_eq!(
                                &original[(y * (w + 2) + x) * 4 + 1..(y * (w + 2) + x + 1) * 4],
                                &[0; 3]
                            );
                            assert_eq!(
                                packed[y * (packed_w as usize + 2) * 4 + x],
                                expected[y * w + x]
                            );
                        }
                        let row = y * (packed_w as usize + 2) * 4;
                        assert!(packed[row + w..row + packed_w as usize * 4]
                            .iter()
                            .all(|b| *b == 0));
                        assert!(packed
                            [row + packed_w as usize * 4..row + (packed_w as usize + 2) * 4]
                            .iter()
                            .all(|b| *b == 0xcd));
                        assert!(original[(y * (w + 2) + w) * 4..(y + 1) * (w + 2) * 4]
                            .iter()
                            .all(|b| *b == 0xcd));
                    }
                    assert!(packed[h * (packed_w as usize + 2) * 4..].iter().all(|b| *b == 0xcd));
                    assert!(original[h * (w + 2) * 4..].iter().all(|b| *b == 0xcd));
                    checked += w * h * 3;
                }
            }
            source.recycle(&r);
        }
    }
    eprintln!("MORPHOLOGY_PACKED_HORIZONTAL verified {checked} original GPU/CPU/packed bytes, zero row tails and poisoned suffix columns/row");
}

fn compare(
    r: &WgpuRuntime,
    old: &frozen::FrozenSpatialPipelines,
    image: &Image<u8, 1>,
    kw: u32,
    kh: u32,
    op: GpuMorphology,
) -> usize {
    let (w, h) = (image.width(), image.height());
    let expected = separable_oracle(image, kw, kh, op);
    if w * h < 2000 {
        assert_eq!(expected, oracle(image, kw, kh, op));
    }
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut previous = frozen::baseline_morphology(r, &source, kw, kh, op, None, old).unwrap();
    let mut actual = morphology_gpu(r, &source, kw, kh, op).unwrap();
    let p = morphology_plan(w as u32, h as u32, kw, kh, op, &r.device().limits()).unwrap();
    assert_eq!(actual.receipt().stages(), previous.receipt().stages());
    assert_eq!(actual.metadata(), previous.metadata());
    assert_eq!(actual.receipt().host_to_device_bytes(), previous.receipt().host_to_device_bytes());
    assert_eq!(actual.receipt().device_to_host_bytes(), 0);
    assert_eq!(
        actual.receipt().gpu_to_gpu_bytes(),
        p.bytes + if p.separable { p.scratch_bytes } else { 0 }
    );
    assert_eq!(previous.receipt().gpu_to_gpu_bytes(), p.bytes * (1 + u64::from(p.separable)));
    actual.receipt().validate_resident_chain(p.bytes).unwrap();
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
    for (got, want) in
        physical.readback_u8::<4>(r).unwrap().as_slice().chunks_exact(4).zip(&expected)
    {
        assert_eq!(got, &[*want, 0, 0, 0]);
    }
    drop(physical);
    source.recycle(r);
    actual.recycle(r);
    previous.recycle(r);
    w * h * 6
}

#[test]
fn packed_morphology_public_calls_match_complete_parent_rectangle_and_separable_oracles() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::create_spatial_pipelines(r.device());
    let mut checked = 0;
    for (w, h) in [
        (1, 1),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 9),
        (17, 19),
        (33, 35),
        (257, 259),
        (1, 4097),
        (2, 1025),
        (3, 1025),
        (4, 1025),
        (5, 1025),
        (3, 4315),
        (3, 4316),
    ] {
        for pattern in 0..4 {
            for (kw, kh) in [(1, 1), (3, 3), (9, 9), (31, 31), (1, 31), (31, 1), (3, 27), (27, 3)] {
                for op in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                    checked += compare(&r, &old, &fixture(w, h, pattern), kw, kh, op);
                }
            }
        }
    }
    for pattern in [0, 2] {
        for (kw, kh) in [(3, 3), (9, 9), (31, 31)] {
            for op in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                checked += compare(&r, &old, &fixture(1025, 1025, pattern), kw, kh, op);
            }
        }
    }
    eprintln!("MORPHOLOGY_PACKED_PUBLIC verified {checked} complete frozen GPU/CPU/physical bytes, original threshold and single-column controls");
}

#[test]
fn packed_morphology_cold_mixed_variants_share_cache_and_keep_profiled_owned_outputs() {
    let r = std::sync::Arc::new(
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap(),
    );
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = [(257, 259, 9, 9), (257, 259, 31, 31), (1, 4097, 31, 31), (17, 19, 3, 3)]
        .into_iter()
        .map(|(w, h, kw, kh)| {
            let r = r.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                let image = fixture(w, h, 0);
                let source = GpuImage::upload_u8(&r, image.view()).unwrap();
                let passes = morphology_pass_count(w as u32, h as u32, kw, kh);
                let mut profiler = r
                    .timestamp_queries_enabled()
                    .then(|| crate::GpuStageProfiler::new(&r, passes).unwrap());
                gate.wait();
                let result = morphology_gpu_with_profiler(
                    &r,
                    &source,
                    kw,
                    kh,
                    GpuMorphology::Dilate,
                    profiler.as_mut(),
                );
                if let Some(profiler) = profiler.as_mut() {
                    profiler.finish_recording(&result);
                }
                let output = result.unwrap();
                if let Some(profiler) = profiler {
                    let report = profiler.collect(&r).unwrap();
                    assert_eq!(report.device_to_host_bytes, 16 * u64::from(passes));
                    assert_eq!(
                        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
                        if passes == 2 {
                            vec!["morphology_horizontal_gpu", "morphology_gpu"]
                        } else {
                            vec!["morphology_gpu"]
                        }
                    );
                }
                let cache = r.image_spatial_pipelines.get().unwrap();
                let pointer = if w == 1 {
                    cache.separable_morphology(r.device()) as *const _ as usize
                } else if passes == 2 {
                    cache.packed_morphology(r.device()) as *const _ as usize
                } else {
                    &cache.morphology as *const _ as usize
                };
                source.recycle(&r);
                (output, image, kw, kh, pointer)
            })
        })
        .collect();
    let mut outputs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(outputs[0].4, outputs[1].4);
    assert_ne!(outputs[0].4, outputs[2].4);
    assert_ne!(outputs[0].4, outputs[3].4);
    r.clear_image_texture_pool();
    for (output, image, kw, kh, _) in &mut outputs {
        output
            .receipt()
            .validate_resident_chain((image.width() * image.height() * 4) as u64)
            .unwrap();
        assert_eq!(
            output.readback_u8::<1>(&r).unwrap().as_slice(),
            separable_oracle(image, *kw, *kh, GpuMorphology::Dilate)
        );
    }
}

#[test]
fn packed_morphology_recycles_actual_extent_and_keeps_queued_changed_inputs_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    let image = fixture(33, 35, 0);
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let before = r.image_texture_pool_stats();
    for (kw, kh) in [(0, 31), (2, 31), (31, 0), (31, 2), (33, 31)] {
        assert!(morphology_gpu(&r, &source, kw, kh, GpuMorphology::Erode).is_err());
    }
    assert_eq!(r.image_texture_pool_stats(), before);
    assert!(r.image_spatial_pipelines.get().is_none());
    source.recycle(&r);
    let mut outputs = Vec::new();
    for (w, pattern) in [(33, 0), (34, 1), (35, 2), (36, 3), (33, 2)] {
        let image = fixture(w, 35, pattern);
        let source = GpuImage::upload_u8(&r, image.view()).unwrap();
        let output = morphology_gpu(&r, &source, 31, 31, GpuMorphology::Erode).unwrap();
        source.recycle(&r);
        let before = r.image_texture_pool_stats();
        let scratch = create_texture(&r, 9, 35, "probe-packed-recycled-extent");
        assert_eq!(r.image_texture_pool_stats().allocation_count, before.allocation_count);
        assert_eq!((scratch.width(), scratch.height()), (9, 35));
        poison(&r, &scratch, 9, 35, 0xcd);
        r.recycle_image_texture(9, 35, scratch);
        assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * 35 * 4 + 9 * 35 * 4) as u64);
        outputs.push((output, image));
    }
    r.clear_image_texture_pool();
    let source = GpuImage::upload_u8(&r, image.view()).unwrap();
    let p = morphology_plan(33, 35, 31, 31, GpuMorphology::Erode, &r.device().limits()).unwrap();
    let scratch = create_texture(&r, 9, 35, "abandoned-packed-scratch");
    let uniform = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encode_morphology(
        &r,
        r.image_spatial_pipelines.get().unwrap(),
        &p,
        &source.view(),
        &outputs[0].0.view(),
        Some(&scratch.create_view(&Default::default())),
        &uniform,
        &mut encoder,
        None,
    );
    drop(encoder);
    drop(scratch);
    assert_eq!(r.image_texture_pool_stats().cached_resources, 0);
    drop(source);
    for (output, image) in &mut outputs {
        output
            .receipt()
            .validate_resident_chain((image.width() * image.height() * 4) as u64)
            .unwrap();
        assert_eq!(
            output.readback_u8::<1>(&r).unwrap().as_slice(),
            separable_oracle(image, 31, 31, GpuMorphology::Erode)
        );
    }
}

#[path = "morphology_packed_benchmark.rs"]
mod benchmark;

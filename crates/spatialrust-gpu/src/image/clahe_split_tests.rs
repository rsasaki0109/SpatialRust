use super::*;

#[test]
fn split_counts_scan_and_lookup_match_cpu_through_u32_max_mass() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut input = Vec::new();
    let mut expected = Vec::new();
    let mut rows = 0;
    for mass in [1_u32, 255, 256, 257, 65535, u32::MAX] {
        for pattern in 0..4 {
            let mut partials = [0_u32; 2048];
            match pattern {
                0 => partials[0] = mass,
                1 => {
                    for shard in 0..8 {
                        partials[shard * 256 + 255] =
                            mass / 8 + u32::from(shard < (mass % 8) as usize);
                    }
                }
                2 => {
                    for (i, value) in partials.iter_mut().enumerate() {
                        *value = mass / 2048 + u32::from(i < (mass % 2048) as usize);
                    }
                }
                _ => {
                    let mut remaining = u64::from(mass);
                    let mut seed = 19_u32;
                    for i in 0..2047 {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let take = u64::from(seed) % (remaining + 1);
                        partials[(i * 37) % 2048] = take as u32;
                        remaining -= take;
                    }
                    partials[(2047 * 37) % 2048] = remaining as u32;
                }
            }
            let counts: Vec<_> = (0..256)
                .map(|bin| {
                    (0..8).map(|shard| u64::from(partials[shard * 256 + bin])).sum::<u64>() as u32
                })
                .collect();
            assert_eq!(counts.iter().map(|&n| u64::from(n)).sum::<u64>(), u64::from(mass));
            for clip in [0, 1, (mass / 256).max(1), mass] {
                input.extend([mass, clip, 0, 0]);
                input.extend(partials);
                let scanned = reference(&counts, clip);
                for (count, pair) in counts.iter().zip(scanned.chunks_exact(2)) {
                    expected.extend([*count, pair[0], pair[1]]);
                }
                rows += 1;
            }
        }
    }
    let mut partial_words = Vec::new();
    let mut headers = Vec::new();
    for row in input.chunks_exact(2052) {
        headers.extend_from_slice(&row[..2]);
        partial_words.extend_from_slice(&row[4..]);
    }
    let source = format!(
        "{}\n{}",
        split::LOOKUP_SHADER,
        r#"
struct TestInput { area: u32, limit: u32, };
@group(0) @binding(3) var<storage, read> test_inputs: array<TestInput>;
@compute @workgroup_size(256)
fn prove_split_counts(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x;
    let count = merged_count(row, lane, 8u);
    let cumulative = clahe_cumulative(lane, count, test_inputs[row].limit);
    if lane == 0u { scale = reciprocal_info(test_inputs[row].area); }
    workgroupBarrier();
    let start = (row*256u+lane)*3u;
    lookups[start] = count;
    lookups[start+1u] = cumulative;
    lookups[start+2u] = mapped(cumulative, scale);
}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clahe-split-count-proof"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("prove_split_counts"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&headers),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let partials = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&partial_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (expected.len() * 4) as u64;
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: partials.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: input.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(rows, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let actual = read_staging_bytes(runtime.device(), &staging, bytes as usize).unwrap();
    for (i, (&want, word)) in expected.iter().zip(actual.chunks_exact(4)).enumerate() {
        assert_eq!(
            u32::from_le_bytes(word.try_into().unwrap()),
            want,
            "row {}, bin {}, field {}",
            i / 768,
            i % 768 / 3,
            i % 3
        );
    }
    assert_eq!(rows, 96);
    assert_eq!(expected.len(), 73728);
}

#[test]
fn lazy_split_cache_concurrency_moves_and_retained_images_preserve_receipts() {
    use spatialrust_image::Image;
    let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
    let conditions: [(usize, usize, u32, u32); 4] =
        [(257, 259, 1, 1), (257, 257, 1, 1), (17, 19, 8, 8), (257, 259, 3, 5)];
    let images: Vec<_> = conditions
        .iter()
        .enumerate()
        .map(|(pattern, &(w, h, _, _))| {
            Image::<u8, 1>::try_new(
                w,
                h,
                (0..w * h)
                    .map(|i| match pattern {
                        0 => 43,
                        1 => ((i * 37 + i / w * 19) % 256) as u8,
                        2 => 255,
                        _ => ((i / 17) % 2 * 255) as u8,
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let expected: Vec<_> = images
        .iter()
        .zip(conditions)
        .map(|(image, (_, _, x, y))| {
            spatialrust_vision::clahe(image.view(), 2., x as usize, y as usize).unwrap()
        })
        .collect();
    let sources: Vec<_> =
        images.iter().map(|image| GpuImage::upload_u8(&boxed, image.view()).unwrap()).collect();
    assert!(boxed.image_clahe_pipelines.get().is_none());
    let mut retained = std::thread::scope(|scope| {
        let handles: Vec<_> = sources
            .iter()
            .zip(conditions)
            .map(|(source, (_, _, x, y))| {
                let runtime = &boxed;
                scope.spawn(move || clahe_gpu(runtime, source, 2., x, y).unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let runtime = *boxed;
    let pipelines = runtime.image_clahe_pipelines.get().unwrap();
    let has_partials =
        plan(257, 259, 2., 1, 1, &runtime.device().limits()).unwrap().split.is_some();
    assert_eq!(pipelines.split.get().is_some(), has_partials);
    assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    for _ in 0..3 {
        clahe_gpu(&runtime, &sources[0], 2., 1, 1).unwrap().recycle(&runtime);
    }
    let before = runtime.scratch_buffer_pool_stats();
    clahe_gpu(&runtime, &sources[0], 2., 1, 1).unwrap().recycle(&runtime);
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before.allocation_count);
    runtime.set_scratch_buffer_pool_budget(0);
    runtime.set_image_texture_pool_budget(0);
    for ((output, want), (w, h, x, y)) in retained.iter_mut().zip(expected).zip(conditions) {
        let p = plan(w as u32, h as u32, 2., x, y, &runtime.device().limits()).unwrap();
        let mut stages = vec!["upload_u8_texture", "clahe_geometry_gpu"];
        if p.split.is_some() {
            stages.push("clahe_histogram_gpu");
        }
        stages.extend(["clahe_lookup_gpu", "clahe_apply_gpu"]);
        assert_eq!(output.receipt().stages(), stages);
        assert_eq!(
            output.receipt().gpu_to_gpu_bytes(),
            ((w + h) * 16 + w * h * 4) as u64
                + u64::from(x * y) * 1024
                + p.split.as_ref().map_or(0, |p| p.partial_bytes)
        );
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), want.as_slice());
    }
    for output in retained {
        output.recycle(&runtime);
    }
    for source in sources {
        source.recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().cached_bytes, 0);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
    let other = WgpuRuntime::new_headless().unwrap();
    let source = GpuImage::upload_u8(&other, images[2].view()).unwrap();
    clahe_gpu(&other, &source, 2., 8, 8).unwrap().recycle(&other);
    assert!(other.image_clahe_pipelines.get().unwrap().split.get().is_none());
}

#[test]
fn partial_histograms_cover_chunk_lane_padding_empty_chunks_and_group_reuse() {
    use spatialrust_image::Image;
    let runtime = WgpuRuntime::new_headless().unwrap();
    let pipelines = split::Pipelines::new(runtime.device());
    let reflected = |mut p: i64, length: u32| {
        if length == 1 {
            return 0;
        }
        while p < 0 || p >= i64::from(length) {
            p = if p < 0 { -p } else { 2 * (i64::from(length) - 1) - p };
        }
        p as u32
    };
    let mut compared = 0_u64;
    for (w, h) in [
        (1_u32, 1_u32),
        (1, 8192),
        (8192, 1),
        (15, 17),
        (255, 1),
        (256, 1),
        (257, 1),
        (63, 65),
        (64, 64),
        (65, 65),
        (257, 259),
        (1025, 1027),
    ] {
        for hot in [false, true] {
            let image = Image::<u8, 1>::try_new(
                w as usize,
                h as usize,
                (0..w * h)
                    .map(|i| if hot { 43 } else { ((i * 37 + i / w * 19) % 256) as u8 })
                    .collect(),
            )
            .unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let view = source.view();
            for (x, y) in [(1, 1), (3, 5)] {
                let tw = w.div_ceil(x);
                let th = h.div_ceil(y);
                let area = tw * th;
                for chunks in [2, 7, 32] {
                    let p = split::Params {
                        width: w,
                        height: h,
                        tile_width: tw,
                        tile_height: th,
                        tiles_x: x,
                        tile_count: x * y,
                        chunks,
                        partial_groups: 1,
                        clip_count: 0,
                        tile_groups: 1,
                        padding: [0; 2],
                    };
                    let uniform =
                        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::bytes_of(&p),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let dirty = vec![u32::MAX; (x * y * chunks * 256) as usize];
                    let output =
                        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&dirty),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                    let bytes = (dirty.len() * 4) as u64;
                    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: bytes,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipelines.layouts[0],
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output.as_entire_binding(),
                            },
                        ],
                    });
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipelines.pipelines[0]);
                        pass.set_bind_group(0, &group, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
                    runtime.queue().submit(Some(encoder.finish()));
                    let result =
                        read_staging_bytes(runtime.device(), &staging, bytes as usize).unwrap();
                    for tile in 0..x * y {
                        let ox = (tile % x) * tw;
                        let oy = (tile / x) * th;
                        for chunk in 0..chunks {
                            let chunk_size = u64::from(area).div_ceil(u64::from(chunks));
                            let start = u64::from(chunk) * chunk_size;
                            let end = (start + chunk_size).min(u64::from(area));
                            let mut expected = [0_u32; 256];
                            for index in start..end {
                                let px = reflected(i64::from(ox) + index as i64 % i64::from(tw), w);
                                let py = reflected(i64::from(oy) + index as i64 / i64::from(tw), h);
                                expected[image.as_slice()[(py * w + px) as usize] as usize] += 1;
                            }
                            for (bin, want) in expected.into_iter().enumerate() {
                                let at = ((tile * chunks + chunk) * 256) as usize * 4 + bin * 4;
                                assert_eq!(u32::from_le_bytes(result[at..at+4].try_into().unwrap()),want,
                                    "{w}x{h}, grid{x}x{y}, hot{hot}, chunks{chunks}, tile{tile}, chunk{chunk}, bin{bin}");
                                compared += 1;
                            }
                        }
                    }
                }
            }
            source.recycle(&runtime);
        }
    }
    assert_eq!(compared, 4_030_464);
    eprintln!("Verified {compared} partial histogram bins over 144 split layouts");
}

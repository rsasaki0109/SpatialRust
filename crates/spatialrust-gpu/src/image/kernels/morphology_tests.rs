use super::*;
use spatialrust_image::Image;

fn fixture(w: usize, h: usize, pattern: usize) -> Image<u8, 1> {
    Image::try_new(
        w,
        h,
        (0..w * h)
            .map(|i| match pattern {
                0 => ((i * 7919 + i / w * 31) % 256) as u8,
                1 => 0,
                2 => 255,
                _ => {
                    if (i % w + i / w) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                }
            })
            .collect(),
    )
    .unwrap()
}

fn oracle(image: &Image<u8, 1>, kw: u32, kh: u32, operation: GpuMorphology) -> Vec<u8> {
    let (w, h) = (image.width() as i64, image.height() as i64);
    (0..w * h)
        .map(|i| {
            let mut value = if operation == GpuMorphology::Erode { 255 } else { 0 };
            for dy in -(i64::from(kh) / 2)..=i64::from(kh) / 2 {
                for dx in -(i64::from(kw) / 2)..=i64::from(kw) / 2 {
                    let sx = (i % w + dx).clamp(0, w - 1);
                    let sy = (i / w + dy).clamp(0, h - 1);
                    let sample = image.as_slice()[(sy * w + sx) as usize];
                    value = if operation == GpuMorphology::Erode {
                        value.min(sample)
                    } else {
                        value.max(sample)
                    };
                }
            }
            value
        })
        .collect()
}

fn previous_pipeline(runtime: &WgpuRuntime, cache: &SpatialPipelines) -> wgpu::ComputePipeline {
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-morphology"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../tests/fixtures/morphology_previous.wgsl").into(),
        ),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("morphology"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_previous(
    runtime: &WgpuRuntime,
    cache: &SpatialPipelines,
    previous: &wgpu::ComputePipeline,
    p: &MorphologyPlan,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    encoder: &mut wgpu::CommandEncoder,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) {
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cache.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, "previous_morphology"),
    });
    pass.set_pipeline(previous);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(
        p.params.output_width.div_ceil(16),
        p.params.output_height.div_ceil(16),
        1,
    );
}

#[test]
fn exact_extrema_match_frozen_gpu_and_independent_full_rectangle_oracle() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = create_spatial_pipelines(runtime.device());
    let previous = previous_pipeline(&runtime, &cache);
    let mut checked = 0;
    for (w, h) in [(1, 1), (1, 33), (33, 1), (2, 3), (17, 19), (257, 3), (33, 35)] {
        for pattern in 0..4 {
            let image = fixture(w, h, pattern);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            for (kw, kh) in [
                (1, 1),
                (3, 3),
                (7, 7),
                (9, 9),
                (3, 27),
                (27, 3),
                (1, 31),
                (31, 1),
                (15, 31),
                (31, 31),
            ] {
                for operation in [GpuMorphology::Erode, GpuMorphology::Dilate] {
                    let mut p = morphology_plan(
                        w as u32,
                        h as u32,
                        kw,
                        kh,
                        operation,
                        &runtime.device().limits(),
                    )
                    .unwrap();
                    let uniform =
                        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::bytes_of(&p.params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let scratch = create_texture(&runtime, w as u32, h as u32, "dirty-extrema");
                    runtime.queue().write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &scratch,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &vec![255; w * h * 4],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w as u32 * 4),
                            rows_per_image: Some(h as u32),
                        },
                        wgpu::Extent3d {
                            width: w as u32,
                            height: h as u32,
                            depth_or_array_layers: 1,
                        },
                    );
                    let outputs: [_; 3] = std::array::from_fn(|_| {
                        create_texture(&runtime, w as u32, h as u32, "extrema-test")
                    });
                    let mut encoder = runtime.device().create_command_encoder(&Default::default());
                    encode_previous(
                        &runtime,
                        &cache,
                        &previous,
                        &p,
                        &source.view(),
                        &outputs[0].create_view(&Default::default()),
                        &uniform,
                        &mut encoder,
                        None,
                    );
                    for i in 0..2 {
                        // Exercise both mathematical paths independently of selection thresholds.
                        p.separable = i == 1;
                        encode_morphology(
                            &runtime,
                            &cache,
                            &p,
                            &source.view(),
                            &outputs[i + 1].create_view(&Default::default()),
                            Some(&scratch.create_view(&Default::default())),
                            &uniform,
                            &mut encoder,
                            None,
                        );
                    }
                    runtime.queue().submit(Some(encoder.finish()));
                    runtime.recycle_image_texture(w as u32, h as u32, scratch);
                    let expected: Vec<_> = oracle(&image, kw, kh, operation)
                        .into_iter()
                        .flat_map(|value| [value, 0, 0, 0])
                        .collect();
                    for output in outputs {
                        let mut gpu = GpuImage::from_parts(
                            &runtime,
                            w as u32,
                            h as u32,
                            4,
                            output,
                            source.metadata(),
                            Default::default(),
                        )
                        .unwrap();
                        assert_eq!(
                            gpu.readback_u8::<4>(&runtime).unwrap().as_slice(),
                            expected,
                            "{w}x{h}, {kw}x{kh}, {operation:?}, pattern={pattern}"
                        );
                        gpu.recycle(&runtime);
                    }
                    checked += w * h;
                }
            }
            source.recycle(&runtime);
        }
    }
    assert_eq!(checked, 185_760);
}

#[test]
fn morphology_plan_bounds_kernels_coordinates_textures_and_dispatch() {
    let limits = wgpu::Limits::default();
    let valid =
        |w, h, kw, kh, l: &wgpu::Limits| morphology_plan(w, h, kw, kh, GpuMorphology::Erode, l);
    for (w, h, kw, kh) in [
        (0, 1, 3, 3),
        (1, 0, 3, 3),
        (1, 1, 0, 3),
        (1, 1, 2, 3),
        (1, 1, 3, 0),
        (1, 1, 3, 2),
        (1, 1, 33, 3),
        (1, 1, 3, 33),
        (u32::MAX, u32::MAX, 1, 1),
    ] {
        assert!(valid(w, h, kw, kh, &limits).is_err());
    }
    for kw in (1..=31).step_by(2) {
        for kh in (1..=31).step_by(2) {
            assert!(valid(1, 1, kw, kh, &limits).is_ok());
        }
    }
    assert_eq!(morphology_pass_count(12945, 1, 9, 9), 1);
    assert_eq!(morphology_pass_count(12946, 1, 9, 9), 2);
    assert_eq!(morphology_pass_count(u32::MAX, u32::MAX, u32::MAX, u32::MAX), 2);
    assert_eq!(morphology_pass_count(u32::MAX, u32::MAX, 0, 0), 1);
    assert_eq!(morphology_pass_count(3840, 2160, 3, 3), 1);
    let p = valid(33, 35, 31, 31, &limits).unwrap();
    assert!(p.separable);
    assert_eq!(p.bytes, 33 * 35 * 4);
    for field in 0..14 {
        let mut unsupported = limits.clone();
        match field {
            0 => unsupported.max_texture_dimension_2d = 32,
            1 => unsupported.max_compute_invocations_per_workgroup = 255,
            2 => unsupported.max_compute_workgroup_size_x = 15,
            3 => unsupported.max_compute_workgroup_size_y = 15,
            4 => unsupported.max_compute_workgroups_per_dimension = 2,
            5 => unsupported.max_sampled_textures_per_shader_stage = 0,
            6 => unsupported.max_storage_textures_per_shader_stage = 0,
            7 => unsupported.max_uniform_buffers_per_shader_stage = 0,
            8 => unsupported.max_uniform_buffer_binding_size = 31,
            9 => unsupported.max_compute_workgroups_per_dimension = 0,
            10 => unsupported.max_compute_workgroup_size_z = 0,
            11 => unsupported.max_buffer_size = 31,
            12 => unsupported.max_bind_groups = 0,
            _ => unsupported.max_bindings_per_bind_group = 2,
        }
        assert!(valid(33, 35, 9, 9, &unsupported).is_err(), "limit {field}");
    }
    let mut huge = limits.clone();
    huge.max_texture_dimension_2d = u32::MAX;
    huge.max_compute_workgroups_per_dimension = u32::MAX;
    assert!(valid(i32::MAX as u32 - 14, 1, 31, 1, &huge).is_ok());
    assert!(valid(i32::MAX as u32 - 13, 1, 31, 1, &huge).is_err());
    assert!(valid(i32::MAX as u32 + 1, 1, 1, 1, &huge).is_err());
    // Texture scratch adds no storage-buffer or buffer-binding requirement.
    huge.max_buffer_size = 32;
    huge.max_storage_buffer_binding_size = 0;
    huge.max_storage_buffers_per_shader_stage = 0;
    assert!(valid(33, 35, 31, 31, &huge).is_ok());
}

#[test]
fn invalid_public_morphology_inputs_do_not_allocate_or_initialize_cache() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = fixture(2, 3, 0);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let color = Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap();
    let color = GpuImage::upload_u8(&runtime, color.view()).unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let before = runtime.image_texture_pool_stats();
    for (kw, kh) in [(0, 3), (2, 3), (3, 0), (3, 2), (33, 3), (3, 33)] {
        assert!(morphology_gpu(&runtime, &source, kw, kh, GpuMorphology::Erode).is_err());
    }
    assert!(morphology_gpu(&runtime, &color, 3, 3, GpuMorphology::Erode).is_err());
    assert!(morphology_gpu(&other, &source, 3, 3, GpuMorphology::Erode).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert_eq!(other.image_texture_pool_stats().allocation_count, 0);
    assert!(runtime.image_spatial_pipelines.get().is_none());
    assert!(other.image_spatial_pipelines.get().is_none());
}

#[test]
fn strided_roi_metadata_and_retained_concurrent_outputs_stay_exact() {
    use spatialrust_image::{ColorRange, ColorSpace, ImageMetadata, ImageRegion, ImageView};
    let runtime = WgpuRuntime::new_headless().unwrap();
    let stride = 41;
    let data: Vec<_> = (0..stride * 39).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let metadata = ImageMetadata {
        color_space: ColorSpace::Gray,
        color_range: ColorRange::Limited,
        ..Default::default()
    };
    let view = ImageView::<u8, 1>::new_with_metadata(37, 39, stride, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(3, 2, 33, 35))
        .unwrap();
    let packed = Image::<u8, 1>::try_new(
        33,
        35,
        (0..35)
            .flat_map(|y| data[(y + 2) * stride + 3..(y + 2) * stride + 36].iter().copied())
            .collect(),
    )
    .unwrap();
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|i| {
                let runtime = &runtime;
                let source = &source;
                scope.spawn(move || {
                    morphology_gpu(
                        runtime,
                        source,
                        31,
                        31,
                        if i % 2 == 0 { GpuMorphology::Erode } else { GpuMorphology::Dilate },
                    )
                    .unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    let runtime = Box::new(runtime);
    for (i, mut output) in outputs.into_iter().enumerate() {
        assert_eq!(output.metadata(), metadata);
        assert_eq!(output.receipt().host_to_device_bytes(), 33 * 35 * 4);
        assert_eq!(output.receipt().gpu_to_gpu_bytes(), 33 * 35 * 8);
        assert_eq!(output.receipt().device_to_host_bytes(), 0);
        assert_eq!(
            &output.receipt().stages()[1..],
            ["morphology_horizontal_gpu", "morphology_gpu"]
        );
        assert_eq!(
            output.readback_u8::<1>(&runtime).unwrap().as_slice(),
            oracle(
                &packed,
                31,
                31,
                if i % 2 == 0 { GpuMorphology::Erode } else { GpuMorphology::Dilate }
            )
        );
        output.recycle(&runtime);
    }
    let before = runtime.image_texture_pool_stats();
    // Rapid submission/reuse without intervening readback exercises queue ordering.
    for i in 0..20 {
        let result = morphology_gpu(
            &runtime,
            &source,
            31,
            31,
            if i % 2 == 0 { GpuMorphology::Erode } else { GpuMorphology::Dilate },
        )
        .unwrap();
        result.recycle(&runtime);
    }
    assert_eq!(runtime.image_texture_pool_stats().allocation_count, before.allocation_count);
    assert!(runtime.image_spatial_pipelines.get().unwrap().separable_morphology.get().is_some());
    runtime.set_image_texture_pool_budget(33 * 35 * 4);
    assert!(runtime.image_texture_pool_stats().cached_bytes <= 33 * 35 * 4);
    runtime.set_image_texture_pool_budget(0);
    let mut result = morphology_gpu(&runtime, &source, 31, 31, GpuMorphology::Erode).unwrap();
    assert_eq!(
        result.readback_u8::<1>(&runtime).unwrap().as_slice(),
        oracle(&packed, 31, 31, GpuMorphology::Erode)
    );
    result.recycle(&runtime);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
}

#[path = "morphology_benchmark.rs"]
mod benchmark;

#[test]
fn small_path_stays_single_pass_and_abandoned_scratch_is_not_pooled() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = fixture(17, 19, 0);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut result = morphology_gpu(&runtime, &source, 9, 9, GpuMorphology::Dilate).unwrap();
    assert_eq!(result.receipt().gpu_to_gpu_bytes(), 17 * 19 * 4);
    assert_eq!(&result.receipt().stages()[1..], ["morphology_gpu"]);
    let cache = runtime.image_spatial_pipelines.get().unwrap();
    assert!(cache.separable_morphology.get().is_none());
    assert_eq!(
        result.readback_u8::<1>(&runtime).unwrap().as_slice(),
        oracle(&image, 9, 9, GpuMorphology::Dilate)
    );
    let mut p =
        morphology_plan(17, 19, 9, 9, GpuMorphology::Erode, &runtime.device().limits()).unwrap();
    p.separable = true;
    let scratch = create_texture(&runtime, 17, 19, "abandoned-morphology");
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&p.params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encode_morphology(
        &runtime,
        cache,
        &p,
        &source.view(),
        &result.view(),
        Some(&scratch.create_view(&Default::default())),
        &uniform,
        &mut encoder,
        None,
    );
    drop(encoder);
    drop(scratch);
    assert_eq!(runtime.image_texture_pool_stats().cached_resources, 0);
    // The unsubmitted overwrite must not affect the retained dilation result.
    assert_eq!(
        result.readback_u8::<1>(&runtime).unwrap().as_slice(),
        oracle(&image, 9, 9, GpuMorphology::Dilate)
    );
}

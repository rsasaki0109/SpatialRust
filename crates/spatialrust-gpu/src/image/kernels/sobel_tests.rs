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
                3 => {
                    if (i % w + i / w) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                }
                4 => ((i % w * 3 + i / w * 5) % 32) as u8,
                _ => {
                    if i == w * h / 2 {
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

fn oracle(image: &Image<u8, 1>) -> Vec<u8> {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let sample = |x: i64, y: i64| {
        i64::from(image.as_slice()[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize])
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut gx = 0;
            let mut gy = 0;
            let horizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
            let vertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
            for dy in 0..3 {
                for dx in 0..3 {
                    let v = sample(x + dx as i64 - 1, y + dy as i64 - 1);
                    gx += horizontal[dy][dx] * v;
                    gy += vertical[dy][dx] * v;
                }
            }
            (gx.abs() + gy.abs()).min(255) as u8
        })
        .collect()
}

fn pipeline_from_shader(
    runtime: &WgpuRuntime,
    cache: &SpatialPipelines,
    source: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frozen-sobel"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
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
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn previous_pipeline(runtime: &WgpuRuntime, cache: &SpatialPipelines) -> wgpu::ComputePipeline {
    pipeline_from_shader(
        runtime,
        cache,
        include_str!("../../../tests/fixtures/sobel_previous.wgsl"),
        "sobel",
    )
}

fn experimental_pipeline(runtime: &WgpuRuntime, cache: &SpatialPipelines) -> wgpu::ComputePipeline {
    pipeline_from_shader(
        runtime,
        cache,
        include_str!("../../../tests/fixtures/sobel_tiled_experiment.wgsl"),
        "sobel_tiled",
    )
}

fn run_pipeline(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    pipeline: &wgpu::ComputePipeline,
    channels: u32,
    profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuImage> {
    require_gray(source, "sobel_gpu")?;
    source.validate_runtime(runtime)?;
    validate_spatial_dispatch(
        [source.width(), source.height()],
        [source.width(), source.height()],
        &runtime.device().limits(),
    )?;
    if let Some(profiler) = &profiler {
        profiler.validate(runtime, 1)?;
    }
    dispatch(
        runtime,
        source,
        source.width(),
        source.height(),
        3,
        3,
        0,
        pipeline,
        "sobel_gpu",
        channels,
        profiler,
    )
}

#[test]
fn tiled_direct_and_frozen_sobel_match_independent_full_rgba_oracle() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache =
        runtime.image_spatial_pipelines.get_or_init(|| create_spatial_pipelines(runtime.device()));
    let previous = previous_pipeline(&runtime, cache);
    let experimental = experimental_pipeline(&runtime, cache);
    let mut pixels = 0;
    for (w, h) in
        [(1, 1), (1, 17), (17, 1), (15, 15), (16, 16), (17, 19), (31, 33), (257, 259), (512, 257)]
    {
        for pattern in 0..6 {
            let image = fixture(w, h, pattern);
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let expected: Vec<_> = oracle(&image).into_iter().flat_map(|v| [v, 0, 0, 0]).collect();
            for pipeline in [&previous, &cache.sobel, &experimental] {
                let mut output = run_pipeline(&runtime, &source, pipeline, 4, None).unwrap();
                assert_eq!(
                    output.readback_u8::<4>(&runtime).unwrap().as_slice(),
                    expected,
                    "{w}x{h}, pattern {pattern}"
                );
                output.recycle(&runtime);
            }
            source.recycle(&runtime);
            pixels += w * h;
        }
    }
    eprintln!("Verified {pixels} pixels across 54 fixtures and three GPU variants, {} exact RGBA bytes overall", pixels * 4 * 3);
}

#[test]
fn spatial_dispatch_validates_limits_without_additional_storage_requirements() {
    let limits = wgpu::Limits::default();
    let valid = |w, h, limits: &wgpu::Limits| validate_spatial_dispatch([w, h], [w, h], limits);
    assert!(valid(1023, 1025, &limits).is_ok());
    assert!(valid(1024, 1024, &limits).is_ok());
    for (w, h) in [(0, 1), (1, 0), (u32::MAX, 1), (1, u32::MAX)] {
        assert!(valid(w, h, &limits).is_err());
    }
    let mut changed = limits.clone();
    changed.max_compute_workgroup_storage_size = 0;
    assert!(valid(1024, 1024, &changed).is_ok());
    changed.max_texture_dimension_2d = 255;
    assert!(valid(256, 256, &changed).is_err());
    for field in 0..12 {
        let mut changed = limits.clone();
        match field {
            0 => changed.max_compute_invocations_per_workgroup = 255,
            1 => changed.max_compute_workgroup_size_x = 15,
            2 => changed.max_compute_workgroup_size_y = 15,
            3 => changed.max_compute_workgroup_size_z = 0,
            4 => changed.max_compute_workgroups_per_dimension = 15,
            5 => changed.max_sampled_textures_per_shader_stage = 0,
            6 => changed.max_storage_textures_per_shader_stage = 0,
            7 => changed.max_uniform_buffers_per_shader_stage = 0,
            8 => changed.max_uniform_buffer_binding_size = 31,
            9 => changed.max_buffer_size = 31,
            10 => changed.max_bind_groups = 0,
            _ => changed.max_bindings_per_bind_group = 2,
        }
        assert!(valid(256, 256, &changed).is_err(), "field {field}");
    }
    let mut huge = limits;
    huge.max_texture_dimension_2d = u32::MAX;
    huge.max_compute_workgroups_per_dimension = u32::MAX;
    assert!(valid(i32::MAX as u32, 1, &huge).is_ok());
    assert!(valid(i32::MAX as u32 + 1, 1, &huge).is_err());
    huge.max_storage_buffers_per_shader_stage = 0;
    huge.max_storage_buffer_binding_size = 0;
    assert!(valid(1024, 1024, &huge).is_ok());
    assert!(validate_resize_dispatch([65537, 1], [65536, 1], &huge).is_ok());
    assert!(validate_resize_dispatch([65537, 1], [65537, 1], &huge).is_err());
    assert!(validate_resize_dispatch([1, 65537], [1, 65536], &huge).is_ok());
    assert!(validate_resize_dispatch([1, 65537], [1, 65537], &huge).is_err());
}

#[test]
fn invalid_sobel_runtime_channels_and_profile_capacity_do_not_allocate() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let color = Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap();
    let color = GpuImage::upload_u8(&runtime, color.view()).unwrap();
    let source = GpuImage::upload_u8(&runtime, fixture(1024, 1024, 0).view()).unwrap();
    let before = runtime.image_texture_pool_stats();
    assert!(sobel_gpu(&runtime, &color).is_err());
    assert!(sobel_gpu(&other, &source).is_err());
    assert!(resize_nearest_gpu(&other, &source, 17, 19).is_err());
    let max = runtime.device().limits().max_texture_dimension_2d;
    for (w, h) in [(0, 1), (1, 0), (max + 1, 1), (1, max + 1), (u32::MAX, 1), (1, u32::MAX)] {
        assert!(resize_nearest_gpu(&runtime, &source, w, h).is_err());
    }
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert!(runtime.image_spatial_pipelines.get().is_none());
    assert!(other.image_spatial_pipelines.get().is_none());
    let small = GpuImage::upload_u8(&runtime, fixture(17, 19, 4).view()).unwrap();
    let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
    let output = sobel_gpu_with_profiler(&runtime, &small, Some(&mut profiler)).unwrap();
    let before = runtime.image_texture_pool_stats();
    assert!(sobel_gpu_with_profiler(&runtime, &source, Some(&mut profiler)).is_err());
    assert!(
        resize_nearest_gpu_with_profiler(&runtime, &source, 17, 19, Some(&mut profiler)).is_err()
    );
    assert_eq!(runtime.image_texture_pool_stats(), before);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.stages.len(), 1);
    assert_eq!(report.stages[0].stage, "sobel_gpu");
    assert_eq!(report.device_to_host_bytes, 16);
    output.recycle(&runtime);
    small.recycle(&runtime);
    source.recycle(&runtime);
    color.recycle(&runtime);
}

#[test]
fn strided_roi_cold_concurrency_live_outputs_and_pool_reuse_stay_exact() {
    use spatialrust_image::{ColorRange, ColorSpace, ImageMetadata, ImageRegion, ImageView};
    let runtime = WgpuRuntime::new_headless().unwrap();
    let stride = 1057;
    let data: Vec<_> = (0..stride * 1033).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let metadata = ImageMetadata {
        color_space: ColorSpace::Gray,
        color_range: ColorRange::Limited,
        ..Default::default()
    };
    let view = ImageView::<u8, 1>::new_with_metadata(1039, 1033, stride, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(7, 3, 1025, 1025))
        .unwrap();
    let packed = Image::<u8, 1>::try_new(
        1025,
        1025,
        (0..1025)
            .flat_map(|y| data[(y + 3) * stride + 7..(y + 3) * stride + 1032].iter().copied())
            .collect(),
    )
    .unwrap();
    let expected = oracle(&packed);
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let outputs = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                let runtime = &runtime;
                let source = &source;
                scope.spawn(move || sobel_gpu(runtime, source).unwrap())
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    assert!(runtime.image_spatial_pipelines.get().is_some());
    source.recycle(&runtime);
    // Reuse the old source texture while earlier outputs remain independently live.
    let source = GpuImage::upload_u8(&runtime, fixture(1025, 1025, 2).view()).unwrap();
    for _ in 0..12 {
        sobel_gpu(&runtime, &source).unwrap().recycle(&runtime);
    }
    let runtime = Box::new(runtime);
    for mut output in outputs {
        assert_eq!(output.metadata(), metadata);
        assert_eq!(output.receipt().stages(), ["upload_u8_texture", "sobel_gpu"]);
        assert_eq!(output.receipt().host_to_device_bytes(), 1025 * 1025 * 4);
        assert_eq!(output.receipt().gpu_to_gpu_bytes(), 1025 * 1025 * 4);
        assert_eq!(output.receipt().device_to_host_bytes(), 0);
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected);
        output.recycle(&runtime);
    }
    let before = runtime.image_texture_pool_stats();
    for _ in 0..12 {
        sobel_gpu(&runtime, &source).unwrap().recycle(&runtime);
    }
    assert_eq!(runtime.image_texture_pool_stats().allocation_count, before.allocation_count);
    runtime.set_image_texture_pool_budget(0);
    let mut output = sobel_gpu(&runtime, &source).unwrap();
    assert!(output.readback_u8::<1>(&runtime).unwrap().as_slice().iter().all(|&v| v == 0));
    output.recycle(&runtime);
    source.recycle(&runtime);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
}

#[test]
fn validated_resize_preserves_frozen_full_rgba_mapping_and_metadata() {
    use spatialrust_image::{ColorRange, ImageMetadata};
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache =
        runtime.image_spatial_pipelines.get_or_init(|| create_spatial_pipelines(runtime.device()));
    let previous = pipeline_from_shader(
        &runtime,
        cache,
        include_str!("../../../tests/fixtures/sobel_previous.wgsl"),
        "resize_nearest",
    );
    let metadata = ImageMetadata { color_range: ColorRange::Limited, ..Default::default() };
    let mut bytes = 0;
    for (w, h) in [(1usize, 1usize), (1, 17), (17, 1), (15, 19), (16, 16), (17, 19), (257, 259)] {
        let image = Image::<u8, 4>::try_new_with_metadata(
            w,
            h,
            (0..w * h * 4).map(|i| ((i * 37 + i / 7) % 256) as u8).collect(),
            metadata,
        )
        .unwrap();
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        for (ow, oh) in [(1u32, 1u32), (3, 5), (15, 17), (16, 16), (17, 19), (31, 33)] {
            let mut expected = Vec::new();
            for y in 0..oh {
                for x in 0..ow {
                    let sx = (u64::from(x) * w as u64 / u64::from(ow)) as usize;
                    let sy = (u64::from(y) * h as u64 / u64::from(oh)) as usize;
                    expected.extend_from_slice(
                        &image.as_slice()[(sy * w + sx) * 4..(sy * w + sx) * 4 + 4],
                    );
                }
            }
            let old = dispatch(
                &runtime,
                &source,
                ow,
                oh,
                1,
                1,
                0,
                &previous,
                "resize_nearest_gpu",
                4,
                None,
            )
            .unwrap();
            let current = resize_nearest_gpu(&runtime, &source, ow, oh).unwrap();
            for mut output in [old, current] {
                assert_eq!(output.metadata(), metadata);
                assert_eq!(output.receipt().host_to_device_bytes(), (w * h * 4) as u64);
                assert_eq!(output.receipt().gpu_to_gpu_bytes(), u64::from(ow) * u64::from(oh) * 4);
                assert_eq!(output.receipt().device_to_host_bytes(), 0);
                assert_eq!(output.readback_u8::<4>(&runtime).unwrap().as_slice(), expected);
                output.recycle(&runtime);
                bytes += expected.len();
            }
        }
        source.recycle(&runtime);
    }
    eprintln!(
        "Verified 42 resize mappings and {bytes} exact RGBA bytes across current/frozen outputs."
    );
}

#[path = "sobel_benchmark.rs"]
mod benchmark;

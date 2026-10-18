use super::*;
use spatialrust_image::{Image, ImageMetadata, ImageRegion, ImageView};
use spatialrust_vision::{gaussian_blur_u8, BorderMode};

fn border<const C: usize>(value: GpuGaussianBorder) -> BorderMode<u8, C> {
    match value {
        GpuGaussianBorder::Replicate => BorderMode::Replicate,
        GpuGaussianBorder::Reflect => BorderMode::Reflect,
        GpuGaussianBorder::Reflect101 => BorderMode::Reflect101,
        GpuGaussianBorder::Wrap => BorderMode::Wrap,
        GpuGaussianBorder::Constant(pixel) => {
            BorderMode::Constant(std::array::from_fn(|c| pixel[c]))
        }
    }
}

fn fixture<const C: usize>(w: usize, h: usize, hot: bool) -> Image<u8, C> {
    Image::try_new(
        w,
        h,
        (0..w * h * C)
            .map(|i| if hot { 255 } else { ((i * 7919 + i / (w * C) * 31) % 256) as u8 })
            .collect(),
    )
    .unwrap()
}

fn compare_channels<const C: usize>(runtime: &WgpuRuntime) -> usize {
    let borders = [
        GpuGaussianBorder::Replicate,
        GpuGaussianBorder::Reflect,
        GpuGaussianBorder::Reflect101,
        GpuGaussianBorder::Wrap,
        GpuGaussianBorder::Constant([19, 73, 129, 251]),
    ];
    let mut checked = 0;
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (17, 11), (65, 33)] {
        let image = fixture::<C>(w, h, false);
        let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
        for kw in [3, 5, 7] {
            for kh in [3, 5, 7] {
                for (sx, sy) in [(0.17, 0.8), (1.2, 2.3), (7.0, 1.9)] {
                    for b in borders {
                        let expected =
                            gaussian_blur_u8(image.view(), kw, kh, sx, sy, border(b)).unwrap();
                        let options = GpuGaussianBlurOptions {
                            kernel_width: kw as u32,
                            kernel_height: kh as u32,
                            sigma_x: sx,
                            sigma_y: sy,
                            border: b,
                        };
                        let mut output = gaussian_blur_gpu(runtime, &source, options).unwrap();
                        assert_eq!(output.channels(), C as u32);
                        assert_eq!(output.metadata(), image.metadata());
                        assert_eq!(output.receipt().host_to_device_bytes(), (w * h * 4) as u64);
                        assert_eq!(
                            output.receipt().gpu_to_gpu_bytes(),
                            (w * h * (if kw == 7 || kh == 7 { C } else { C.div_ceil(2) }) * 4
                                + w * h * 4) as u64
                        );
                        assert_eq!(output.receipt().device_to_host_bytes(), 0);
                        assert_eq!(
                            output.readback_u8::<C>(runtime).unwrap().as_slice(),
                            expected.as_slice(),
                            "{w}x{h} C={C} {options:?}"
                        );
                        output.recycle(runtime);
                        checked += w * h * C;
                    }
                }
            }
        }
        source.recycle(runtime);
    }
    checked
}

#[test]
fn logical_channels_kernels_borders_and_strip_sizes_match_cpu_exactly() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let checked = compare_channels::<1>(&runtime)
        + compare_channels::<2>(&runtime)
        + compare_channels::<3>(&runtime)
        + compare_channels::<4>(&runtime);
    eprintln!("GPU_GAUSSIAN_CPU exact_byte_comparisons={checked} calls=3240");
}

#[test]
fn sigma_extremes_constant_mass_and_unused_components_match_cpu() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for hot in [false, true] {
        let image = fixture::<3>(11, 9, hot);
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        for size in [3, 5, 7] {
            for sigma in [f64::from_bits(1), 1e-300, 1e-100, 0.05, 0.5, 1e100, f64::MAX] {
                for b in [GpuGaussianBorder::Replicate, GpuGaussianBorder::Constant([255; 4])] {
                    let options = GpuGaussianBlurOptions {
                        kernel_width: size,
                        kernel_height: size,
                        sigma_x: sigma,
                        sigma_y: sigma,
                        border: b,
                    };
                    let expected = gaussian_blur_u8(
                        image.view(),
                        size as usize,
                        size as usize,
                        sigma,
                        sigma,
                        border(b),
                    )
                    .unwrap();
                    let mut output = gaussian_blur_gpu(&runtime, &source, options).unwrap();
                    assert_eq!(
                        output.readback_u8::<3>(&runtime).unwrap().as_slice(),
                        expected.as_slice()
                    );
                    // Inspect the complete physical texture, including unused alpha.
                    let inspect = GpuImage::from_parts(
                        &runtime,
                        11,
                        9,
                        4,
                        output.texture().clone(),
                        output.metadata(),
                        Default::default(),
                    );
                    // from_parts is an owned texture view of the same live resource.
                    let mut inspect = inspect.unwrap();
                    let rgba = inspect.readback_u8::<4>(&runtime).unwrap();
                    assert!(rgba.as_slice().chunks_exact(4).all(|pixel| pixel[3] == 0));
                    drop(inspect);
                    output.recycle(&runtime);
                }
            }
        }
    }
}

#[test]
fn gaussian_plan_limits_cover_scratch_indices_and_preallocation() {
    let limits = wgpu::Limits::default();
    let o = GpuGaussianBlurOptions::default();
    assert_eq!(std::mem::size_of::<Params>(), 128);
    assert_eq!(plan(257, 259, 4, o, &limits).unwrap().scratch_bytes, 257 * 259 * 8);
    for (w, h, c) in
        [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 5), (8193, 1, 1), (u32::MAX, u32::MAX, 4)]
    {
        assert!(plan(w, h, c, o, &limits).is_err());
    }
    for (kw, kh) in [(0, 3), (1, 3), (2, 3), (3, 9), (u32::MAX, 7)] {
        assert!(plan(
            1,
            1,
            1,
            GpuGaussianBlurOptions { kernel_width: kw, kernel_height: kh, ..o },
            &limits
        )
        .is_err());
    }
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(plan(1, 1, 1, GpuGaussianBlurOptions { sigma_x: sigma, ..o }, &limits).is_err());
        assert!(plan(1, 1, 1, GpuGaussianBlurOptions { sigma_y: sigma, ..o }, &limits).is_err());
    }
    for field in 0..14 {
        let mut bad = limits.clone();
        match field {
            0 => bad.max_compute_invocations_per_workgroup = 255,
            1 => bad.max_compute_workgroup_size_x = 255,
            2 => bad.max_compute_workgroups_per_dimension = 0,
            3 => bad.max_sampled_textures_per_shader_stage = 0,
            4 => bad.max_storage_textures_per_shader_stage = 0,
            5 => bad.max_storage_buffers_per_shader_stage = 0,
            6 => bad.max_uniform_buffers_per_shader_stage = 0,
            7 => bad.max_uniform_buffer_binding_size = 127,
            8 => bad.max_buffer_size = 127,
            9 => bad.max_storage_buffer_binding_size = 15,
            10 => bad.max_bind_groups = 0,
            11 => bad.max_bindings_per_bind_group = 2,
            12 => bad.max_compute_workgroup_size_y = 0,
            _ => bad.max_compute_workgroup_size_z = 0,
        }
        assert!(plan(1, 1, 4, o, &bad).is_err(), "field={field}");
    }
    let mut exact = limits.clone();
    exact.max_buffer_size = 128;
    exact.max_storage_buffer_binding_size = 16;
    assert!(plan(1, 1, 4, o, &exact).is_ok());
    assert!(plan(2, 1, 4, o, &exact).is_err());
    exact.max_compute_workgroups_per_dimension = 1;
    exact.max_buffer_size = limits.max_buffer_size;
    exact.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
    assert_eq!(plan(257, 259, 1, o, &exact).unwrap().params.groups, 1);
    let mut huge = limits;
    huge.max_texture_dimension_2d = u32::MAX;
    huge.max_buffer_size = u64::MAX;
    huge.max_storage_buffer_binding_size = u32::MAX;
    assert!(plan(65536, 65536, 1, o, &huge).is_err());
    assert!(plan(i32::MAX as u32, 1, 1, o, &huge).is_err());
    assert!(plan(65535, 65535, 4, o, &huge).is_err());
}

#[test]
fn invalid_runtime_options_and_profiler_do_not_allocate_resources() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let other =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let image = fixture::<1>(17, 19, false);
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let before = runtime.image_texture_pool_stats();
    let scratch = runtime.scratch_buffer_pool_stats();
    let o = GpuGaussianBlurOptions::default();
    assert!(gaussian_blur_gpu(&other, &source, o).is_err());
    for options in [
        GpuGaussianBlurOptions { kernel_width: 9, ..o },
        GpuGaussianBlurOptions { sigma_x: f64::NAN, ..o },
        GpuGaussianBlurOptions { sigma_y: 0.0, ..o },
    ] {
        assert!(gaussian_blur_gpu(&runtime, &source, options).is_err());
    }
    if runtime.timestamp_queries_enabled() && other.timestamp_queries_enabled() {
        let mut short = GpuStageProfiler::new(&runtime, 1).unwrap();
        assert!(gaussian_blur_gpu_profiled(&runtime, &source, o, &mut short).is_err());
        let mut wrong = GpuStageProfiler::new(&other, 2).unwrap();
        assert!(gaussian_blur_gpu_profiled(&runtime, &source, o, &mut wrong).is_err());
    }
    assert_eq!(runtime.image_texture_pool_stats(), before);
    assert_eq!(runtime.scratch_buffer_pool_stats(), scratch);
    assert!(runtime.image_gaussian_pipelines.get().is_none());
}

#[test]
fn roi_metadata_profiler_chaining_and_concurrent_live_results_preserve_ownership() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let metadata =
        ImageMetadata { color_range: spatialrust_image::ColorRange::Limited, ..Default::default() };
    let data: Vec<_> = (0..81 * 31).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let view = ImageView::<u8, 3>::new_with_metadata(25, 31, 81, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(3, 2, 17, 19))
        .unwrap();
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let o = GpuGaussianBlurOptions { kernel_width: 7, kernel_height: 3, ..Default::default() };
    let expected = gaussian_blur_u8(view, 7, 3, o.sigma_x, o.sigma_y, border(o.border)).unwrap();
    let mut profiler =
        runtime.timestamp_queries_enabled().then(|| GpuStageProfiler::new(&runtime, 2).unwrap());
    let profiled = if let Some(ref mut profiler) = profiler {
        gaussian_blur_gpu_profiled(&runtime, &source, o, profiler).unwrap()
    } else {
        gaussian_blur_gpu(&runtime, &source, o).unwrap()
    };
    let mut results = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| gaussian_blur_gpu(&runtime, &source, o).unwrap()))
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    results.push(profiled);
    source.recycle(&runtime);
    let runtime = Box::new(runtime);
    for output in &mut results {
        assert_eq!(output.metadata(), metadata);
        output.receipt().validate_resident_chain(17 * 19 * 4).unwrap();
        assert_eq!(output.readback_u8::<3>(&runtime).unwrap(), expected);
    }
    if let Some(profiler) = profiler {
        let timing = profiler.collect(&runtime).unwrap();
        assert_eq!(
            timing.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
            ["gaussian_horizontal_gpu", "gaussian_blur_gpu"]
        );
    }
    assert_eq!(runtime.initialized_image_pipeline_count(), 1);
    for output in results {
        output.recycle(&runtime);
    }
    let source = GpuImage::upload_u8(&runtime, view).unwrap();
    let allocations = runtime.scratch_buffer_pool_stats().allocation_count;
    for _ in 0..5 {
        let mut output = gaussian_blur_gpu(&runtime, &source, o).unwrap();
        assert_eq!(output.readback_u8::<3>(&runtime).unwrap(), expected);
        output.recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, allocations);
    let gray = crate::rgb_to_gray_gpu(&runtime, &source).unwrap();
    let blurred = gaussian_blur_gpu(&runtime, &gray, o).unwrap();
    let edges = crate::sobel_gpu(&runtime, &blurred).unwrap();
    let tensor = crate::pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(17 * 19 * 4).unwrap();
    assert_eq!(
        tensor.receipt().stages(),
        [
            "upload_u8_texture",
            "rgb_to_gray_gpu",
            "gaussian_horizontal_gpu",
            "gaussian_blur_gpu",
            "sobel_gpu",
            "pack_ai_chw_gpu"
        ]
    );
}

#[test]
fn grid_stride_tail_and_q15_high_limb_match_cpu() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for hot in [false, true] {
        let image = fixture::<1>(1025, 1025, hot);
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let options =
            GpuGaussianBlurOptions { kernel_width: 7, kernel_height: 7, ..Default::default() };
        let expected = gaussian_blur_u8(
            image.view(),
            7,
            7,
            options.sigma_x,
            options.sigma_y,
            border(options.border),
        )
        .unwrap();
        let mut output = gaussian_blur_gpu(&runtime, &source, options).unwrap();
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap(), expected);
        output.recycle(&runtime);
    }
}

#[path = "gaussian_benchmark.rs"]
mod benchmark;

#[path = "gaussian_packed_tests.rs"]
mod packed;

#[path = "gaussian_pairs_tests.rs"]
mod pairs;

#[path = "gaussian_split_tests.rs"]
mod split;

#[path = "gaussian_mirror_tests.rs"]
mod mirror;

#[path = "gaussian_vmirror_tests.rs"]
mod vmirror;

#[path = "gaussian_rows_tests.rs"]
mod rows;

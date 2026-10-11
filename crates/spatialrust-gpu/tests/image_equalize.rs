#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]

use spatialrust_gpu::{
    equalize_histogram_gpu, equalize_histogram_gpu_profiled, rgb_to_gray_gpu, GpuImage,
    GpuStageProfiler, WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::{ColorRange, ColorSpace, Image, ImageMetadata, ImageRegion, ImageView};

fn check(runtime: &WgpuRuntime, view: ImageView<'_, u8, 1>) {
    let expected = spatialrust_vision::equalize_histogram(view).unwrap();
    let source = GpuImage::upload_u8(runtime, view).unwrap();
    let mut output = equalize_histogram_gpu(runtime, &source).unwrap();
    output.receipt().validate_resident_chain((view.width() * view.height() * 4) as u64).unwrap();
    source.recycle(runtime);
    let actual = output.readback_u8::<1>(runtime).unwrap();
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!((actual.width(), actual.height()), (view.width(), view.height()));
    output.recycle(runtime);
}

#[test]
fn cpu_byte_parity_covers_odd_grids_constant_images_and_binary64_half_drift() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for (width, height) in
        [(1, 1), (255, 1), (256, 1), (257, 1), (1, 257), (257, 257), (1001, 1003)]
    {
        for pattern in 0..4 {
            let pixels = (0..width * height)
                .map(|i| match pattern {
                    0 => 0,
                    1 => 255,
                    2 => 43,
                    _ => ((i * 37 + i / 997) % 129 + 32) as u8,
                })
                .collect();
            let image = Image::<u8, 1>::try_new(width, height, pixels).unwrap();
            check(&runtime, image.view());
        }
    }
    let mut values = vec![10];
    values.extend([20; 25]);
    values.extend([30; 25]);
    let image = Image::<u8, 1>::try_new(51, 1, values).unwrap();
    assert_eq!(spatialrust_vision::equalize_histogram(image.view()).unwrap().as_slice()[1], 127);
    check(&runtime, image.view());
    for values in [vec![10, 10, 20, 30], vec![0, 1, 2], vec![0, 1, 2, 3, 4, 5, 6]] {
        let image = Image::<u8, 1>::try_new(values.len(), 1, values).unwrap();
        check(&runtime, image.view());
    }
    assert_eq!(runtime.initialized_image_pipeline_count(), 3);
}

#[test]
fn strided_roi_and_rgb_gray_chain_preserve_metadata_without_intermediate_downloads() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let stride = 1043;
    let metadata = ImageMetadata {
        color_space: ColorSpace::Gray,
        color_range: ColorRange::Limited,
        ..Default::default()
    };
    let mut data = vec![255; stride * 1029];
    for y in 0..1029 {
        for x in 0..1029 {
            data[y * stride + x] = ((x * 17 + y * 31) % 129 + 32) as u8;
        }
    }
    let roi = ImageView::<u8, 1>::new_with_metadata(1029, 1029, stride, &data, metadata)
        .unwrap()
        .subview(ImageRegion::new(2, 1, 1025, 1027))
        .unwrap();
    check(&runtime, roi);
    let rgb = Image::<u8, 3>::try_new(
        257,
        129,
        (0..257 * 129 * 3).map(|i| (i * 37 % 256) as u8).collect(),
    )
    .unwrap();
    let gray_data = rgb
        .as_slice()
        .chunks_exact(3)
        .map(|p| {
            ((77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2]) + 128) >> 8) as u8
        })
        .collect();
    let host_gray = Image::<u8, 1>::try_new(257, 129, gray_data).unwrap();
    let expected = spatialrust_vision::equalize_histogram(host_gray.view()).unwrap();
    let source = GpuImage::upload_u8(&runtime, rgb.view()).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let mut output = equalize_histogram_gpu(&runtime, &gray).unwrap();
    output.receipt().validate_resident_chain(257 * 129 * 4).unwrap();
    assert_eq!(
        output.receipt().stages(),
        [
            "upload_u8_texture",
            "rgb_to_gray_gpu",
            "histogram_u8_partials_gpu",
            "histogram_u8_reduce_gpu",
            "histogram_cdf_u32_gpu",
            "equalize_histogram_lookup_gpu",
            "equalize_histogram_apply_gpu"
        ]
    );
    assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
    output.recycle(&runtime);
    gray.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn warmed_pools_and_retained_outputs_survive_stale_storage_runtime_moves_and_concurrency() {
    let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
    let first = Image::<u8, 1>::try_new(257, 257, vec![43; 257 * 257]).unwrap();
    let source = GpuImage::upload_u8(&boxed, first.view()).unwrap();
    let mut retained = equalize_histogram_gpu(&boxed, &source).unwrap();
    let runtime = *boxed;
    for value in [0, 255, 17, 43] {
        let image = Image::<u8, 1>::try_new(257, 257, vec![value; 257 * 257]).unwrap();
        check(&runtime, image.view());
    }
    let buffers = runtime.buffer_pool().stats();
    let scratch = runtime.scratch_buffer_pool_stats();
    let textures = runtime.image_texture_pool_stats();
    for _ in 0..5 {
        equalize_histogram_gpu(&runtime, &source).unwrap().recycle(&runtime);
    }
    assert_eq!(runtime.buffer_pool().stats().allocation_count, buffers.allocation_count);
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, scratch.allocation_count);
    assert_eq!(runtime.image_texture_pool_stats().allocation_count, textures.allocation_count);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    (0..3)
                        .map(|_| equalize_histogram_gpu(&runtime, &source).unwrap())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    source.recycle(&runtime);
    runtime.buffer_pool().set_budget(0);
    runtime.set_scratch_buffer_pool_budget(0);
    runtime.set_image_texture_pool_budget(0);
    assert!(retained.readback_u8::<1>(&runtime).unwrap().as_slice().iter().all(|&v| v == 43));
    retained.recycle(&runtime);
    for mut output in outputs {
        assert!(output.readback_u8::<1>(&runtime).unwrap().as_slice().iter().all(|&v| v == 43));
        output.recycle(&runtime);
    }
    assert_eq!(runtime.buffer_pool().stats().cached_bytes, 0);
    assert_eq!(runtime.scratch_buffer_pool_stats().cached_bytes, 0);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
}

#[test]
fn channel_and_foreign_runtime_rejections_do_not_allocate_work() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let rgb = Image::<u8, 3>::try_new(1, 1, vec![3, 4, 5]).unwrap();
    let source = GpuImage::upload_u8(&runtime, rgb.view()).unwrap();
    let before = runtime.buffer_pool().stats();
    let textures = runtime.image_texture_pool_stats();
    assert!(equalize_histogram_gpu(&runtime, &source).is_err());
    assert_eq!(runtime.buffer_pool().stats(), before);
    assert_eq!(runtime.image_texture_pool_stats(), textures);
    assert!(equalize_histogram_gpu(&other, &source).is_err());
    assert_eq!(other.buffer_pool().stats().allocation_count, 0);
    assert_eq!(other.image_texture_pool_stats().allocation_count, 0);
    assert_eq!(other.initialized_image_pipeline_count(), 0);
    source.recycle(&runtime);
}

#[test]
fn five_pass_profiling_matches_plain_output_and_preserves_capacity_rejection() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image =
        Image::<u8, 1>::try_new(257, 129, (0..257 * 129).map(|i| (i % 251) as u8).collect())
            .unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut short = GpuStageProfiler::new(&runtime, 4).unwrap();
    let before = runtime.buffer_pool().stats();
    assert!(equalize_histogram_gpu_profiled(&runtime, &source, &mut short).is_err());
    assert_eq!(runtime.buffer_pool().stats(), before);
    assert!(short.collect(&runtime).unwrap().stages.is_empty());
    let mut profiler = GpuStageProfiler::new(&runtime, 5).unwrap();
    let mut profiled = equalize_histogram_gpu_profiled(&runtime, &source, &mut profiler).unwrap();
    let mut plain = equalize_histogram_gpu(&runtime, &source).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    let before = runtime.scratch_buffer_pool_stats();
    assert!(equalize_histogram_gpu_profiled(&runtime, &source, &mut profiler).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.device_to_host_bytes, 80);
    assert_eq!(
        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        [
            "histogram_u8_partials_gpu",
            "histogram_u8_reduce_gpu",
            "histogram_cdf_u32_gpu",
            "equalize_histogram_lookup_gpu",
            "equalize_histogram_apply_gpu"
        ]
    );
    assert!(report.stages.iter().all(|s| s.elapsed_ns.is_finite() && s.elapsed_ns >= 0.0));
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    assert_eq!(
        profiled.readback_u8::<1>(&runtime).unwrap().as_slice(),
        plain.readback_u8::<1>(&runtime).unwrap().as_slice()
    );
    let rgb = Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap();
    let bad = GpuImage::upload_u8(&runtime, rgb.view()).unwrap();
    let mut failed = GpuStageProfiler::new(&runtime, 5).unwrap();
    assert!(equalize_histogram_gpu_profiled(&runtime, &bad, &mut failed).is_err());
    assert!(failed.collect(&runtime).is_err());
    profiled.recycle(&runtime);
    plain.recycle(&runtime);
    bad.recycle(&runtime);
    source.recycle(&runtime);
}

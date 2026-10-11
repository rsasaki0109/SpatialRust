#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]
use spatialrust_gpu::{
    clahe_gpu, clahe_gpu_profiled, rgb_to_gray_gpu, GpuImage, GpuStageProfiler,
    WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::{ColorRange, ColorSpace, Image, ImageMetadata, ImageRegion, ImageView};

fn check(runtime: &WgpuRuntime, view: ImageView<'_, u8, 1>, clip: f64, x: u32, y: u32) {
    let expected = spatialrust_vision::clahe(view, clip, x as usize, y as usize).unwrap();
    let source = GpuImage::upload_u8(runtime, view).unwrap();
    let mut output = clahe_gpu(runtime, &source, clip, x, y).unwrap();
    output.receipt().validate_resident_chain((view.width() * view.height() * 4) as u64).unwrap();
    let geometry = ((view.width() + view.height()) * 16) as u64;
    let area = view.width().div_ceil(x as usize) * view.height().div_ceil(y as usize);
    let chunks = area.div_ceil(8192).clamp(2, 32);
    let partials = u64::from(x) * u64::from(y) * chunks as u64 * 1024;
    let limits = runtime.device().limits();
    let split = area >= 65_536
        && x * y <= 16
        && partials
            <= limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    let mut stages = vec!["upload_u8_texture", "clahe_geometry_gpu"];
    if split {
        stages.push("clahe_histogram_gpu");
    }
    stages.extend(["clahe_lookup_gpu", "clahe_apply_gpu"]);
    assert_eq!(output.receipt().stages(), stages);
    assert_eq!(
        output.receipt().gpu_to_gpu_bytes(),
        geometry
            + u64::from(x) * u64::from(y) * 1024
            + (view.width() * view.height() * 4) as u64
            + if split { partials } else { 0 }
    );
    source.recycle(runtime);
    let actual = output.readback_u8::<1>(runtime).unwrap();
    if actual.as_slice() != expected.as_slice() {
        let i =
            actual.as_slice().iter().zip(expected.as_slice()).position(|(a, b)| a != b).unwrap();
        panic!(
            "{}x{}, grid {x}x{y}, clip {clip}, pixel ({},{}): {} != {}",
            view.width(),
            view.height(),
            i % view.width(),
            i / view.width(),
            actual.as_slice()[i],
            expected.as_slice()[i]
        );
    }
    assert_eq!(actual.metadata(), expected.metadata());
    output.recycle(runtime);
}

#[test]
fn cpu_byte_parity_covers_odd_grids_padding_constants_clipping_and_half_rounding() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for (width, height) in [(1, 1), (1, 37), (37, 1), (17, 19), (51, 1), (257, 259)] {
        for pattern in 0..4 {
            let data = (0..width * height)
                .map(|i| match pattern {
                    0 => 0,
                    1 => 43,
                    2 => 255,
                    _ => ((i * 37 + i / 997) % 129 + 32) as u8,
                })
                .collect();
            let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
            for (clip, x, y) in [
                (0., 1, 1),
                (0., 8, 8),
                (0.01, 13, 11),
                (2., 3, 5),
                (40., 8, 8),
                (f64::MAX, 17, 19),
            ] {
                check(&runtime, image.view(), clip, x, y);
            }
        }
    }
    let values = [vec![20; 25], vec![30; 25]].concat();
    let image = Image::<u8, 1>::try_new(50, 1, values).unwrap();
    assert_eq!(spatialrust_vision::clahe(image.view(), 0., 1, 1).unwrap().as_slice()[0], 127);
    check(&runtime, image.view(), 0., 1, 1);
    assert_eq!(runtime.initialized_image_pipeline_count(), 1);
}

#[test]
fn strided_roi_large_grid_stride_and_rgb_gray_chain_preserve_metadata() {
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
    for (clip, x, y) in [(2., 13, 11), (0., 3, 5)] {
        check(&runtime, roi, clip, x, y);
    }
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
    let gray_host = Image::<u8, 1>::try_new(257, 129, gray_data).unwrap();
    let expected = spatialrust_vision::clahe(gray_host.view(), 2., 8, 8).unwrap();
    let source = GpuImage::upload_u8(&runtime, rgb.view()).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let mut output = clahe_gpu(&runtime, &gray, 2., 8, 8).unwrap();
    output.receipt().validate_resident_chain(257 * 129 * 4).unwrap();
    assert_eq!(
        output.receipt().stages(),
        [
            "upload_u8_texture",
            "rgb_to_gray_gpu",
            "clahe_geometry_gpu",
            "clahe_lookup_gpu",
            "clahe_apply_gpu"
        ]
    );
    source.recycle(&runtime);
    gray.recycle(&runtime);
    assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
    output.recycle(&runtime);
}

#[test]
fn warmed_scratch_stale_storage_live_outputs_moves_and_concurrency_are_independent() {
    let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
    let image =
        Image::<u8, 1>::try_new(257, 259, (0..257 * 259).map(|i| (i * 37 % 256) as u8).collect())
            .unwrap();
    let expected = spatialrust_vision::clahe(image.view(), 2., 13, 11).unwrap();
    let source = GpuImage::upload_u8(&boxed, image.view()).unwrap();
    let mut retained = clahe_gpu(&boxed, &source, 2., 13, 11).unwrap();
    let runtime = *boxed;
    for value in [255, 0, 43] {
        let image = Image::<u8, 1>::try_new(257, 259, vec![value; 257 * 259]).unwrap();
        check(&runtime, image.view(), 0., 13, 11);
    }
    let before = runtime.scratch_buffer_pool_stats();
    let textures = runtime.image_texture_pool_stats();
    for _ in 0..5 {
        clahe_gpu(&runtime, &source, 2., 13, 11).unwrap().recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before.allocation_count);
    assert_eq!(runtime.image_texture_pool_stats().allocation_count, textures.allocation_count);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    (0..3)
                        .map(|_| clahe_gpu(&runtime, &source, 2., 13, 11).unwrap())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    source.recycle(&runtime);
    runtime.set_scratch_buffer_pool_budget(0);
    runtime.set_image_texture_pool_budget(0);
    assert_eq!(retained.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
    retained.recycle(&runtime);
    for mut output in outputs {
        assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
        output.recycle(&runtime);
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().cached_bytes, 0);
    assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
}

#[test]
fn invalid_parameters_foreign_runtime_and_channels_reject_without_work() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 1>::try_new(17, 19, vec![43; 17 * 19]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let scratch = runtime.scratch_buffer_pool_stats();
    let textures = runtime.image_texture_pool_stats();
    for (clip, x, y) in [
        (-1., 8, 8),
        (f64::NAN, 8, 8),
        (f64::INFINITY, 8, 8),
        (2., 0, 8),
        (2., 8, 0),
        (2., u32::MAX, u32::MAX),
    ] {
        assert!(clahe_gpu(&runtime, &source, clip, x, y).is_err());
    }
    assert_eq!(runtime.scratch_buffer_pool_stats(), scratch);
    assert_eq!(runtime.image_texture_pool_stats(), textures);
    assert_eq!(runtime.initialized_image_pipeline_count(), 0);
    assert!(clahe_gpu(&other, &source, 2., 8, 8).is_err());
    assert_eq!(other.scratch_buffer_pool_stats().allocation_count, 0);
    assert_eq!(other.initialized_image_pipeline_count(), 0);
    let rgb = Image::<u8, 3>::try_new(1, 1, vec![1, 2, 3]).unwrap();
    let uploaded = GpuImage::upload_u8(&runtime, rgb.view()).unwrap();
    let textures = runtime.image_texture_pool_stats();
    assert!(clahe_gpu(&runtime, &uploaded, 2., 8, 8).is_err());
    assert_eq!(runtime.image_texture_pool_stats(), textures);
    assert_eq!(runtime.initialized_image_pipeline_count(), 0);
    source.recycle(&runtime);
    uploaded.recycle(&runtime);
}

#[test]
fn three_pass_profile_matches_plain_preserves_capacity_and_invalidates_inner_failure() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image =
        Image::<u8, 1>::try_new(257, 129, (0..257 * 129).map(|i| (i * 37 % 256) as u8).collect())
            .unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut short = GpuStageProfiler::new(&runtime, 2).unwrap();
    assert!(clahe_gpu_profiled(&runtime, &source, 2., 8, 8, &mut short).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, 0);
    assert!(short.collect(&runtime).unwrap().stages.is_empty());
    let mut profiler = GpuStageProfiler::new(&runtime, 3).unwrap();
    let mut profiled = clahe_gpu_profiled(&runtime, &source, 2., 8, 8, &mut profiler).unwrap();
    let mut plain = clahe_gpu(&runtime, &source, 2., 8, 8).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    let before = runtime.scratch_buffer_pool_stats();
    assert!(clahe_gpu_profiled(&runtime, &source, 2., 8, 8, &mut profiler).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.device_to_host_bytes, 48);
    assert_eq!(
        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        ["clahe_geometry_gpu", "clahe_lookup_gpu", "clahe_apply_gpu"]
    );
    assert_eq!(
        profiled.readback_u8::<1>(&runtime).unwrap().as_slice(),
        plain.readback_u8::<1>(&runtime).unwrap().as_slice()
    );
    assert_eq!(plain.receipt().device_to_host_bytes(), 257 * 129 * 4);
    let mut failed = GpuStageProfiler::new(&runtime, 3).unwrap();
    assert!(clahe_gpu_profiled(&runtime, &source, -1., 8, 8, &mut failed).is_err());
    assert!(failed.collect(&runtime).is_err());
    profiled.recycle(&runtime);
    plain.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn split_four_pass_profile_checks_capacity_before_work_and_preserves_prior_stages() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image =
        Image::<u8, 1>::try_new(257, 259, (0..257 * 259).map(|i| (i * 37 % 256) as u8).collect())
            .unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    let textures = runtime.image_texture_pool_stats();
    let mut short = GpuStageProfiler::new(&runtime, 3).unwrap();
    assert!(clahe_gpu_profiled(&runtime, &source, 2., 1, 1, &mut short).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    assert_eq!(runtime.image_texture_pool_stats(), textures);
    assert_eq!(runtime.initialized_image_pipeline_count(), 0);
    assert!(short.collect(&runtime).unwrap().stages.is_empty());
    let small = Image::<u8, 1>::try_new(17, 19, vec![43; 17 * 19]).unwrap();
    let small_source = GpuImage::upload_u8(&runtime, small.view()).unwrap();
    let mut prior = GpuStageProfiler::new(&runtime, 4).unwrap();
    let previous = clahe_gpu_profiled(&runtime, &small_source, 2., 8, 8, &mut prior).unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    assert!(clahe_gpu_profiled(&runtime, &source, 2., 1, 1, &mut prior).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    let report = prior.collect(&runtime).unwrap();
    assert_eq!(report.stages.len(), 3);
    assert_eq!(report.device_to_host_bytes, 48);
    let mut profiler = GpuStageProfiler::new(&runtime, 4).unwrap();
    let mut output = clahe_gpu_profiled(&runtime, &source, 2., 1, 1, &mut profiler).unwrap();
    let mut plain = clahe_gpu(&runtime, &source, 2., 1, 1).unwrap();
    assert_eq!(output.receipt(), plain.receipt());
    assert_eq!(output.receipt().device_to_host_bytes(), 0);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.device_to_host_bytes, 64);
    assert_eq!(
        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        ["clahe_geometry_gpu", "clahe_histogram_gpu", "clahe_lookup_gpu", "clahe_apply_gpu"]
    );
    let expected = spatialrust_vision::clahe(image.view(), 2., 1, 1).unwrap();
    assert_eq!(output.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
    assert_eq!(plain.readback_u8::<1>(&runtime).unwrap().as_slice(), expected.as_slice());
    previous.recycle(&runtime);
    output.recycle(&runtime);
    plain.recycle(&runtime);
    source.recycle(&runtime);
    small_source.recycle(&runtime);
}

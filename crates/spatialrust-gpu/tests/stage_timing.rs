#![cfg(all(feature = "gpu-image", feature = "gpu-aoso-staging", not(target_arch = "wasm32")))]

use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, PointCloudBuilder, SpatialTensor, StandardSchemas,
};
use spatialrust_gpu::{
    run_aoso_voxel_normal_frame, run_aoso_voxel_normal_frame_profiled, run_gpu_vision_chain,
    run_gpu_vision_chain_profiled, GpuImage, GpuStageProfiler, GpuTimingReport,
    GpuVisionChainOptions, WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::Image;
use std::sync::OnceLock;

fn runtime() -> Option<&'static WgpuRuntime> {
    static RUNTIME: OnceLock<Option<WgpuRuntime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).ok()
        })
        .as_ref()
        .filter(|runtime| runtime.timestamp_queries_enabled())
}

fn image(runtime: &WgpuRuntime) -> GpuImage {
    let image =
        Image::<u8, 3>::try_new(16, 12, (0..16 * 12 * 3).map(|i| (i * 13 % 256) as u8).collect())
            .unwrap();
    GpuImage::upload_u8(runtime, image.view()).unwrap()
}

fn options() -> GpuVisionChainOptions {
    GpuVisionChainOptions { width: 8, height: 6, ..Default::default() }
}

fn cloud(n: usize) -> PointCloud {
    let mut buffers = PointBufferSet::new();
    for axis in ["x", "y", "z"] {
        buffers.insert(axis, PointBuffer::F32(Vec::new()));
    }
    if n == 0 {
        return PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            Default::default(),
        )
        .unwrap();
    }
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        builder.push_point([(i % 8) as f32 * 0.1 - 0.4, (i / 8) as f32 * 0.1 - 0.4, 0.0]).unwrap();
    }
    builder.build().unwrap()
}

fn check_report(report: &GpuTimingReport, count: usize) {
    assert_eq!(report.stages.len(), count);
    assert_eq!(report.device_to_host_bytes, count as u64 * 16);
    assert!(report
        .stages
        .iter()
        .all(|sample| sample.elapsed_ns.is_finite() && sample.elapsed_ns >= 0.0));
}

#[test]
fn default_runtime_does_not_enable_queries() {
    let runtime = WgpuRuntime::shared().unwrap();
    assert!(!runtime.timestamp_queries_enabled());
    assert!(GpuStageProfiler::new(&runtime, 6).is_err());
}

#[test]
fn image_passes_preserve_results_and_keep_readback_explicit() {
    let Some(runtime) = runtime() else {
        return;
    };
    let image = image(runtime);
    let mut profiler = GpuStageProfiler::new(runtime, 6).unwrap();
    let mut profiled =
        run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).unwrap();
    let mut plain = run_gpu_vision_chain(runtime, &image, options()).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    let report = profiler.collect(runtime).unwrap();
    check_report(&report, 6);
    assert_eq!(
        report.stages.iter().map(|sample| sample.stage).collect::<Vec<_>>(),
        [
            "resize_nearest_gpu",
            "rgb_to_gray_gpu",
            "box_blur_gpu",
            "sobel_gpu",
            "morphology_gpu",
            "pack_ai_chw_gpu"
        ]
    );
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    assert_eq!(profiled.readback_f32(runtime).unwrap(), plain.readback_f32(runtime).unwrap());
    profiled.recycle(runtime);
    plain.recycle(runtime);
    image.recycle(runtime);
}

#[test]
fn shared_recording_preserves_point_frames_and_accumulates_images() {
    let Some(runtime) = runtime() else {
        return;
    };
    let image = image(runtime);
    let cloud = cloud(64);
    let tensor = SpatialTensor::new(&cloud, 7).unwrap();
    let mut profiler = GpuStageProfiler::new(runtime, 167).unwrap();
    let before = run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).unwrap();
    let mut profiled =
        run_aoso_voxel_normal_frame_profiled(runtime, &tensor, [0.0; 3], 2.0, 0.25, &mut profiler)
            .unwrap();
    let after = run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).unwrap();
    let mut plain = run_aoso_voxel_normal_frame(runtime, &tensor, [0.0; 3], 2.0, 0.25).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    assert_eq!(profiled.receipt().device_to_host_bytes(), 8);
    let report = profiler.collect(runtime).unwrap();
    check_report(&report, 167);
    for (stage, count) in [
        ("voxel-keys", 2),
        ("voxel-sort-build", 2),
        ("voxel-radix-histogram", 48),
        ("voxel-radix-prefix", 48),
        ("voxel-radix-scatter", 48),
        ("voxel-compact-scan-blocks", 2),
        ("radius-normals", 1),
    ] {
        assert_eq!(report.stages.iter().filter(|sample| sample.stage == stage).count(), count);
    }
    assert_eq!(profiled.receipt().device_to_host_bytes(), 8);
    assert_eq!(
        profiled.readback_positions(runtime).unwrap(),
        plain.readback_positions(runtime).unwrap()
    );
    let actual = profiled.voxel_segments().unwrap().to_voxel_segments(runtime).unwrap();
    let expected = plain.voxel_segments().unwrap().to_voxel_segments(runtime).unwrap();
    assert_eq!(actual, expected);
    let normals = profiled.normals().unwrap().readback(runtime).unwrap();
    let expected = plain.normals().unwrap().readback(runtime).unwrap();
    for (a, b) in normals.iter().zip(&expected) {
        assert_eq!(a.normal, b.normal);
        assert_eq!(a.curvature, b.curvature);
        assert!((a.normal[2].abs() - 1.0).abs() < 1e-6);
        assert!(a.curvature.abs() < 1e-6);
    }
    profiled.recycle(runtime).unwrap();
    plain.recycle(runtime).unwrap();
    before.recycle(runtime);
    after.recycle(runtime);
    image.recycle(runtime);
}

#[test]
fn frame_query_capacity_tracks_empty_inputs_and_hierarchy_levels() {
    let Some(runtime) = runtime() else {
        return;
    };
    for (n, count) in
        [(0, 0), (1, 11), (2, 155), (3, 155), (257, 159), (65_536, 159), (65_537, 163)]
    {
        let cloud = cloud(n);
        let tensor = SpatialTensor::new(&cloud, 1).unwrap();
        let mut profiler = GpuStageProfiler::new(runtime, count.max(1)).unwrap();
        let frame = run_aoso_voxel_normal_frame_profiled(
            runtime,
            &tensor,
            [0.0; 3],
            2.0,
            0.25,
            &mut profiler,
        )
        .unwrap();
        check_report(&profiler.collect(runtime).unwrap(), count as usize);
        assert_eq!(frame.receipt().device_to_host_bytes(), if n == 0 { 0 } else { 8 });
        frame.recycle(runtime).unwrap();
    }
}

#[test]
fn capacity_rejection_preserves_prior_recording() {
    let Some(runtime) = runtime() else {
        return;
    };
    assert!(GpuStageProfiler::new(runtime, 0).is_err());
    assert!(GpuStageProfiler::new(runtime, 2049).is_err());
    let image = image(runtime);
    let mut profiler = GpuStageProfiler::new(runtime, 6).unwrap();
    let tensor = run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).unwrap();
    assert!(run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).is_err());
    let cloud = cloud(1);
    let points = SpatialTensor::new(&cloud, 1).unwrap();
    assert!(run_aoso_voxel_normal_frame_profiled(
        runtime,
        &points,
        [0.0; 3],
        2.0,
        0.25,
        &mut profiler
    )
    .is_err());
    check_report(&profiler.collect(runtime).unwrap(), 6);
    tensor.recycle(runtime);
    image.recycle(runtime);
}

#[test]
fn separable_blur_reserves_seven_stages_and_preserves_chain_results() {
    let Some(runtime) = runtime() else {
        return;
    };
    let image = image(runtime);
    let large = GpuVisionChainOptions { width: 32, height: 32, blur_kernel: 9, ..options() };
    let mut insufficient = GpuStageProfiler::new(runtime, 6).unwrap();
    assert!(run_gpu_vision_chain_profiled(runtime, &image, large, &mut insufficient).is_err());
    check_report(&insufficient.collect(runtime).unwrap(), 0);
    let mut profiler = GpuStageProfiler::new(runtime, 7).unwrap();
    let mut profiled =
        run_gpu_vision_chain_profiled(runtime, &image, large, &mut profiler).unwrap();
    let mut plain = run_gpu_vision_chain(runtime, &image, large).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    let report = profiler.collect(runtime).unwrap();
    check_report(&report, 7);
    assert_eq!(
        report.stages.iter().map(|sample| sample.stage).collect::<Vec<_>>(),
        [
            "resize_nearest_gpu",
            "rgb_to_gray_gpu",
            "box_blur_horizontal_gpu",
            "box_blur_gpu",
            "sobel_gpu",
            "morphology_gpu",
            "pack_ai_chw_gpu"
        ]
    );
    assert_eq!(profiled.readback_f32(runtime).unwrap(), plain.readback_f32(runtime).unwrap());
    profiled.recycle(runtime);
    plain.recycle(runtime);
    image.recycle(runtime);
}

#[test]
fn separable_morphology_reserves_stages_and_preserves_prior_recording() {
    let Some(runtime) = runtime() else {
        return;
    };
    let image = image(runtime);
    for blur_kernel in [3, 9] {
        let large = GpuVisionChainOptions {
            width: 128,
            height: 128,
            blur_kernel,
            morphology_kernel: 9,
            ..options()
        };
        let count = if blur_kernel == 9 { 8 } else { 7 };
        let mut insufficient = GpuStageProfiler::new(runtime, count - 1).unwrap();
        assert!(run_gpu_vision_chain_profiled(runtime, &image, large, &mut insufficient).is_err());
        check_report(&insufficient.collect(runtime).unwrap(), 0);
        let mut profiler = GpuStageProfiler::new(runtime, count).unwrap();
        let mut profiled =
            run_gpu_vision_chain_profiled(runtime, &image, large, &mut profiler).unwrap();
        let mut plain = run_gpu_vision_chain(runtime, &image, large).unwrap();
        assert_eq!(profiled.receipt(), plain.receipt());
        assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
        let report = profiler.collect(runtime).unwrap();
        check_report(&report, count as usize);
        let mut stages = vec!["resize_nearest_gpu", "rgb_to_gray_gpu"];
        if blur_kernel == 9 {
            stages.push("box_blur_horizontal_gpu");
        }
        stages.extend([
            "box_blur_gpu",
            "sobel_gpu",
            "morphology_horizontal_gpu",
            "morphology_gpu",
            "pack_ai_chw_gpu",
        ]);
        assert_eq!(report.stages.iter().map(|sample| sample.stage).collect::<Vec<_>>(), stages);
        assert_eq!(profiled.readback_f32(runtime).unwrap(), plain.readback_f32(runtime).unwrap());
        profiled.recycle(runtime);
        plain.recycle(runtime);
    }
    let mut prior = GpuStageProfiler::new(runtime, 12).unwrap();
    let tensor = run_gpu_vision_chain_profiled(runtime, &image, options(), &mut prior).unwrap();
    let large =
        GpuVisionChainOptions { width: 128, height: 128, morphology_kernel: 9, ..options() };
    assert!(run_gpu_vision_chain_profiled(runtime, &image, large, &mut prior).is_err());
    check_report(&prior.collect(runtime).unwrap(), 6);
    tensor.recycle(runtime);
    image.recycle(runtime);
}

#[test]
fn partial_chain_failure_invalidates_collection() {
    let Some(runtime) = runtime() else {
        return;
    };
    let image = image(runtime);
    let mut profiler = GpuStageProfiler::new(runtime, 12).unwrap();
    let bad = GpuVisionChainOptions { blur_kernel: 0, ..options() };
    assert!(run_gpu_vision_chain_profiled(runtime, &image, bad, &mut profiler).is_err());
    assert!(run_gpu_vision_chain_profiled(runtime, &image, options(), &mut profiler).is_err());
    assert!(profiler.collect(runtime).is_err());
    image.recycle(runtime);
}

#[test]
fn profiler_rejects_other_device_before_recording_or_collection() {
    let Some(runtime) = runtime() else {
        return;
    };
    let other =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let image = image(&other);
    let mut profiler = GpuStageProfiler::new(runtime, 6).unwrap();
    assert!(run_gpu_vision_chain_profiled(&other, &image, options(), &mut profiler).is_err());
    assert!(profiler.collect(&other).is_err());
    image.recycle(&other);
}

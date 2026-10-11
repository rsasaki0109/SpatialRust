#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]

use spatialrust_gpu::{
    histogram_u8_gpu, histogram_u8_gpu_profiled, rgb_to_gray_gpu, GpuImage, GpuStageProfiler,
    WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::Image;

#[test]
fn rgb_to_gray_histogram_matches_cpu_without_intermediate_image_readback() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let pixels: Vec<_> = (0..257 * 129 * 3).map(|i| (i * 37 % 256) as u8).collect();
    let image = Image::<u8, 3>::try_new(257, 129, pixels).unwrap();
    let mut expected = [0u32; 256];
    for rgb in image.as_slice().chunks_exact(3) {
        let luma =
            (77 * u32::from(rgb[0]) + 150 * u32::from(rgb[1]) + 29 * u32::from(rgb[2]) + 128) >> 8;
        expected[luma as usize] += 1;
    }
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let mut histogram = histogram_u8_gpu(&runtime, &gray, 0).unwrap();
    histogram.receipt().validate_resident_chain(257 * 129 * 4).unwrap();
    assert_eq!(
        histogram.receipt().stages(),
        [
            "upload_u8_texture",
            "rgb_to_gray_gpu",
            "histogram_u8_partials_gpu",
            "histogram_u8_reduce_gpu"
        ]
    );
    assert_eq!(histogram.readback_u32(&runtime).unwrap(), expected);
    assert_eq!(histogram.receipt().device_to_host_bytes(), 1024);
    histogram.recycle(&runtime);
    gray.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn histogram_timestamps_are_exactly_two_passes_with_explicit_collection() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image = Image::<u8, 1>::try_new(257, 129, vec![42; 257 * 129]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut insufficient = GpuStageProfiler::new(&runtime, 1).unwrap();
    let before = runtime.buffer_pool().stats();
    assert!(histogram_u8_gpu_profiled(&runtime, &source, 0, &mut insufficient).is_err());
    assert_eq!(runtime.buffer_pool().stats(), before);
    assert!(insufficient.collect(&runtime).unwrap().stages.is_empty());
    let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
    let mut profiled = histogram_u8_gpu_profiled(&runtime, &source, 0, &mut profiler).unwrap();
    let mut plain = histogram_u8_gpu(&runtime, &source, 0).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    assert!(histogram_u8_gpu_profiled(&runtime, &source, 0, &mut profiler).is_err());
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.device_to_host_bytes, 32);
    assert_eq!(
        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        ["histogram_u8_partials_gpu", "histogram_u8_reduce_gpu"]
    );
    assert!(report.stages.iter().all(|s| s.elapsed_ns.is_finite() && s.elapsed_ns >= 0.0));
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    assert_eq!(profiled.readback_u32(&runtime).unwrap(), plain.readback_u32(&runtime).unwrap());
    profiled.recycle(&runtime);
    plain.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn failed_recording_and_foreign_profiler_are_rejected() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image = Image::<u8, 1>::try_new(1, 1, vec![9]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
    assert!(histogram_u8_gpu_profiled(&runtime, &source, 1, &mut profiler).is_err());
    assert!(profiler.collect(&runtime).is_err());
    let other =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let mut foreign = GpuStageProfiler::new(&runtime, 2).unwrap();
    assert!(histogram_u8_gpu_profiled(&other, &source, 0, &mut foreign).is_err());
    assert!(foreign.collect(&runtime).unwrap().stages.is_empty());
    assert_eq!(other.scratch_buffer_pool_stats().allocation_count, 0);
    source.recycle(&runtime);
}

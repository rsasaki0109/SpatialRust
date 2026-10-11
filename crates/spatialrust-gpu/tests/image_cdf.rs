#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]

use spatialrust_gpu::{
    histogram_cdf_gpu, histogram_cdf_gpu_profiled, histogram_u8_gpu, histogram_u8_gpu_profiled,
    rgb_to_gray_gpu, GpuImage, GpuStageProfiler, WgpuPowerPreference, WgpuRuntime,
};
use spatialrust_image::{Image, ImageRegion, ImageView};

fn prefix(bins: &[u32; 256]) -> [u32; 256] {
    let mut count = 0;
    std::array::from_fn(|i| {
        count += bins[i];
        count
    })
}

fn check_channels<const C: usize>(runtime: &WgpuRuntime) {
    let stride = 21 * C + 17;
    let data: Vec<_> = (0..23 * stride).map(|i| (i * 37 % 256) as u8).collect();
    let roi = ImageView::<u8, C>::new(21, 23, stride, &data)
        .unwrap()
        .subview(ImageRegion::new(3, 2, 17, 19))
        .unwrap();
    let source = GpuImage::upload_u8(runtime, roi).unwrap();
    for channel in 0..C {
        let mut bins = [0; 256];
        for y in 0..roi.height() {
            for pixel in roi.row(y).unwrap().chunks_exact(C) {
                bins[pixel[channel] as usize] += 1;
            }
        }
        let histogram = histogram_u8_gpu(runtime, &source, channel as u32).unwrap();
        let before = histogram.receipt().clone();
        let mut cdf = histogram_cdf_gpu(runtime, &histogram).unwrap();
        assert_eq!(histogram.receipt(), &before);
        // The CDF owns its result independently of the submitted source reads.
        histogram.recycle(runtime);
        cdf.receipt().validate_resident_chain(17 * 19 * 4).unwrap();
        assert_eq!(cdf.pixel_count(), 17 * 19);
        assert_eq!(cdf.channel(), channel as u32);
        assert_eq!(cdf.receipt().gpu_to_gpu_bytes(), before.gpu_to_gpu_bytes() + 1024);
        assert_eq!(cdf.readback_u32(runtime).unwrap(), prefix(&bins));
        assert_eq!(cdf.receipt().device_to_host_bytes(), 1024);
        cdf.recycle(runtime);
    }
    source.recycle(runtime);
}

#[test]
fn all_channels_and_strided_rois_produce_inclusive_counts_without_bin_downloads() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    check_channels::<1>(&runtime);
    check_channels::<2>(&runtime);
    check_channels::<3>(&runtime);
    check_channels::<4>(&runtime);
}

#[test]
fn rgb_gray_histogram_cdf_chain_stays_resident_until_explicit_readback() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 3>::try_new(
        257,
        129,
        (0..257 * 129 * 3).map(|i| (i * 37 % 256) as u8).collect(),
    )
    .unwrap();
    let mut bins = [0; 256];
    for rgb in image.as_slice().chunks_exact(3) {
        let luma =
            (77 * u32::from(rgb[0]) + 150 * u32::from(rgb[1]) + 29 * u32::from(rgb[2]) + 128) >> 8;
        bins[luma as usize] += 1;
    }
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let histogram = histogram_u8_gpu(&runtime, &gray, 0).unwrap();
    let mut cdf = histogram_cdf_gpu(&runtime, &histogram).unwrap();
    histogram.recycle(&runtime);
    gray.recycle(&runtime);
    source.recycle(&runtime);
    cdf.receipt().validate_resident_chain(257 * 129 * 4).unwrap();
    assert_eq!(
        cdf.receipt().stages(),
        [
            "upload_u8_texture",
            "rgb_to_gray_gpu",
            "histogram_u8_partials_gpu",
            "histogram_u8_reduce_gpu",
            "histogram_cdf_u32_gpu"
        ]
    );
    assert_eq!(cdf.readback_u32(&runtime).unwrap(), prefix(&bins));
    assert_eq!(cdf.readback_u32(&runtime).unwrap()[255], 257 * 129);
    assert_eq!(cdf.receipt().device_to_host_bytes(), 2048);
    cdf.recycle(&runtime);
}

#[test]
fn receipt_inherits_prior_bin_readback_but_not_later_source_changes() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 1>::try_new(1, 1, vec![43]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut histogram = histogram_u8_gpu(&runtime, &source, 0).unwrap();
    let bins = histogram.readback_u32(&runtime).unwrap();
    let mut cdf = histogram_cdf_gpu(&runtime, &histogram).unwrap();
    assert_eq!(cdf.receipt().device_to_host_bytes(), 1024);
    assert!(cdf.receipt().validate_resident_chain(4).is_err());
    assert_eq!(cdf.readback_u32(&runtime).unwrap(), prefix(&bins));
    assert_eq!(cdf.receipt().device_to_host_bytes(), 2048);
    assert_eq!(histogram.receipt().device_to_host_bytes(), 1024);
    histogram.readback_u32(&runtime).unwrap();
    assert_eq!(cdf.receipt().stages().last(), Some(&"readback_cdf_u32"));
    assert_eq!(histogram.receipt().stages().last(), Some(&"readback_histogram_u32"));
    cdf.recycle(&runtime);
    histogram.recycle(&runtime);
    source.recycle(&runtime);
}

#[test]
fn shared_timing_records_three_passes_and_rejects_capacity_and_foreign_sources() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    if !runtime.timestamp_queries_enabled() {
        return;
    }
    let image = Image::<u8, 1>::try_new(257, 129, vec![42; 257 * 129]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut profiler = GpuStageProfiler::new(&runtime, 3).unwrap();
    let histogram = histogram_u8_gpu_profiled(&runtime, &source, 0, &mut profiler).unwrap();
    let mut profiled = histogram_cdf_gpu_profiled(&runtime, &histogram, &mut profiler).unwrap();
    let mut plain = histogram_cdf_gpu(&runtime, &histogram).unwrap();
    assert_eq!(profiled.receipt(), plain.receipt());
    let before = runtime.buffer_pool().stats();
    assert!(histogram_cdf_gpu_profiled(&runtime, &histogram, &mut profiler).is_err());
    assert_eq!(runtime.buffer_pool().stats(), before);
    let report = profiler.collect(&runtime).unwrap();
    assert_eq!(report.device_to_host_bytes, 48);
    assert_eq!(
        report.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        ["histogram_u8_partials_gpu", "histogram_u8_reduce_gpu", "histogram_cdf_u32_gpu"]
    );
    assert!(report.stages.iter().all(|s| s.elapsed_ns.is_finite() && s.elapsed_ns >= 0.0));
    assert_eq!(profiled.receipt().device_to_host_bytes(), 0);
    assert_eq!(profiled.readback_u32(&runtime).unwrap(), plain.readback_u32(&runtime).unwrap());
    let mut single = GpuStageProfiler::new(&runtime, 1).unwrap();
    histogram_cdf_gpu_profiled(&runtime, &histogram, &mut single).unwrap().recycle(&runtime);
    assert_eq!(single.collect(&runtime).unwrap().device_to_host_bytes, 16);
    let other =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    let mut foreign = GpuStageProfiler::new(&runtime, 1).unwrap();
    assert!(histogram_cdf_gpu_profiled(&other, &histogram, &mut foreign).is_err());
    assert!(foreign.collect(&runtime).unwrap().stages.is_empty());
    assert_eq!(other.buffer_pool().stats().allocation_count, 0);
    if other.timestamp_queries_enabled() {
        let mut matching = GpuStageProfiler::new(&other, 1).unwrap();
        assert!(histogram_cdf_gpu_profiled(&other, &histogram, &mut matching).is_err());
        assert!(matching.collect(&other).is_err());
    }
    profiled.recycle(&runtime);
    plain.recycle(&runtime);
    histogram.recycle(&runtime);
    source.recycle(&runtime);
}

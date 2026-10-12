#![cfg(feature = "gpu-image")]

use spatialrust::gpu::{histogram_u8_gpu, GpuImage, WgpuRuntime};
use spatialrust::image::Image;

#[test]
fn facade_large_gpu_histogram_preserves_resident_bins_and_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 1>::try_new(2048, 2048, vec![73; 2048 * 2048]).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut histogram = histogram_u8_gpu(&runtime, &source, 0).unwrap();
    assert_eq!(histogram.receipt().host_to_device_bytes(), 2048 * 2048 * 4);
    assert_eq!(histogram.receipt().gpu_to_gpu_bytes(), 256 * 1024 + 1024);
    assert_eq!(histogram.receipt().device_to_host_bytes(), 0);
    let mut expected = [0; 256];
    expected[73] = 2048 * 2048;
    assert_eq!(histogram.readback_u32(&runtime).unwrap(), expected);
    assert_eq!(histogram.receipt().device_to_host_bytes(), 1024);
    histogram.recycle(&runtime);
    source.recycle(&runtime);
}

#![cfg(feature = "gpu-image")]

use spatialrust::gpu::{GpuImage, WgpuRuntime};
use spatialrust::image::{ImageRegion, ImageView};

#[test]
fn facade_gpu_image_transfers_preserve_strided_roi_and_explicit_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let data: Vec<u8> = (0..(67 * 3 + 11) * 9).map(|i| (i * 37) as u8).collect();
    let view = ImageView::<u8, 3>::new(67, 9, 67 * 3 + 11, &data)
        .unwrap()
        .subview(ImageRegion::new(1, 1, 65, 7))
        .unwrap();
    let expected: Vec<_> = (0..7).flat_map(|y| view.row(y).unwrap().iter().copied()).collect();
    let mut image = GpuImage::upload_u8(&runtime, view).unwrap();
    assert_eq!(image.receipt().host_to_device_bytes(), 65 * 7 * 4);
    assert_eq!(image.receipt().device_to_host_bytes(), 0);
    assert_eq!(image.receipt().stages(), &["upload_u8_texture"]);
    assert_eq!(image.readback_u8::<3>(&runtime).unwrap().as_slice(), expected);
    assert_eq!(image.receipt().device_to_host_bytes(), 65 * 7 * 4);
    assert_eq!(image.receipt().stages(), &["upload_u8_texture", "readback_u8_texture"]);
    image.recycle(&runtime);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_large_image_transfers_keep_parallel_row_bytes_and_owned_outputs() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (width, height) = (1297, 809);
    let stride = (width + 5) * 3 + 11;
    let data: Vec<u8> = (0..stride * (height + 7)).map(|i| (i * 37) as u8).collect();
    let view = ImageView::<u8, 3>::new(width + 5, height + 7, stride, &data)
        .unwrap()
        .subview(ImageRegion::new(2, 3, width, height))
        .unwrap();
    let expected: Vec<_> = (0..height).flat_map(|y| view.row(y).unwrap().iter().copied()).collect();
    let mut first = GpuImage::upload_u8(&runtime, view).unwrap();
    let mut second = GpuImage::upload_u8(&runtime, view).unwrap();
    let first_host = first.readback_u8::<3>(&runtime).unwrap();
    assert_eq!(second.readback_u8::<3>(&runtime).unwrap().as_slice(), expected);
    assert_eq!(first_host.as_slice(), expected);
    let physical_bytes = (width * height * 4) as u64;
    assert_eq!(first.receipt().host_to_device_bytes(), physical_bytes);
    assert_eq!(first.receipt().device_to_host_bytes(), physical_bytes);
    assert_eq!(first.receipt(), second.receipt());
    first.recycle(&runtime);
    second.recycle(&runtime);
}

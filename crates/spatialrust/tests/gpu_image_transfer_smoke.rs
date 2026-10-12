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

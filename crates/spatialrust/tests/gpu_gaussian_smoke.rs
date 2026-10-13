#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]

use spatialrust::gpu::{
    gaussian_blur_gpu, pack_ai_chw_gpu, rgb_to_gray_gpu, sobel_gpu, GpuGaussianBlurOptions,
    GpuGaussianBorder, GpuImage, WgpuRuntime,
};
use spatialrust::image::Image;

#[test]
fn facade_gaussian_preserves_color_and_runs_a_resident_gray_edge_chain() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 3>::try_new(17, 11, [19, 73, 129].repeat(17 * 11)).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let options = GpuGaussianBlurOptions {
        kernel_width: 7,
        kernel_height: 5,
        border: GpuGaussianBorder::Reflect101,
        ..Default::default()
    };
    let mut color = gaussian_blur_gpu(&runtime, &source, options).unwrap();
    assert_eq!(color.channels(), 3);
    assert_eq!(color.readback_u8::<3>(&runtime).unwrap(), image);
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let blurred = gaussian_blur_gpu(&runtime, &gray, options).unwrap();
    let edges = sobel_gpu(&runtime, &blurred).unwrap();
    let tensor = pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(17 * 11 * 4).unwrap();
    assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
    source.recycle(&runtime);
    color.recycle(&runtime);
    gray.recycle(&runtime);
    blurred.recycle(&runtime);
    edges.recycle(&runtime);
}

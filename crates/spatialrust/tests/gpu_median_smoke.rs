#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]
use spatialrust::gpu::{
    median_blur_gpu, pack_ai_chw_gpu, rgb_to_gray_gpu, sobel_gpu, GpuImage, GpuImageBorder,
    WgpuRuntime,
};
use spatialrust::image::Image;

#[test]
#[ignore = "requires native GPU adapter"]
fn facade_median_preserves_constant_color_and_chains_resident_edges() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 3>::try_new(17, 11, [19, 73, 129].repeat(17 * 11)).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let mut color = median_blur_gpu(&runtime, &source, 5, GpuImageBorder::Replicate).unwrap();
    assert_eq!(color.readback_u8::<3>(&runtime).unwrap(), image);
    let filtered = median_blur_gpu(&runtime, &source, 3, GpuImageBorder::Replicate).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &filtered).unwrap();
    let edges = sobel_gpu(&runtime, &gray).unwrap();
    let tensor = pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(17 * 11 * 4).unwrap();
    assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
}

#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]

use spatialrust::gpu::{
    canny_gpu, gaussian_blur_gpu, pack_ai_chw_gpu, rgb_to_gray_gpu, GpuCannyOptions, GpuImage,
    WgpuRuntime,
};
use spatialrust::image::Image;

#[test]
fn facade_canny_keeps_a_gray_gaussian_edge_ai_chain_on_device() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let image = Image::<u8, 3>::try_new(17, 19, [19, 73, 129].repeat(17 * 19)).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    let gray = rgb_to_gray_gpu(&runtime, &source).unwrap();
    let blurred = gaussian_blur_gpu(&runtime, &gray, Default::default()).unwrap();
    for aperture_size in [3, 5, 7] {
        let mut edges = canny_gpu(
            &runtime,
            &blurred,
            GpuCannyOptions { aperture_size, l2_gradient: true, ..Default::default() },
        )
        .unwrap();
        let tensor = pack_ai_chw_gpu(&runtime, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
        tensor.receipt().validate_resident_chain(17 * 19 * 4).unwrap();
        assert_eq!(edges.readback_u8::<1>(&runtime).unwrap().as_slice(), vec![0; 17 * 19]);
        edges.recycle(&runtime);
    }
    source.recycle(&runtime);
    gray.recycle(&runtime);
    blurred.recycle(&runtime);
}

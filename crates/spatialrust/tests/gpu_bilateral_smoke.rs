#![cfg(all(feature = "gpu-image", not(target_arch = "wasm32")))]
use spatialrust::{
    gpu::{
        pack_ai_chw_gpu, rgb_to_gray_gpu, sobel_gpu, GpuBilateralFilter, GpuBilateralOptions,
        GpuImage, WgpuRuntime,
    },
    image::Image,
};
#[test]
#[ignore = "requires native GPU adapter"]
fn facade_bilateral_reuses_controls_and_retains_resident_chain() {
    let r = WgpuRuntime::new_headless().unwrap();
    let im = Image::<u8, 3>::try_new(17, 11, [19, 73, 129].repeat(17 * 11)).unwrap();
    let src = GpuImage::upload_u8(&r, im.view()).unwrap();
    let plan = GpuBilateralFilter::prepare(&r, 3, Default::default()).unwrap();
    let mut out = plan.filter(&r, &src).unwrap();
    assert_eq!(out.readback_u8::<3>(&r).unwrap(), im);
    let out = plan.filter(&r, &src).unwrap();
    let gray = rgb_to_gray_gpu(&r, &out).unwrap();
    let edges = sobel_gpu(&r, &gray).unwrap();
    let tensor = pack_ai_chw_gpu(&r, &edges, 1.0 / 255.0, [0.0; 4], [1.0; 4]).unwrap();
    tensor.receipt().validate_resident_chain(17 * 11 * 4).unwrap();
    assert!(plan.receipt().host_to_device_bytes() > 0);
    let _ = GpuBilateralOptions::default();
}

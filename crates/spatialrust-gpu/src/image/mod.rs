//! GPU-resident packed images with explicit host/device transfers.

pub(crate) mod ai_tensor;
mod gpu_image;
pub(crate) mod histogram;
pub(crate) mod kernels;
mod vision_chain;

pub use ai_tensor::{pack_ai_chw_gpu, GpuAiTensor};
pub use gpu_image::{GpuImage, GpuImageReceipt};
pub use histogram::{histogram_u8_gpu, histogram_u8_gpu_profiled, GpuImageHistogram};
pub use kernels::{
    box_blur_gpu, copy_gpu_image, morphology_gpu, resize_nearest_gpu, rgb_to_gray_gpu, sobel_gpu,
    GpuImageBorder, GpuMorphology,
};
pub use vision_chain::{
    run_gpu_vision_chain, run_gpu_vision_chain_profiled, GpuVisionChainOptions,
};

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_image::Image;

    fn runtime() -> Option<std::sync::Arc<crate::WgpuRuntime>> {
        crate::WgpuRuntime::shared().ok()
    }

    fn cpu_rgb_to_gray(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
        (0..width * height)
            .map(|index| {
                let base = index * 3;
                let value = (77_u32 * u32::from(rgb[base])
                    + 150_u32 * u32::from(rgb[base + 1])
                    + 29_u32 * u32::from(rgb[base + 2])
                    + 128)
                    >> 8;
                value as u8
            })
            .collect()
    }

    #[test]
    fn upload_copy_chain_has_no_mid_host_readback() {
        let Some(runtime) = runtime() else {
            return;
        };
        let image = Image::<u8, 3>::try_new(
            8,
            4,
            (0..8 * 4 * 3).map(|index| (index % 200) as u8).collect(),
        )
        .unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        assert_eq!(uploaded.receipt().host_to_device_bytes(), (8 * 4 * 4) as u64);
        assert_eq!(uploaded.receipt().device_to_host_bytes(), 0);
        let copied = copy_gpu_image(&runtime, &uploaded).unwrap();
        assert_eq!(copied.receipt().device_to_host_bytes(), 0);
        assert!(copied.receipt().gpu_to_gpu_bytes() > 0);
        let mut owned = copied;
        let readback = owned.readback_u8::<3>(&runtime).unwrap();
        assert_eq!(readback.as_slice(), image.as_slice());
        assert!(owned.receipt().device_to_host_bytes() > 0);
    }

    #[test]
    fn gray_and_box_blur_chain_matches_cpu_luma() {
        let Some(runtime) = runtime() else {
            return;
        };
        let width = 32;
        let height = 24;
        let data = (0..width * height * 3)
            .map(|index| ((index * 13) % 200 + 20) as u8)
            .collect::<Vec<_>>();
        let image = Image::<u8, 3>::try_new(width, height, data.clone()).unwrap();
        let expected_gray = cpu_rgb_to_gray(width, height, &data);
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let gray = rgb_to_gray_gpu(&runtime, &uploaded).unwrap();
        assert_eq!(gray.receipt().device_to_host_bytes(), 0);
        let blurred = box_blur_gpu(&runtime, &gray, 3, 3, GpuImageBorder::Replicate).unwrap();
        assert_eq!(blurred.receipt().device_to_host_bytes(), 0);
        assert!(blurred.receipt().stages().contains(&"upload_u8_texture"));
        assert!(blurred.receipt().stages().contains(&"rgb_to_gray_gpu"));
        assert!(blurred.receipt().stages().contains(&"box_blur_gpu"));
        let mut gray_owned = gray;
        let gray_host = gray_owned.readback_u8::<1>(&runtime).unwrap();
        assert_eq!(gray_host.as_slice(), expected_gray.as_slice());
        let mut blur_owned = blurred;
        let blur_host = blur_owned.readback_u8::<1>(&runtime).unwrap();
        assert_eq!(blur_host.width(), width);
        assert_eq!(blur_host.height(), height);
    }

    #[test]
    fn rejects_cross_channel_readback() {
        let Some(runtime) = runtime() else {
            return;
        };
        let image = Image::<u8, 1>::try_new(4, 4, vec![7u8; 16]).unwrap();
        let mut gpu = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        assert!(gpu.readback_u8::<3>(&runtime).is_err());
    }

    #[test]
    fn resize_edge_morphology_chain_stays_on_texture() {
        let Some(runtime) = runtime() else {
            return;
        };
        let image = Image::<u8, 3>::try_new(
            8,
            8,
            (0..8 * 8 * 3).map(|index| ((index * 17) % 251) as u8).collect(),
        )
        .unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let resized = resize_nearest_gpu(&runtime, &uploaded, 16, 12).unwrap();
        let gray = rgb_to_gray_gpu(&runtime, &resized).unwrap();
        let edges = sobel_gpu(&runtime, &gray).unwrap();
        let dilated = morphology_gpu(&runtime, &edges, 3, 3, GpuMorphology::Dilate).unwrap();
        assert_eq!(dilated.receipt().device_to_host_bytes(), 0);
        assert_eq!(dilated.receipt().host_to_device_bytes(), 8 * 8 * 4);
        assert_eq!(
            dilated.receipt().stages(),
            &[
                "upload_u8_texture",
                "resize_nearest_gpu",
                "rgb_to_gray_gpu",
                "sobel_gpu",
                "morphology_gpu",
            ]
        );
        let mut output = dilated;
        let host = output.readback_u8::<1>(&runtime).unwrap();
        assert_eq!((host.width(), host.height()), (16, 12));
        assert!(host.as_slice().iter().any(|&value| value != 0));
    }

    #[test]
    fn texture_resize_and_morphology_match_known_pixels() {
        let Some(runtime) = runtime() else {
            return;
        };
        let gray = Image::<u8, 1>::try_new(2, 2, vec![0, 10, 20, 30]).unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, gray.view()).unwrap();
        let mut resized = resize_nearest_gpu(&runtime, &uploaded, 4, 4).unwrap();
        let host = resized.readback_u8::<1>(&runtime).unwrap();
        assert_eq!(host.as_slice(), &[0, 0, 10, 10, 0, 0, 10, 10, 20, 20, 30, 30, 20, 20, 30, 30]);

        let impulse = Image::<u8, 1>::try_new(3, 3, vec![0, 0, 0, 0, 255, 0, 0, 0, 0]).unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, impulse.view()).unwrap();
        let mut dilated = morphology_gpu(&runtime, &uploaded, 3, 3, GpuMorphology::Dilate).unwrap();
        assert_eq!(dilated.readback_u8::<1>(&runtime).unwrap().as_slice(), &[255; 9]);
        let mut eroded = morphology_gpu(&runtime, &uploaded, 3, 3, GpuMorphology::Erode).unwrap();
        assert_eq!(eroded.readback_u8::<1>(&runtime).unwrap().as_slice(), &[0; 9]);

        let flat = Image::<u8, 1>::try_new(4, 4, vec![42; 16]).unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, flat.view()).unwrap();
        let mut edges = sobel_gpu(&runtime, &uploaded).unwrap();
        assert_eq!(edges.readback_u8::<1>(&runtime).unwrap().as_slice(), &[0; 16]);
    }

    #[test]
    fn image_pipelines_are_cached_per_runtime_device() {
        let (Ok(first), Ok(second)) =
            (crate::WgpuRuntime::new_headless(), crate::WgpuRuntime::new_headless())
        else {
            return;
        };
        let rgb = Image::<u8, 3>::try_new(1, 1, vec![255, 0, 0]).unwrap();
        for runtime in [&first, &second] {
            let uploaded = GpuImage::upload_u8(runtime, rgb.view()).unwrap();
            let mut gray = rgb_to_gray_gpu(runtime, &uploaded).unwrap();
            assert_eq!(gray.readback_u8::<1>(runtime).unwrap().as_slice(), &[77]);
        }
    }

    #[test]
    fn vision_chain_uploads_once_stays_resident_and_reuses_pools() {
        let Some(runtime) = runtime() else {
            return;
        };
        let image = Image::<u8, 3>::try_new(
            32,
            24,
            (0..32 * 24 * 3).map(|index| ((index * 29) % 251) as u8).collect(),
        )
        .unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let options = GpuVisionChainOptions { width: 16, height: 12, ..Default::default() };
        let mut tensor = run_gpu_vision_chain(&runtime, &uploaded, options).unwrap();
        assert_eq!(tensor.shape(), [1, 12, 16]);
        assert_eq!(tensor.receipt().host_to_device_bytes(), 32 * 24 * 4);
        assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
        tensor.receipt().validate_resident_chain(32 * 24 * 4).unwrap();
        assert!(tensor.receipt().validate_resident_chain(1).is_err());
        assert_eq!(
            tensor.receipt().stages(),
            &[
                "upload_u8_texture",
                "resize_nearest_gpu",
                "rgb_to_gray_gpu",
                "box_blur_gpu",
                "sobel_gpu",
                "morphology_gpu",
                "pack_ai_chw_gpu",
            ]
        );
        assert_eq!(runtime.initialized_image_pipeline_count(), 4);
        assert!(runtime.cached_image_texture_count() > 0);

        let values = tensor.readback_f32(&runtime).unwrap();
        assert_eq!(values.len(), 16 * 12);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(tensor.receipt().device_to_host_bytes() > 0);
        assert!(tensor.receipt().validate_resident_chain(32 * 24 * 4).is_err());
        tensor.recycle(&runtime);
        assert!(runtime.buffer_pool().cached_buffer_count() > 0);

        let tensor = run_gpu_vision_chain(&runtime, &uploaded, options).unwrap();
        assert_eq!(runtime.initialized_image_pipeline_count(), 4);
        assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
        tensor.receipt().validate_resident_chain(32 * 24 * 4).unwrap();
        tensor.recycle(&runtime);

        let cached_textures = runtime.cached_image_texture_count();
        let cached_buffers = runtime.buffer_pool().cached_buffer_count();
        let tensor = run_gpu_vision_chain(&runtime, &uploaded, options).unwrap();
        tensor.recycle(&runtime);
        assert_eq!(runtime.cached_image_texture_count(), cached_textures);
        assert_eq!(runtime.buffer_pool().cached_buffer_count(), cached_buffers);
        uploaded.recycle(&runtime);
    }

    #[test]
    fn texture_and_tensor_retention_stays_bounded_across_resolutions() {
        let runtime = crate::WgpuRuntime::new_headless().unwrap();
        runtime.set_image_texture_pool_budget(8 * 1024);
        runtime.buffer_pool().set_budget(4 * 1024);
        for side in [16, 24, 32, 48, 64, 16] {
            let image = Image::<u8, 3>::try_new(side, side, vec![91; side * side * 3]).unwrap();
            let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let tensor = run_gpu_vision_chain(
                &runtime,
                &uploaded,
                GpuVisionChainOptions {
                    width: (side / 2) as u32,
                    height: (side / 2) as u32,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
            tensor.recycle(&runtime);
            uploaded.recycle(&runtime);
            assert!(runtime.image_texture_pool_stats().cached_bytes <= 8 * 1024);
            assert!(runtime.buffer_pool().stats().cached_bytes <= 4 * 1024);
        }
        assert!(runtime.image_texture_pool_stats().discarded_count > 0);
        let image = Image::<u8, 3>::try_new(8, 8, vec![73; 8 * 8 * 3]).unwrap();
        let mut live = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        runtime.set_image_texture_pool_budget(0);
        runtime.buffer_pool().set_budget(0);
        assert_eq!(runtime.image_texture_pool_stats().cached_bytes, 0);
        assert_eq!(runtime.buffer_pool().stats().cached_bytes, 0);
        assert_eq!(live.readback_u8::<3>(&runtime).unwrap().as_slice(), image.as_slice());
        live.recycle(&runtime);
        assert_eq!(runtime.cached_image_texture_count(), 0);
    }

    #[test]
    fn repeated_resident_image_chain_reuses_every_pooled_allocation() {
        let runtime = crate::WgpuRuntime::new_headless().unwrap();
        let image = Image::<u8, 3>::try_new(32, 24, vec![82; 32 * 24 * 3]).unwrap();
        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let options = GpuVisionChainOptions { width: 16, height: 12, ..Default::default() };
        run_gpu_vision_chain(&runtime, &uploaded, options).unwrap().recycle(&runtime);
        let textures = runtime.image_texture_pool_stats();
        let buffers = runtime.buffer_pool().stats();
        for _ in 0..8 {
            let tensor = run_gpu_vision_chain(&runtime, &uploaded, options).unwrap();
            assert_eq!(tensor.receipt().device_to_host_bytes(), 0);
            tensor.recycle(&runtime);
        }
        let after = runtime.image_texture_pool_stats();
        assert_eq!(after.allocation_count, textures.allocation_count);
        assert_eq!(after.cached_bytes, textures.cached_bytes);
        assert_eq!(after.reuse_count - textures.reuse_count, 5 * 8);
        assert_eq!(runtime.buffer_pool().stats().allocation_count, buffers.allocation_count);
        assert_eq!(runtime.buffer_pool().stats().reuse_count - buffers.reuse_count, 8);
        runtime.clear_image_texture_pool();
        assert_eq!(runtime.cached_image_texture_count(), 0);
        uploaded.recycle(&runtime);
    }

    #[test]
    fn image_and_tensor_ownership_survives_runtime_moves_and_rejects_other_pools() {
        let runtime = Box::new(crate::WgpuRuntime::new_headless().unwrap());
        let image = Image::<u8, 3>::try_new(4, 4, vec![73; 4 * 4 * 3]).unwrap();
        let mut uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let mut tensor = pack_ai_chw_gpu(&runtime, &uploaded, 1.0, [0.0; 4], [1.0; 4]).unwrap();
        let moved = *runtime;
        assert_eq!(uploaded.readback_u8::<3>(&moved).unwrap().as_slice(), image.as_slice());
        assert_eq!(tensor.readback_f32(&moved).unwrap(), vec![73.0; 4 * 4 * 3]);
        let other = crate::WgpuRuntime::new_headless().unwrap();
        assert!(uploaded.readback_u8::<3>(&other).is_err());
        assert!(tensor.readback_f32(&other).is_err());
        uploaded.recycle(&other);
        tensor.recycle(&other);
        assert_eq!(other.cached_image_texture_count(), 0);
        assert_eq!(other.buffer_pool().cached_buffer_count(), 0);
    }
}

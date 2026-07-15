//! GPU-resident packed images with explicit host/device transfers.

mod gpu_image;
mod kernels;

pub use gpu_image::{GpuImage, GpuImageReceipt};
pub use kernels::{box_blur_gpu, copy_gpu_image, rgb_to_gray_gpu, GpuImageBorder};

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_image::Image;

    fn runtime() -> Option<crate::WgpuRuntime> {
        crate::WgpuRuntime::new_headless().ok()
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
        assert_eq!(uploaded.receipt().host_to_device_bytes(), (8 * 4 * 3 * 4) as u64);
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
        assert!(blurred.receipt().stages().contains(&"upload_u8"));
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
}

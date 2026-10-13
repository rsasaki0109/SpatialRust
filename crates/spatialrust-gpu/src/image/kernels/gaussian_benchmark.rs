//! Isolated completed-call measurement; output checks occur outside timers.
use super::*;
use spatialrust_vision::{gaussian_blur_u8_into, GaussianBlurU8Workspace};
use std::time::Instant;

fn cpu<const C: usize>(image: &Image<u8, C>, options: GpuGaussianBlurOptions) -> Image<u8, C> {
    gaussian_blur_u8(
        image.view(),
        options.kernel_width as usize,
        options.kernel_height as usize,
        options.sigma_x,
        options.sigma_y,
        border(options.border),
    )
    .unwrap()
}

fn measure<const C: usize>(runtime: &WgpuRuntime, w: usize, h: usize, size: u32, hot: bool) {
    let image = fixture::<C>(w, h, hot);
    let options =
        GpuGaussianBlurOptions { kernel_width: size, kernel_height: size, ..Default::default() };
    let expected = cpu(&image, options);
    let mut workspace = GaussianBlurU8Workspace::new();
    let mut packed = vec![0; w * h * C];
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    let cpu_into = |packed: &mut [u8], workspace: &mut GaussianBlurU8Workspace| {
        gaussian_blur_u8_into(
            image.view(),
            size as usize,
            size as usize,
            options.sigma_x,
            options.sigma_y,
            border(options.border),
            packed,
            workspace,
        )
        .unwrap();
    };
    for _ in 0..3 {
        assert_eq!(cpu(&image, options), expected);
        cpu_into(&mut packed, &mut workspace);
        assert_eq!(packed, expected.as_slice());
        let mut output = gaussian_blur_gpu(runtime, &source, options).unwrap();
        assert_eq!(output.readback_u8::<C>(runtime).unwrap(), expected);
        output.recycle(runtime);
        let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let mut output = gaussian_blur_gpu(runtime, &uploaded, options).unwrap();
        assert_eq!(output.readback_u8::<C>(runtime).unwrap(), expected);
        uploaded.recycle(runtime);
        output.recycle(runtime);
    }
    let (mut cpu_ms, mut cpu_into_ms, mut full_gpu_ms, mut resident_ms) =
        (vec![], vec![], vec![], vec![]);
    let mut receipt = None;
    for round in 0..20 {
        for offset in 0..4 {
            let start = Instant::now();
            match (round + offset) % 4 {
                0 => {
                    let result = cpu(&image, options);
                    cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(result, expected);
                }
                1 => {
                    cpu_into(&mut packed, &mut workspace);
                    cpu_into_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(packed, expected.as_slice());
                }
                2 => {
                    let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
                    let mut output = gaussian_blur_gpu(runtime, &uploaded, options).unwrap();
                    let result = output.readback_u8::<C>(runtime).unwrap();
                    receipt = Some(output.receipt().clone());
                    uploaded.recycle(runtime);
                    output.recycle(runtime);
                    full_gpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(result, expected);
                }
                _ => {
                    let mut output = gaussian_blur_gpu(runtime, &source, options).unwrap();
                    runtime.wait_idle();
                    resident_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    output.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                    assert_eq!(output.readback_u8::<C>(runtime).unwrap(), expected);
                    output.recycle(runtime);
                }
            }
        }
    }
    let receipt = receipt.unwrap();
    assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
    assert_eq!(receipt.gpu_to_gpu_bytes(), (w * h * (C + 1) * 4) as u64);
    assert_eq!(receipt.device_to_host_bytes(), ((w * 4).div_ceil(256) * 256 * h) as u64);
    println!("GPU_GAUSSIAN_BENCH width={w} height={h} channels={C} kernel={size} hot={} h2d={} g2g={} d2h={} cpu_ms={cpu_ms:?} cpu_into_ms={cpu_into_ms:?} full_gpu_ms={full_gpu_ms:?} resident_ms={resident_ms:?}",u8::from(hot),receipt.host_to_device_bytes(),receipt.gpu_to_gpu_bytes(),receipt.device_to_host_bytes());
    source.recycle(runtime);
}

#[test]
#[ignore = "run this exact completed-call survey alone in release mode"]
fn completed_gaussian_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!("GPU_GAUSSIAN_ENV backend={} device_type={}", info.backend, info.device_type);
    measure::<1>(&runtime, 256, 192, 3, false);
    measure::<3>(&runtime, 256, 192, 7, false);
    for size in [3, 5, 7] {
        measure::<1>(&runtime, 1920, 1080, size, false);
        measure::<3>(&runtime, 1920, 1080, size, false);
    }
    measure::<4>(&runtime, 1920, 1080, 7, false);
    measure::<1>(&runtime, 3840, 2160, 5, false);
    measure::<3>(&runtime, 3840, 2160, 7, false);
    measure::<4>(&runtime, 3840, 2160, 7, true);
}

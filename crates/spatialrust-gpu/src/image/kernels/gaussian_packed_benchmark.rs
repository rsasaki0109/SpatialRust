use super::*;
use std::{hint::black_box, time::Instant};

fn measure<const C: usize>(
    r: &WgpuRuntime,
    old: &frozen::FrozenGaussianPipelines,
    w: usize,
    h: usize,
    kernel: u32,
) {
    let image = fixture::<C>(w, h, false);
    let o = GpuGaussianBlurOptions {
        kernel_width: kernel,
        kernel_height: kernel,
        ..Default::default()
    };
    let expected = gaussian_blur_u8(
        image.view(),
        kernel as usize,
        kernel as usize,
        o.sigma_x,
        o.sigma_y,
        border(o.border),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    r.wait_idle();
    for round in 0..33 {
        for offset in 0..4 {
            let variant = (round + offset) % 4;
            let full = variant >= 2;
            let current = variant % 2 == 1;
            let start = Instant::now();
            let uploaded =
                full.then(|| GpuImage::upload_u8(black_box(r), black_box(image.view())).unwrap());
            let input = uploaded.as_ref().unwrap_or(&source);
            let mut output = if current {
                gaussian_blur_gpu(black_box(r), black_box(input), black_box(o)).unwrap()
            } else {
                frozen::baseline_gaussian(black_box(r), black_box(input), black_box(o), None, old)
                    .unwrap()
            };
            let downloaded = if full {
                Some(output.readback_u8::<C>(r).unwrap())
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let downloaded = downloaded.unwrap_or_else(|| output.readback_u8::<C>(r).unwrap());
            assert_eq!(downloaded, expected);
            let scratch = if current && kernel != 7 { C.div_ceil(2) } else { C };
            assert_eq!(output.receipt().gpu_to_gpu_bytes(), (w * h * (scratch + 1) * 4) as u64);
            output.recycle(r);
            if let Some(input) = uploaded {
                input.recycle(r);
            }
            if round >= 3 {
                samples[variant].push(elapsed);
            }
        }
    }
    println!("GAUSSIAN_PACKED_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"kernel\":{kernel},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3]);
    source.recycle(r);
}

#[test]
#[ignore = "paired completed Gaussian calls; run alone in release mode"]
fn benchmark_complete_packed_gaussian_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let old = frozen::FrozenGaussianPipelines::new(r.device());
    println!("GAUSSIAN_PACKED_ENV backend={} device_type={} warmups=3 pairs=30 routes=resident_completed_and_upload_filter_download old_cache=prewarmed output_checks_drop_recycle_excluded=true",r.adapter_info().backend,r.adapter_info().device_type);
    measure::<4>(&r, &old, 16, 17, 3);
    measure::<3>(&r, &old, 257, 259, 5);
    measure::<2>(&r, &old, 1920, 1080, 3);
    measure::<4>(&r, &old, 1920, 1080, 5);
    measure::<3>(&r, &old, 1920, 1080, 7);
    measure::<1>(&r, &old, 1920, 1080, 5);
    measure::<4>(&r, &old, 3840, 2160, 3);
    measure::<3>(&r, &old, 3840, 2160, 5);
}

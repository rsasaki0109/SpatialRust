//! Completed serial CPU, explicit full GPU and resident GPU median calls.
use super::*;
use std::time::Instant;

fn measure<const C: usize>(runtime: &WgpuRuntime, w: usize, h: usize, kernel: u32, profile: usize) {
    let image = fixture::<C>(w, h, profile);
    let expected = median_blur(image.view(), kernel as usize, BorderMode::Replicate).unwrap();
    let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut phases: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..23 {
        for offset in 0..3 {
            let stage = (round + offset) % 3;
            let start = Instant::now();
            let elapsed;
            match stage {
                0 => {
                    let actual =
                        median_blur(image.view(), kernel as usize, BorderMode::Replicate).unwrap();
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                }
                1 => {
                    let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
                    let upload = start.elapsed().as_secs_f64() * 1000.0;
                    let mut output =
                        median_blur_gpu(runtime, &uploaded, kernel, GpuImageBorder::Replicate)
                            .unwrap();
                    let submit = start.elapsed().as_secs_f64() * 1000.0;
                    let actual = output.readback_u8::<C>(runtime).unwrap();
                    let read = start.elapsed().as_secs_f64() * 1000.0;
                    let receipt = output.receipt().clone();
                    uploaded.recycle(runtime);
                    output.recycle(runtime);
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
                    assert_eq!(receipt.gpu_to_gpu_bytes(), (w * h * 4) as u64);
                    assert_eq!(receipt.device_to_host_bytes(), (w * h * 4) as u64);
                    if round >= 3 {
                        for (phase, ms) in phases.iter_mut().zip([
                            upload,
                            submit - upload,
                            read - submit,
                            elapsed - read,
                        ]) {
                            phase.push(ms);
                        }
                    }
                }
                _ => {
                    let mut output =
                        median_blur_gpu(runtime, &source, kernel, GpuImageBorder::Replicate)
                            .unwrap();
                    runtime.wait_idle();
                    elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    output.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                    assert_eq!(output.readback_u8::<C>(runtime).unwrap(), expected);
                    output.recycle(runtime);
                }
            }
            if round >= 3 {
                samples[stage].push(elapsed);
            }
        }
    }
    source.recycle(runtime);
    println!("GPU_MEDIAN_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"kernel\":{kernel},\"profile\":{profile},\"cpu_ms\":{:?},\"full_gpu_ms\":{:?},\"resident_ms\":{:?},\"full_phases_ms\":{:?}}}",samples[0],samples[1],samples[2],phases);
}

#[test]
#[ignore = "requires isolated completed-call release survey; exact test only"]
fn completed_median_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let info = runtime.adapter_info();
    println!(
        "GPU_MEDIAN_ENV backend={} device_type={} available_workers={} texture_budget={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        runtime.image_texture_pool_stats().budget_bytes
    );
    measure::<1>(&runtime, 65, 33, 3, 0);
    measure::<3>(&runtime, 257, 193, 5, 0);
    for kernel in [3, 5] {
        measure::<1>(&runtime, 1024, 1024, kernel, 0);
        measure::<3>(&runtime, 1024, 1024, kernel, 0);
        measure::<4>(&runtime, 1024, 1024, kernel, 0);
    }
    measure::<1>(&runtime, 1920, 1080, 3, 0);
    measure::<3>(&runtime, 1920, 1080, 5, 0);
    measure::<1>(&runtime, 1024, 1024, 5, 2);
    measure::<4>(&runtime, 1024, 1024, 5, 2);
    measure::<3>(&runtime, 1024, 1024, 5, 4);
}

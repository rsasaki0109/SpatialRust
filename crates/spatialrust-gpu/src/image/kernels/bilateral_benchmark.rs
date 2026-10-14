use super::*;
use std::time::Instant;
fn measure<const C: usize>(r: &WgpuRuntime, w: usize, h: usize, d: u32, p: usize) {
    let im = fixture::<C>(w, h, p);
    let expected =
        bilateral_filter(im.view(), d as usize, 40.0, 3.0, BorderMode::Replicate).unwrap();
    let options = GpuBilateralOptions { diameter: d, ..Default::default() };
    let plan = GpuBilateralFilter::prepare(r, C as u32, options).unwrap();
    let source = GpuImage::upload_u8(r, im.view()).unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let start = Instant::now();
            let ms;
            match stage {
                0 => {
                    let actual =
                        bilateral_filter(im.view(), d as usize, 40.0, 3.0, BorderMode::Replicate)
                            .unwrap();
                    ms = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                }
                1 | 2 => {
                    let fresh = if stage == 1 {
                        Some(GpuBilateralFilter::prepare(r, C as u32, options).unwrap())
                    } else {
                        None
                    };
                    let used = fresh.as_ref().unwrap_or(&plan);
                    let input = GpuImage::upload_u8(r, im.view()).unwrap();
                    let mut out = used.filter(r, &input).unwrap();
                    let actual = out.readback_u8::<C>(r).unwrap();
                    let receipt = out.receipt().clone();
                    input.recycle(r);
                    out.recycle(r);
                    drop(fresh);
                    ms = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(actual, expected);
                    assert_eq!(receipt.host_to_device_bytes(), (w * h * 4) as u64);
                    assert_eq!(receipt.gpu_to_gpu_bytes(), (w * h * 4) as u64);
                    assert_eq!(receipt.device_to_host_bytes(), (w * h * 4) as u64);
                }
                _ => {
                    let mut out = plan.filter(r, &source).unwrap();
                    r.wait_idle();
                    ms = start.elapsed().as_secs_f64() * 1000.0;
                    out.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                    assert_eq!(out.readback_u8::<C>(r).unwrap(), expected);
                    out.recycle(r);
                }
            }
            if round >= 3 {
                samples[stage].push(ms);
            }
        }
    }
    source.recycle(r);
    println!("GPU_BILATERAL_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"diameter\":{d},\"profile\":{p},\"control_bytes\":{},\"cpu_ms\":{:?},\"full_new_ms\":{:?},\"full_reuse_ms\":{:?},\"resident_ms\":{:?}}}",plan.receipt().host_to_device_bytes(),samples[0],samples[1],samples[2],samples[3]);
}
#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_gpu_bilateral_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!(
        "GPU_BILATERAL_ENV backend={} device_type={} workers={} texture_budget={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        r.image_texture_pool_stats().budget_bytes
    );
    measure::<1>(&r, 65, 33, 3, 0);
    measure::<3>(&r, 257, 193, 5, 0);
    measure::<1>(&r, 1024, 1024, 3, 0);
    measure::<3>(&r, 1024, 1024, 5, 0);
    measure::<4>(&r, 1024, 1024, 7, 0);
    measure::<3>(&r, 1920, 1080, 5, 0);
    measure::<3>(&r, 1024, 1024, 5, 2);
    measure::<1>(&r, 1024, 1024, 7, 0);
}

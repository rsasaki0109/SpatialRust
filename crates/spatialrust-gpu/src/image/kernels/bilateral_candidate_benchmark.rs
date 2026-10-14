use super::*;
use std::time::Instant;

fn measure<const C: usize>(
    r: &WgpuRuntime,
    frozen: &BilateralPipeline,
    w: usize,
    h: usize,
    d: u32,
    p: usize,
) {
    let image = fixture::<C>(w, h, p);
    let expected =
        bilateral_filter(image.view(), d as usize, 40.0, 3.0, BorderMode::Replicate).unwrap();
    let options = GpuBilateralOptions { diameter: d, ..Default::default() };
    let plan = GpuBilateralFilter::prepare(r, C as u32, options).unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
    for round in 0..23 {
        for offset in 0..6 {
            let stage = (round + offset) % 6;
            let start = Instant::now();
            let ms;
            let actual;
            if stage < 4 {
                let fresh = if stage < 2 {
                    Some(GpuBilateralFilter::prepare(r, C as u32, options).unwrap())
                } else {
                    None
                };
                let used = fresh.as_ref().unwrap_or(&plan);
                let input = GpuImage::upload_u8(r, image.view()).unwrap();
                let mut output = if stage % 2 == 0 {
                    filter_frozen(used, r, &input, frozen).unwrap()
                } else {
                    used.filter(r, &input).unwrap()
                };
                actual = output.readback_u8::<C>(r).unwrap();
                input.recycle(r);
                output.recycle(r);
                drop(fresh);
                ms = start.elapsed().as_secs_f64() * 1000.0;
            } else {
                let mut output = if stage == 4 {
                    filter_frozen(&plan, r, &source, frozen).unwrap()
                } else {
                    plan.filter(r, &source).unwrap()
                };
                r.wait_idle();
                ms = start.elapsed().as_secs_f64() * 1000.0;
                output.receipt().validate_resident_chain((w * h * 4) as u64).unwrap();
                actual = output.readback_u8::<C>(r).unwrap();
                output.recycle(r);
            }
            assert_eq!(actual, expected);
            if round >= 3 {
                samples[stage].push(ms);
            }
        }
    }
    source.recycle(r);
    println!("GPU_BILATERAL_CANDIDATE_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"diameter\":{d},\"profile\":{p},\"control_bytes\":{},\"old_newplan_ms\":{:?},\"new_newplan_ms\":{:?},\"old_reuse_ms\":{:?},\"new_reuse_ms\":{:?},\"old_resident_ms\":{:?},\"new_resident_ms\":{:?}}}",plan.receipt().host_to_device_bytes(),samples[0],samples[1],samples[2],samples[3],samples[4],samples[5]);
}

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_old_and_candidate_bilateral_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!(
        "GPU_BILATERAL_CANDIDATE_ENV backend={} device_type={} workers={} texture_budget={}",
        info.backend,
        info.device_type,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        r.image_texture_pool_stats().budget_bytes
    );
    let frozen = frozen_pipeline(&r);
    measure::<1>(&r, &frozen, 65, 33, 3, 0);
    measure::<3>(&r, &frozen, 257, 193, 5, 0);
    measure::<1>(&r, &frozen, 1024, 1024, 3, 0);
    measure::<3>(&r, &frozen, 1024, 1024, 5, 0);
    measure::<4>(&r, &frozen, 1024, 1024, 7, 0);
    measure::<3>(&r, &frozen, 1920, 1080, 5, 0);
    measure::<3>(&r, &frozen, 1024, 1024, 5, 2);
    measure::<1>(&r, &frozen, 1024, 1024, 7, 0);
}

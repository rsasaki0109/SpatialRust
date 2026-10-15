use super::*;
use std::{hint::black_box, time::Instant};

fn measure<const C: usize>(r: &WgpuRuntime, w: usize, h: usize) {
    let image = Image::<u8, C>::try_new(
        w,
        h,
        (0..w * h * C).map(|i| (i * 37 + i / C * 11) as u8).collect(),
    )
    .unwrap();
    let source = GpuImage::upload_u8(r, image.view()).unwrap();
    let mean = [1.0, -2.0, 3.0, -4.0];
    let std = [2.0, 4.0, 0.5, 1.0];
    let mut tensors = [
        pack_ai_chw_gpu(r, &source, 0.5, mean, std).unwrap(),
        pack_ai_chw_gpu(r, &source, 0.5, mean, std).unwrap(),
    ];
    let expected: Vec<u32> = (0..C)
        .flat_map(|c| {
            let data = image.as_slice();
            (0..w * h)
                .map(move |i| ((f32::from(data[i * C + c]) * 0.5 - mean[c]) / std[c]).to_bits())
        })
        .collect();
    r.wait_idle();
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let mode = (round + offset) % 2;
            let start = Instant::now();
            let result = if mode == 0 {
                black_box(&mut tensors[mode]).parent_readback_f32(r).unwrap()
            } else {
                black_box(&mut tensors[mode]).readback_f32(r).unwrap()
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_bits(&result, &expected);
            if round >= 3 {
                samples[mode].push(elapsed);
            }
        }
    }
    assert_eq!(tensors[0].receipt(), tensors[1].receipt());
    assert_eq!(tensors[1].receipt().device_to_host_bytes(), (w * h * C * 4 * 33) as u64);
    for tensor in tensors {
        tensor.recycle(r);
    }
    source.recycle(r);
    println!("AI_READBACK_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"read_old_ms\":{:?},\"read_ms\":{:?}}}",samples[0],samples[1]);
}

#[test]
#[ignore = "paired complete AI tensor downloads; run alone in release mode"]
fn benchmark_complete_ai_tensor_readbacks() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("AI_READBACK_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_tensor_download output_drop_asserts_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    measure::<3>(&r, 16, 16);
    measure::<3>(&r, 224, 224);
    measure::<3>(&r, 1024, 1024);
    measure::<3>(&r, 1920, 1080);
    measure::<1>(&r, 1024, 1024);
    measure::<2>(&r, 1024, 1024);
    measure::<4>(&r, 1024, 1024);
    measure::<4>(&r, 3840, 2160);
}

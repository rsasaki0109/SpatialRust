use super::*;
use std::{hint::black_box, time::Instant};

fn measure<const C: usize>(r: &WgpuRuntime, w: usize, h: usize, roi: bool) {
    let (source, stride, metadata) = fixture::<C>(w, h, roi);
    let input = view::<C>(w, h, roi, &source, stride, metadata);
    let expected: Vec<u8> = (0..h).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
    let mut images =
        [GpuImage::upload_u8(r, input).unwrap(), GpuImage::upload_u8(r, input).unwrap()];
    r.queue().submit(std::iter::empty());
    r.wait_idle();
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let mode = (round + offset) % 2;
            let start = Instant::now();
            let result = if mode == 0 {
                black_box(&mut images[mode]).parent_readback_u8::<C>(r).unwrap()
            } else {
                black_box(&mut images[mode]).readback_u8::<C>(r).unwrap()
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(result.as_slice(), expected);
            assert_eq!(result.metadata(), metadata);
            if round >= 3 {
                samples[mode].push(elapsed);
            }
        }
    }
    assert_eq!(images[0].receipt(), images[1].receipt());
    assert_eq!(images[1].receipt().device_to_host_bytes(), (w * h * 4 * 33) as u64);
    for image in images {
        image.recycle(r);
    }
    println!("MAPPED_READBACK_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"roi\":{roi},\"read_old_ms\":{:?},\"read_ms\":{:?}}}", samples[0], samples[1]);
}

#[test]
#[ignore = "paired completed public readbacks; run alone in release mode"]
fn benchmark_completed_direct_mapped_readbacks() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("MAPPED_READBACK_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_readback output_drop_asserts_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (w, h) in [(64, 3), (65, 3), (1024, 1024), (1921, 1080), (3840, 2160)] {
        measure::<4>(&r, w, h, false);
    }
    measure::<1>(&r, 1024, 1024, false);
    measure::<2>(&r, 1920, 1080, false);
    measure::<3>(&r, 1920, 1080, true);
}

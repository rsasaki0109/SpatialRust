use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired current-parent host and completed GPU readbacks; run alone in release mode"]
fn benchmark_contiguous_rgba_readbacks() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("RGBA_CONTIGUOUS_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 output_drop_assertions_excluded=true", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for (w, h) in [(64, 3), (65, 3), (1920, 1080), (1921, 1080), (3840, 2160)] {
        let (source, stride, metadata) = fixture::<4>(w, h, false);
        let input = view::<4>(w, h, false, &source, stride, metadata);
        let expected: Vec<_> = (0..h).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
        let padded_row = (w * 4).div_ceil(256) * 256;
        let mut rgba = vec![253; padded_row * h];
        for y in 0..h {
            rgba[y * padded_row..y * padded_row + w * 4]
                .copy_from_slice(&expected[y * w * 4..(y + 1) * w * 4]);
        }
        let mut host: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let mode = (round + offset) % 2;
                let start = Instant::now();
                let result = if mode == 0 {
                    contiguous_parent::baseline_unpack_rgba8::<4>(
                        black_box(&rgba),
                        w,
                        h,
                        padded_row,
                        expected.len(),
                    )
                } else {
                    unpack_rgba8::<4>(black_box(&rgba), w, h, padded_row, expected.len())
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result, expected);
                if round >= 3 {
                    host[mode].push(elapsed);
                }
            }
        }
        println!("RGBA_CONTIGUOUS_RECORD {{\"boundary\":\"host_unpack_only\",\"width\":{w},\"height\":{h},\"padded_row\":{padded_row},\"old_ms\":{:?},\"new_ms\":{:?}}}", host[0], host[1]);
        let mut images =
            [GpuImage::upload_u8(&r, input).unwrap(), GpuImage::upload_u8(&r, input).unwrap()];
        r.queue().submit(std::iter::empty());
        r.wait_idle();
        let mut complete: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let mode = (round + offset) % 2;
                let start = Instant::now();
                let result = if mode == 0 {
                    black_box(&mut images[mode]).baseline_contiguous_readback_u8::<4>(&r).unwrap()
                } else {
                    black_box(&mut images[mode]).readback_u8::<4>(&r).unwrap()
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result.as_slice(), expected);
                assert_eq!(result.metadata(), metadata);
                if round >= 3 {
                    complete[mode].push(elapsed);
                }
            }
        }
        assert_eq!(images[0].receipt(), images[1].receipt());
        assert_eq!(images[1].receipt().device_to_host_bytes(), (w * h * 4 * 33) as u64);
        for image in images {
            image.recycle(&r);
        }
        println!("RGBA_CONTIGUOUS_RECORD {{\"boundary\":\"completed_public_readback\",\"width\":{w},\"height\":{h},\"padded_row\":{padded_row},\"old_ms\":{:?},\"new_ms\":{:?}}}", complete[0], complete[1]);
    }
}

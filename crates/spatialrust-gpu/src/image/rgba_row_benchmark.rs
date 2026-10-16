use super::*;
use std::{hint::black_box, time::Instant};

fn measure<T>(
    boundary: &str,
    shape: (usize, usize, bool, usize),
    mut run: impl FnMut(usize) -> T,
    mut check_and_drop: impl FnMut(T),
) {
    let (w, h, roi, padded_row) = shape;
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..2 {
            let mode = (round + offset) % 2;
            let start = Instant::now();
            let output = run(mode);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            check_and_drop(output);
            if round >= 3 {
                samples[mode].push(elapsed);
            }
        }
    }
    println!("RGBA_ROWS_RECORD {{\"boundary\":\"{boundary}\",\"width\":{w},\"height\":{h},\"roi\":{roi},\"padded_row\":{padded_row},\"old_ms\":{:?},\"new_ms\":{:?}}}", samples[0], samples[1]);
}

#[test]
#[ignore = "paired immediate-parent row packing and GPU transfers; run alone in release mode"]
fn benchmark_rgba_owned_row_transfers() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("RGBA_ROWS_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 output_drop_assertions_recycle_excluded=true", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for (w, h, roi) in
        [(64, 3, false), (65, 3, true), (1920, 1080, true), (1921, 1080, true), (3840, 2160, true)]
    {
        let (source, stride, metadata) = fixture::<4>(w, h, roi);
        let input = view::<4>(w, h, roi, &source, stride, metadata);
        let expected: Vec<_> = (0..h).flat_map(|y| input.row(y).unwrap().iter().copied()).collect();
        let padded_row = (w * 4).div_ceil(256) * 256;
        let shape = (w, h, roi, padded_row);
        measure(
            "host_pack_only",
            shape,
            |mode| {
                if mode == 0 {
                    row_parent::row_parent_pack_rgba8(black_box(input), expected.len())
                } else {
                    row_parent::row_candidate_pack_rgba8(black_box(input), expected.len())
                }
            },
            |output| {
                assert_eq!(output, expected);
            },
        );
        measure(
            "public_upload_and_wait_idle",
            shape,
            |mode| {
                let image = if mode == 0 {
                    GpuImage::row_parent_upload_u8(&r, black_box(input)).unwrap()
                } else {
                    GpuImage::row_candidate_upload_u8(&r, black_box(input)).unwrap()
                };
                // The production upload is asynchronous; completion is explicit here.
                r.queue().submit(std::iter::empty());
                r.wait_idle();
                image
            },
            |mut output| {
                assert_eq!(output.metadata(), metadata);
                assert_eq!(output.receipt().host_to_device_bytes(), expected.len() as u64);
                assert_eq!(output.receipt().device_to_host_bytes(), 0);
                assert_eq!(output.readback_u8::<4>(&r).unwrap().as_slice(), expected);
                output.recycle(&r);
            },
        );
        let mut rgba = vec![253; padded_row * h];
        for y in 0..h {
            rgba[y * padded_row..y * padded_row + w * 4]
                .copy_from_slice(&expected[y * w * 4..(y + 1) * w * 4]);
        }
        measure(
            "host_unpack_only",
            shape,
            |mode| {
                if mode == 0 {
                    row_parent::row_parent_unpack_rgba8::<4>(
                        black_box(&rgba),
                        w,
                        h,
                        padded_row,
                        expected.len(),
                    )
                } else {
                    row_parent::row_candidate_unpack_rgba8::<4>(
                        black_box(&rgba),
                        w,
                        h,
                        padded_row,
                        expected.len(),
                    )
                }
            },
            |output| {
                assert_eq!(output, expected);
            },
        );
        let mut images =
            [GpuImage::upload_u8(&r, input).unwrap(), GpuImage::upload_u8(&r, input).unwrap()];
        r.queue().submit(std::iter::empty());
        r.wait_idle();
        measure(
            "completed_public_readback",
            shape,
            |mode| {
                if mode == 0 {
                    images[mode].row_parent_readback_u8::<4>(&r).unwrap()
                } else {
                    images[mode].row_candidate_readback_u8::<4>(&r).unwrap()
                }
            },
            |output| {
                assert_eq!(output.as_slice(), expected);
                assert_eq!(output.metadata(), metadata);
            },
        );
        assert_eq!(images[0].receipt(), images[1].receipt());
        assert_eq!(images[1].receipt().device_to_host_bytes(), (expected.len() * 33) as u64);
        for image in images {
            image.recycle(&r);
        }
    }
}

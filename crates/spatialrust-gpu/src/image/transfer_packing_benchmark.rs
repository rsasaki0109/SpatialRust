use super::*;
use std::{hint::black_box, time::Instant};

fn measure<const CHANNELS: usize>() {
    for (width, height) in [(640, 480), (1920, 1080), (3840, 2160)] {
        let stride = width * CHANNELS + 11;
        let source: Vec<_> = (0..stride * height).map(|i| (i % 251) as u8).collect();
        let view = ImageView::<u8, CHANNELS>::new(width, height, stride, &source).unwrap();
        let rgba = previous::pack_rgba8(view, width * height * 4);
        let padded_row = (width * 4).div_ceil(256) * 256;
        let mut padded = vec![253; padded_row * height];
        for (y, row) in rgba.chunks_exact(width * 4).enumerate() {
            padded[y * padded_row..y * padded_row + width * 4].copy_from_slice(row);
        }
        for unpack in [false, true] {
            let expected = if unpack {
                previous::unpack_rgba8::<CHANNELS>(
                    &padded,
                    width,
                    height,
                    padded_row,
                    width * height * CHANNELS,
                )
            } else {
                rgba.clone()
            };
            let run = |current| {
                let start = Instant::now();
                let result = if unpack {
                    if current {
                        unpack_rgba8::<CHANNELS>(
                            black_box(&padded),
                            width,
                            height,
                            padded_row,
                            width * height * CHANNELS,
                        )
                    } else {
                        previous::unpack_rgba8::<CHANNELS>(
                            black_box(&padded),
                            width,
                            height,
                            padded_row,
                            width * height * CHANNELS,
                        )
                    }
                } else if current {
                    pack_rgba8(black_box(view), rgba.len())
                } else {
                    previous::pack_rgba8(black_box(view), rgba.len())
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result, expected);
                black_box(result);
                elapsed
            };
            for _ in 0..3 {
                run(false);
                run(true);
            }
            let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for pair in 0..20 {
                for offset in 0..2 {
                    let mode = (pair + offset) % 2;
                    times[mode].push(run(mode == 1));
                }
            }
            eprintln!("PAIRED {{\"channels\":{CHANNELS},\"width\":{width},\"height\":{height},\"unpack\":{unpack},\"workers\":{},\"previous_ms\":{:?},\"current_ms\":{:?}}}", transfer_workers(width,height,CHANNELS),times[0],times[1]);
        }
    }
}

#[test]
#[ignore = "paired completed CPU image transfer transforms; run alone in release mode"]
fn paired_serial_current_transfer_rows() {
    eprintln!("CONDITIONS available_workers={} warmups=3 pairs=20 boundary=completed_host_transform allocation_included=true", std::thread::available_parallelism().map_or(1, |n| n.get()));
    measure::<1>();
    measure::<2>();
    measure::<3>();
    measure::<4>();
}

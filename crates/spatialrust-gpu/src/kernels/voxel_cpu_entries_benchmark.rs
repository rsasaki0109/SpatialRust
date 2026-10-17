use super::*;
use std::{hint::black_box, time::Instant};

fn prepare_time(data: &[(i64, i64, i64)], current: bool, batch: usize) -> f64 {
    let oracle = clamped(data);
    if current {
        let mut outputs = Vec::with_capacity(batch);
        let start = Instant::now();
        for _ in 0..batch {
            outputs.push(black_box(prepare_cpu_sort_entries(black_box(data))));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000. / batch as f64;
        for output in outputs {
            assert_eq!(output.len(), data.len());
            for (i, (entry, &(x, y, z))) in output.iter().zip(&oracle).enumerate() {
                assert_eq!([entry.ix, entry.iy, entry.iz], [x as i32, y as i32, z as i32]);
                assert_eq!(entry.point_index, i as u32);
            }
        }
        elapsed
    } else {
        let mut outputs = Vec::with_capacity(batch);
        let start = Instant::now();
        for _ in 0..batch {
            outputs.push(black_box(old::prepare(black_box(data))));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000. / batch as f64;
        for output in outputs {
            assert_eq!(output.len(), data.len());
            for (entry, &(x, y, z)) in output.iter().zip(&oracle) {
                assert_eq!([entry.ix, entry.iy, entry.iz], [x as i32, y as i32, z as i32]);
                assert_eq!(entry._pad, 0);
            }
        }
        elapsed
    }
}

fn measure(r: &WgpuRuntime, n: usize, pattern: usize) {
    let data = keys(n, pattern);
    let reference = expected(&data);
    let prep_batch = if n <= 1024 { 64 } else { 1 };
    let mut samples: [[Vec<f64>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        for route_offset in 0..3 {
            let route = (route_offset + round) % 3;
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let current = variant == 1;
                let elapsed = match route {
                    0 => prepare_time(&data, current, prep_batch),
                    1 => {
                        let start = Instant::now();
                        let output = if current {
                            build_voxel_segments_gpu_from_keys(black_box(r), black_box(&data))
                        } else {
                            old::previous_from_keys(black_box(r), black_box(&data))
                        }
                        .unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.;
                        assert_eq!(output.to_voxel_segments(r).unwrap(), reference);
                        assert_eq!(
                            output.construction_readback_bytes(),
                            if n == 0 { 0 } else { 4 }
                        );
                        elapsed
                    }
                    _ => {
                        let start = Instant::now();
                        let output = if current {
                            build_voxel_segments_gpu(black_box(r), black_box(&data))
                        } else {
                            old::previous(black_box(r), black_box(&data))
                        }
                        .unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.;
                        assert_eq!(output, reference);
                        elapsed
                    }
                };
                if round >= 3 {
                    samples[route][variant].push(elapsed);
                }
            }
        }
    }
    println!("VOXEL_CPU_ENTRIES_RECORD {{\"count\":{n},\"pattern\":{pattern},\"prep_batch\":{prep_batch},\"prep_old_ms\":{:?},\"prep_ms\":{:?},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}", samples[0][0], samples[0][1], samples[1][0], samples[1][1], samples[2][0], samples[2][1]);
}

#[test]
#[ignore = "paired CPU preparation and completed CPU-key resident/full calls; run alone in release mode"]
fn benchmark_complete_voxel_cpu_entries_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_CPU_ENTRIES_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n|n.get()));
    for (n, pattern) in [
        (0, 0),
        (1, 2),
        (2, 0),
        (256, 2),
        (257, 2),
        (65_537, 0),
        (65_537, 1),
        (524_288, 0),
        (524_289, 0),
        (1_000_003, 0),
        (1_000_003, 1),
        (1_000_003, 2),
    ] {
        measure(&r, n, pattern);
    }
}

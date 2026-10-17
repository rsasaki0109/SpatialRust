use super::*;
use std::{hint::black_box, time::Instant};

fn prepare_time(data: &[(i64, i64, i64)], current: bool, one: bool, batch: usize) -> f64 {
    let mut outputs = Vec::with_capacity(batch);
    let start = Instant::now();
    for _ in 0..batch {
        let output = if !current {
            old::serial(black_box(data))
        } else if one {
            prepare_with_workers(black_box(data), 1)
        } else {
            prepare(black_box(data))
        };
        outputs.push(black_box(output));
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000. / batch as f64;
    let oracle = old::serial(data);
    for output in outputs {
        assert_eq!(output.len(), oracle.len());
        assert_eq!(output.capacity(), oracle.capacity());
        assert_eq!(
            bytemuck::cast_slice::<VoxelSortEntry, u32>(&output),
            bytemuck::cast_slice::<VoxelSortEntry, u32>(&oracle)
        );
    }
    elapsed
}

fn measure(r: &WgpuRuntime, n: usize, pattern: usize) {
    let data = keys(n, pattern);
    let reference = expected(&data);
    let prep_batch = if n <= 1024 { 64 } else { 1 };
    let selected_workers = workers(n, std::thread::available_parallelism().map_or(1, |n| n.get()));
    let mut samples: [[Vec<f64>; 2]; 4] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        for route_offset in 0..4 {
            let route = (route_offset + round) % 4;
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let current = variant == 1;
                let elapsed = match route {
                    0 | 1 => prepare_time(&data, current, route == 1, prep_batch),
                    2 => {
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
    println!("VOXEL_ENTRY_PREPARATION_RECORD {{\"count\":{n},\"pattern\":{pattern},\"prep_batch\":{prep_batch},\"selected_workers\":{selected_workers},\"prep_old_ms\":{:?},\"prep_ms\":{:?},\"one_old_ms\":{:?},\"one_ms\":{:?},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}", samples[0][0], samples[0][1], samples[1][0], samples[1][1], samples[2][0], samples[2][1], samples[3][0], samples[3][1]);
}

#[test]
#[ignore = "paired host preparation/one-worker controls and complete CPU-key resident/full calls; run alone in release mode"]
fn benchmark_complete_voxel_entry_preparation_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_ENTRY_PREPARATION_ENV backend={} device_type={} available_workers={} max_workers={} min_points={} points_per_worker={} warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n|n.get()), MAX_WORKERS, PARALLEL_MIN_POINTS, POINTS_PER_WORKER);
    for (n, pattern) in [
        (0, 0),
        (1, 2),
        (257, 2),
        (65_537, 0),
        (524_287, 0),
        (524_288, 0),
        (524_289, 0),
        (1_000_003, 0),
        (1_000_003, 1),
        (1_000_003, 2),
        (2_000_003, 0),
        (2_000_003, 1),
        (2_000_003, 2),
    ] {
        measure(&r, n, pattern);
    }
}

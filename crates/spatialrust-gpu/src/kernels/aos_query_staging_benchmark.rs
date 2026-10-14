use super::*;
use crate::{GpuQueryBatch, WgpuRuntime};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_serial_and_bounded_aos_query_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    let limits = r.device().limits();
    println!("AOS_QUERY_ENV backend={} device_type={} available_workers={} max_workers={} storage_bytes={} dispatch_groups={}", info.backend, info.device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),MAX_WORKERS,limits.max_storage_buffer_binding_size,limits.max_compute_workgroups_per_dimension);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (524_287, 0),
        (524_288, 0),
        (524_289, 0),
        (1_048_577, 0),
        (524_289, 1),
        (524_289, 2),
        (524_289, 3),
    ] {
        let rows = fixture(n, profile);
        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for family in 0..2 {
                for offset in 0..2 {
                    let route = (round + offset) % 2;
                    let batches = if family == 0 && n < PARALLEL_MIN_POINTS { 32 } else { 1 };
                    let start = Instant::now();
                    for _ in 0..batches {
                        if family == 0 {
                            let prepared = if route == 0 {
                                previous(black_box(&rows))
                            } else {
                                prepare(black_box(&rows))
                            }
                            .unwrap();
                            black_box(&prepared);
                            drop(prepared);
                        } else {
                            let input = if route == 0 {
                                super::super::super::aos_upload_reference::previous_upload(
                                    &r,
                                    black_box(&rows),
                                )
                            } else {
                                GpuQueryBatch::upload(&r, black_box(&rows))
                            }
                            .unwrap();
                            r.wait_idle();
                            assert_eq!(input.query_count(), n as u32);
                            let receipt = input.upload_transfers();
                            assert_eq!(receipt.host_to_device_bytes(), n as u64 * 16);
                            assert_eq!(receipt.device_to_device_bytes(), 0);
                            assert_eq!(receipt.device_to_host_bytes(), 0);
                            drop(input);
                        }
                    }
                    let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(batches);
                    if round >= 3 {
                        samples[family * 2 + route].push(ms);
                    }
                }
            }
        }
        println!("AOS_QUERY_RECORD {{\"points\":{n},\"profile\":{profile},\"host_old_ms\":{:?},\"host_ms\":{:?},\"upload_old_ms\":{:?},\"upload_ms\":{:?}}}",samples[0],samples[1],samples[2],samples[3]);
    }
}

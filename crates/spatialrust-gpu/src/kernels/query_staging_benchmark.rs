use super::super::super::{GpuQueryBatch, WgpuRuntime};
use super::*;

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_host_and_query_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!("QUERY_STAGING_ENV backend={} device_type={} available_workers={} max_workers={} min_points={} points_per_worker={}", info.backend, info.device_type, std::thread::available_parallelism().map_or(1, |n| n.get()), MAX_WORKERS, PARALLEL_MIN_POINTS, POINTS_PER_WORKER);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (262_145, 0),
        (524_287, 0),
        (524_288, 0),
        (524_289, 0),
        (1_048_577, 0),
        (2_097_153, 0),
        (1_048_577, 1),
        (1_048_577, 2),
    ] {
        let data = fixture(n, profile);
        let axes = [&data[0][..], &data[1][..], &data[2][..]];
        let mut samples: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..5 {
                let route = (round + offset) % 5;
                let start = Instant::now();
                let (count, receipt) = if route < 3 {
                    let rows = match route {
                        0 => previous_prepare(black_box(axes)),
                        1 => prepare_with_workers(black_box(axes), 1),
                        _ => prepare(black_box(axes)),
                    }
                    .unwrap();
                    let count = rows.len();
                    black_box(&rows);
                    drop(rows);
                    (count, None)
                } else {
                    let batch = if route == 3 {
                        gpu::previous_upload(&r, black_box(axes))
                    } else {
                        GpuQueryBatch::upload_soa(&r, black_box(axes))
                    }
                    .unwrap();
                    r.wait_idle();
                    let count = batch.query_count() as usize;
                    let receipt = batch.upload_transfers();
                    drop(batch);
                    (count, Some(receipt))
                };
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(count, n);
                if let Some(receipt) = receipt {
                    assert_eq!(receipt.host_to_device_bytes(), n as u64 * 16);
                    assert_eq!(receipt.device_to_host_bytes(), 0);
                }
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("QUERY_STAGING_RECORD {{\"points\":{n},\"profile\":{profile},\"workers\":{},\"host_old_ms\":{:?},\"host_one_ms\":{:?},\"host_new_ms\":{:?},\"upload_old_ms\":{:?},\"upload_new_ms\":{:?}}}", worker_count(n, std::thread::available_parallelism().map_or(1, |n| n.get())), samples[0], samples[1], samples[2], samples[3], samples[4]);
    }
}

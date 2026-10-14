use super::super::super::{GpuKdTree, WgpuRuntime};
use super::*;

#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_host_and_tree_uploads() {
    let r = WgpuRuntime::new_headless().unwrap();
    let info = r.adapter_info();
    println!("TREE_STAGING_ENV backend={} device_type={} available_workers={} max_workers={} min_points={} points_per_worker={}",info.backend,info.device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),MAX_WORKERS,PARALLEL_MIN_POINTS,POINTS_PER_WORKER);
    for (n, profile) in [
        (1, 0),
        (4097, 0),
        (65_537, 0),
        (131_073, 0),
        (262_145, 0),
        (524_287, 0),
        (524_288, 0),
        (524_289, 0),
        (1_048_577, 0),
        (1_048_577, 1),
        (1_048_577, 2),
    ] {
        let data = fixture(n, profile);
        let tree = KdTree::from_slices(&data[0], &data[1], &data[2]);
        let layout = tree.layout();
        let (x, y, z) = layout.positions();
        let axes = [x, y, z];
        let order = layout.point_order();
        let nodes = layout.nodes().len();
        let mut samples: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..5 {
                let route = (round + offset) % 5;
                let start = Instant::now();
                let receipt = if route < 3 {
                    let rows = match route {
                        0 => previous_prepare(black_box(axes), black_box(order)),
                        1 => prepare_with_workers(black_box(axes), black_box(order), 1),
                        _ => prepare(black_box(axes), black_box(order)),
                    }
                    .unwrap();
                    black_box(&rows);
                    drop(rows);
                    None
                } else {
                    let target = if route == 3 {
                        gpu::previous_upload(&r, black_box(&tree))
                    } else {
                        GpuKdTree::upload(&r, black_box(&tree))
                    }
                    .unwrap();
                    r.wait_idle();
                    let receipt = target.upload_transfers();
                    drop(target);
                    Some(receipt)
                };
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if let Some(receipt) = receipt {
                    assert_eq!(receipt.host_to_device_bytes(), n as u64 * 16 + nodes as u64 * 32);
                    assert_eq!(receipt.device_to_host_bytes(), 0);
                }
                if round >= 3 {
                    samples[route].push(ms);
                }
            }
        }
        println!("TREE_STAGING_RECORD {{\"points\":{n},\"profile\":{profile},\"workers\":{},\"nodes\":{nodes},\"host_old_ms\":{:?},\"host_one_ms\":{:?},\"host_new_ms\":{:?},\"upload_old_ms\":{:?},\"upload_new_ms\":{:?}}}",worker_count(n,std::thread::available_parallelism().map_or(1,|n|n.get())),samples[0],samples[1],samples[2],samples[3],samples[4]);
    }
}

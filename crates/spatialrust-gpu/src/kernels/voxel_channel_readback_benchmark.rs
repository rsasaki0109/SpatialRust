use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete host-input multi-channel voxel calls; run alone in release mode"]
fn benchmark_complete_voxel_channel_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_CHANNEL_ENV backend={} device_type={} available_workers={} gather_batch_channels={} warmups=3 pairs=30 boundary=complete_public_host_input_call output_drop_asserts_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),r.max_gather_channels());
    for (cells, channels) in [
        (16, 2),
        (16, 4),
        (4096, 4),
        (262_144, 2),
        (262_144, 4),
        (262_144, 7),
        (1_048_576, 2),
        (1_048_576, 4),
    ] {
        let f = fixture(&r, cells, channels, 4);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        r.wait_idle();
        let mut routes: [[Vec<f64>; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for (reduce, samples) in routes.iter_mut().enumerate() {
            let expected = if reduce == 1 { &f.mean } else { &f.first };
            for round in 0..33 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = Instant::now();
                    let output = call(
                        black_box(&r),
                        black_box(&refs),
                        black_box(&f.segments),
                        reduce == 1,
                        mode == 0,
                    )
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_bits(&output, expected);
                    assert_eq!(
                        capacity(&output),
                        if mode == 0 {
                            parent_capacity(&r, cells, channels, reduce == 1)
                        } else {
                            cells * channels
                        }
                    );
                    if round >= 3 {
                        samples[mode].push(elapsed);
                    }
                }
            }
        }
        println!("VOXEL_CHANNEL_RECORD {{\"cells\":{cells},\"channels\":{channels},\"points_per_cell\":4,\"gather_old_ms\":{:?},\"gather_ms\":{:?},\"reduce_old_ms\":{:?},\"reduce_ms\":{:?}}}",routes[0][0],routes[0][1],routes[1][0],routes[1][1]);
    }
}

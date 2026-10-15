use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete resident-input voxel XYZ calls; run alone in release mode"]
fn benchmark_complete_voxel_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_XYZ_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_input_call output_drop_asserts_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for cells in [16, 4096, 262_144, 1_048_576] {
        for occupancy in [1, 4] {
            let f = fixture(&r, cells, occupancy);
            r.wait_idle();
            let mut routes: [[Vec<f64>; 2]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
            for (reduce, samples) in routes.iter_mut().enumerate() {
                let expected = if reduce == 1 { &f.mean } else { &f.first };
                for round in 0..33 {
                    for offset in 0..2 {
                        let mode = (round + offset) % 2;
                        let start = Instant::now();
                        let output =
                            call(black_box(&r), black_box(&f), reduce == 1, mode == 0).unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        assert_bits(&output, expected);
                        if round >= 3 {
                            samples[mode].push(elapsed);
                        }
                    }
                }
            }
            println!("VOXEL_XYZ_RECORD {{\"cells\":{cells},\"occupancy\":{occupancy},\"gather_old_ms\":{:?},\"gather_ms\":{:?},\"reduce_old_ms\":{:?},\"reduce_ms\":{:?}}}",routes[0][0],routes[0][1],routes[1][0],routes[1][1]);
        }
    }
}

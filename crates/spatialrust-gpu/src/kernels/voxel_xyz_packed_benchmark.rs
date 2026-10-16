use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete resident-input packed XYZ calls; run alone in release mode"]
fn benchmark_complete_packed_voxel_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_XYZ_PACKED_ENV backend={} device_type={} warmups=3 pairs=30 boundary=complete_public_resident_input_call output_drop_asserts_excluded=true packed_supported={}",r.adapter_info().backend,r.adapter_info().device_type,plan_packed_xyz(&r.device().limits(),1_048_576,isize::MAX as u64).is_some());
    for cells in [16, 4096, 262_144, 1_048_576] {
        for occupancy in [1, 3] {
            let f = fixture(&r, cells, occupancy);
            r.wait_idle();
            let mut routes: [[Vec<f64>; 2]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
            for (route, samples) in routes.iter_mut().enumerate() {
                let mode = if route == 0 { XyzMode::First } else { XyzMode::Centroid };
                let expected = if route == 0 { &f.first } else { &f.mean };
                for round in 0..33 {
                    for offset in 0..2 {
                        let variant = (round + offset) % 2;
                        let start = Instant::now();
                        let output =
                            invoke(black_box(&r), black_box(&f), mode, variant == 0).unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.;
                        assert_bits(&output, expected);
                        assert_eq!(capacities(&output), [cells; 3]);
                        if round >= 3 {
                            samples[variant].push(elapsed);
                        }
                    }
                }
            }
            println!("VOXEL_XYZ_PACKED_RECORD {{\"cells\":{cells},\"occupancy\":{occupancy},\"gather_old_ms\":{:?},\"gather_ms\":{:?},\"reduce_old_ms\":{:?},\"reduce_ms\":{:?}}}",routes[0][0],routes[0][1],routes[1][0],routes[1][1]);
        }
    }
}

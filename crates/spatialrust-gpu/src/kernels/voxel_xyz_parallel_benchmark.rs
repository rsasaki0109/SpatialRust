use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-input XYZ calls; run alone in release mode"]
fn benchmark_complete_parallel_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_XYZ_PARALLEL_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_segments output_drop_asserts_excluded=true threshold_cells=1048576 max_copy_workers=3 max_child_threads=2",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, occupancy, mode) in [
        (16, 4, 0),
        (65_537, 4, 1),
        (1_048_575, 1, 0),
        (1_048_575, 1, 1),
        (1_048_576, 4, 0),
        (1_048_576, 4, 1),
        (1_048_577, 1, 0),
        (2_097_153, 1, 1),
    ] {
        let f = fixture(&r, cells, occupancy);
        let expected = invoke(&r, &f, mode == 1, true).unwrap();
        assert_bits(&expected, if mode == 1 { &f.mean } else { &f.first });
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let out = invoke(black_box(&r), black_box(&f), mode == 1, variant == 0).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_bits(&out, &expected);
                assert_capacity(&out, cells);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_XYZ_PARALLEL_RECORD {{\"cells\":{cells},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

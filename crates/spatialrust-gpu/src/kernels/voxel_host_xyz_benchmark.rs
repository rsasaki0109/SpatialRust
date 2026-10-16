use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete host XYZ centroid calls; run alone in release mode"]
fn benchmark_complete_host_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_HOST_XYZ_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_host_xyz output_drop_asserts_excluded=true scalar_pipeline=unchanged old_submits_maps_waits=3 new_submits_maps_waits=1",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, occupancy) in [
        (16, 4),
        (257, 4),
        (65_537, 1),
        (65_537, 4),
        (262_145, 1),
        (1_048_575, 1),
        (1_048_576, 1),
        (1_048_577, 1),
    ] {
        let f = fixture(cells, occupancy, 0);
        assert!(plan_host_xyz(&r.device().limits(), cells, cells * occupancy, isize::MAX as u64)
            .is_some());
        let expected = invoke(&r, &f, true).unwrap();
        assert_bits(&expected, &cpu_reference(&f));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let out = invoke(black_box(&r), black_box(&f), variant == 0).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_bits(&out, &expected);
                assert_capacity(&out, cells);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_HOST_XYZ_RECORD {{\"cells\":{cells},\"occupancy\":{occupancy},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

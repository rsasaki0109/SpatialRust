use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_fused_voxel_first_average_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_FIRST_AVG_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true fused_supported={}",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),plan_fused_voxel_average(&r.device().limits(),1_048_576,4).is_some());
    for (cells, attributes, u8_channels, occupancy) in [
        (16, 1, 0, 4),
        (16, 4, 3, 4),
        (4096, 4, 0, 4),
        (65_537, 3, 0, 3),
        (262_144, 1, 0, 4),
        (262_144, 4, 3, 4),
        (1_048_576, 1, 0, 4),
        (1_048_576, 4, 3, 4),
    ] {
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
        let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
        let expected = call_baseline(&r, &f, &refs, &bytes);
        assert_analytic(&expected, &expected_average(&f));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let result = if variant == 0 {
                    call_baseline(black_box(&r), black_box(&f), black_box(&refs), black_box(&bytes))
                } else {
                    invoke(
                        black_box(&r),
                        black_box(&f),
                        black_box(&refs),
                        black_box(&bytes),
                        1,
                        false,
                    )
                    .unwrap()
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_output(&result, &expected);
                assert_capacity(&result, cells, attributes, u8_channels, false);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_FIRST_AVG_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"average_old_ms\":{:?},\"average_ms\":{:?}}}",samples[0],samples[1]);
    }
}

use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_voxel_u8_batch_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_U8_BATCH_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true fused_supported={}",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),plan_batched_voxel_u8(&r.device().limits(),1_048_576,4).is_some());
    for (cells, u8_channels, occupancy, mode) in [
        (16, 2, 4, 0),
        (16, 3, 4, 1),
        (4096, 4, 4, 2),
        (65_537, 3, 4, 3),
        (262_144, 1, 4, 0),
        (262_144, 3, 4, 2),
        (1_048_576, 3, 4, 0),
        (1_048_576, 4, 4, 2),
    ] {
        let attributes = 4;
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let expected = call_baseline(&r, &f, mode);
        assert_output(&expected, &super::expected(&f, mode));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let result = if variant == 0 {
                    call_baseline(black_box(&r), black_box(&f), mode)
                } else {
                    call(black_box(&r), black_box(&f), mode, false)
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_output(&result, &expected);
                assert_capacity(&result, cells, attributes, u8_channels, false);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_U8_BATCH_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

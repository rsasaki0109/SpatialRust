use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute combined calls; run alone in release mode"]
fn benchmark_complete_packed_voxel_gather_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_PACKED_GATHER_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true batch_supported={}",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),plan_packed_voxel_attributes(&r.device().limits(),1_048_576,7,r.max_gather_channels() as usize,45_088_768,12_582_912,isize::MAX as u64).is_some());
    for (cells, attributes, u8_channels, mode) in [
        (16, 4, 0, 0),
        (16, 7, 3, 3),
        (4096, 5, 0, 0),
        (65_537, 7, 3, 3),
        (262_144, 5, 3, 0),
        (262_144, 9, 0, 3),
        (1_048_576, 7, 3, 0),
        (1_048_576, 8, 0, 3),
    ] {
        let occupancy = 4;
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let expected = baseline_call(&r, &f, mode);
        assert_output(&expected, &super::expected(&f, mode));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = if variant == 0 {
                    baseline_call(black_box(&r), black_box(&f), mode)
                } else {
                    call(black_box(&r), black_box(&f), mode, false)
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_output(&output, &expected);
                assert_capacity(&output, cells, attributes, u8_channels, false);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_PACKED_GATHER_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

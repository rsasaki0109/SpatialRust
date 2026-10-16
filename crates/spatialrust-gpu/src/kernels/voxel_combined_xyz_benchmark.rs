use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_combined_packed_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_COMBINED_XYZ_ENV backend={} device_type={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true packed_supported={}",r.adapter_info().backend,r.adapter_info().device_type,plan_packed_xyz_to_staging(&r.device().limits(),1_048_576,0,58_720_256,isize::MAX as u64).is_some());
    for (cells, attributes, u8_channels, mode) in [
        (16, 4, 0, 0),
        (16, 0, 3, 1),
        (4096, 5, 0, 2),
        (65_537, 7, 3, 3),
        (262_144, 7, 3, 0),
        (262_144, 9, 0, 1),
        (1_048_576, 7, 3, 2),
        (1_048_576, 8, 0, 3),
    ] {
        let occupancy = 4;
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
        let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
        let expected = complete_call(&r, &f, mode, true);
        assert_output(&expected, &super::expected(&f, mode));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = invoke_current_parent(
                    black_box(&r),
                    black_box(&f),
                    black_box(&refs),
                    black_box(&bytes),
                    mode,
                    variant == 0,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_output(&output, &expected);
                assert_capacity(&output, cells, attributes, u8_channels, false);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_COMBINED_XYZ_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

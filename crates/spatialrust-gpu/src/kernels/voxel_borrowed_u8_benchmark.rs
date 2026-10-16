use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_borrowed_u8_upload_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_BORROWED_U8_ENV backend={} device_type={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true aligned_u8_payload=borrowed unaligned_u8_payload=original_padded_vec",r.adapter_info().backend,r.adapter_info().device_type);
    for (cells, attributes, u8_channels, occupancy, mode) in [
        (16, 0, 1, 1, 0),
        (17, 0, 1, 1, 0),
        (65_536, 0, 3, 4, 1),
        (65_537, 0, 3, 1, 1),
        (1_048_576, 7, 3, 1, 2),
        (1_048_577, 7, 3, 1, 2),
        (1_048_576, 0, 7, 1, 3),
        (1_048_577, 0, 7, 1, 3),
    ] {
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
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
                    black_box(&attrs),
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
        println!("VOXEL_BORROWED_U8_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_combined_parallel_xyz_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_COMBINED_XYZ_PARALLEL_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true threshold_cells=1048576 xyz_max_copy_workers=3 xyz_max_child_threads=2 phases=attributes_bytes_xyz_sequential",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, attributes, u8_channels, occupancy, mode) in [
        (16, 4, 0, 4, 0),
        (65_537, 7, 3, 4, 1),
        (1_048_575, 0, 0, 1, 2),
        (1_048_575, 1, 1, 1, 3),
        (1_048_576, 0, 0, 1, 0),
        (1_048_576, 1, 3, 4, 1),
        (1_048_577, 7, 3, 1, 2),
        (2_097_153, 0, 1, 1, 3),
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
                let out = invoke_current_parent(
                    black_box(&r),
                    black_box(&f),
                    black_box(&attrs),
                    black_box(&bytes),
                    mode,
                    variant == 0,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_output(&out, &expected);
                assert_owned(&out, cells, attributes, u8_channels);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_COMBINED_XYZ_PARALLEL_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

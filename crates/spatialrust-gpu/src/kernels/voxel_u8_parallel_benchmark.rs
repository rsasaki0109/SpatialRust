use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_parallel_u8_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_U8_PARALLEL_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true threshold_cells=4194304 min_bytes_per_channel=4194304 u8_max_child_threads=4 phases=attributes_bytes_xyz_sequential",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, attributes, u8_channels, occupancy, mode) in [
        (16, 4, 3, 4, 0),
        (65_537, 7, 3, 4, 1),
        (4_194_303, 0, 3, 1, 2),
        (4_194_304, 0, 1, 1, 3),
        (4_194_304, 0, 2, 1, 0),
        (4_194_305, 1, 3, 1, 1),
        (4_194_306, 0, 5, 1, 2),
        (4_194_307, 2, 7, 1, 3),
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
        println!("VOXEL_U8_PARALLEL_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

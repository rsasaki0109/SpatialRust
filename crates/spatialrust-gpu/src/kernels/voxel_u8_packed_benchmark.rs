use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete public resident-XYZ/host-attribute calls; run alone in release mode"]
fn benchmark_complete_packed_voxel_u8_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_U8_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_xyz_host_attributes output_drop_asserts_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, attributes, u8_channels) in [
        (16, 0, 1),
        (17, 4, 3),
        (4_096, 0, 3),
        (65_537, 4, 3),
        (262_144, 0, 1),
        (262_144, 4, 3),
        (1_048_576, 0, 3),
        (1_048_576, 4, 3),
    ] {
        let f = fixture(&r, cells, attributes, u8_channels, 4);
        let refs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
        let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
        r.wait_idle();
        let mut routes: [[Vec<f64>; 2]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for (mode, samples) in routes.iter_mut().enumerate() {
            let expected = expected(&f, mode);
            for round in 0..33 {
                for offset in 0..2 {
                    let variant = (round + offset) % 2;
                    let start = Instant::now();
                    let result = if variant == 0 {
                        invoke_baseline(
                            black_box(&r),
                            black_box(&f),
                            black_box(&refs),
                            black_box(&bytes),
                            mode,
                        )
                    } else {
                        invoke(
                            black_box(&r),
                            black_box(&f),
                            black_box(&refs),
                            black_box(&bytes),
                            mode,
                            false,
                        )
                    }
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_output(&result, &expected);
                    assert_capacity(&result, cells, attributes, u8_channels, false);
                    if round >= 3 {
                        samples[variant].push(elapsed);
                    }
                }
            }
        }
        println!("VOXEL_U8_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"points_per_cell\":4,\"mode0_old_ms\":{:?},\"mode0_ms\":{:?},\"mode1_old_ms\":{:?},\"mode1_ms\":{:?},\"mode2_old_ms\":{:?},\"mode2_ms\":{:?},\"mode3_old_ms\":{:?},\"mode3_ms\":{:?}}}",routes[0][0],routes[0][1],routes[1][0],routes[1][1],routes[2][0],routes[2][1],routes[3][0],routes[3][1]);
    }
}

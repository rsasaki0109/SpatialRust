use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete wide byte-attribute calls; run alone in release mode"]
fn benchmark_complete_wide_voxel_u8_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_U8_WIDE_ENV backend={} device_type={} available_workers={} byte_batch_width=4 scalar_single_tail=true warmups=3 pairs=30 boundary=complete_public_calls output_drop_asserts_excluded=true shaders=unchanged", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for (cells, attributes, u8_channels, occupancy, mode) in [
        (16, 4, 4, 4, 0),
        (257, 4, 5, 4, 1),
        (65_537, 7, 7, 4, 2),
        (65_537, 7, 9, 1, 3),
        (262_145, 4, 17, 1, 0),
        (262_145, 4, 17, 1, 1),
        (1_048_577, 4, 7, 1, 2),
        (1_048_577, 4, 5, 1, 3),
    ] {
        let f = fixture(&r, cells, attributes, u8_channels, occupancy);
        let attrs = f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
        let bytes = f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        if u8_channels > 4 {
            let offset = cells as u64 * 4 * (3 + attributes) as u64;
            let target = offset + cells.div_ceil(4) as u64 * 4 * u8_channels as u64;
            assert!(plan_wide_voxel_u8(
                &r.device().limits(),
                cells as u32,
                u8_channels,
                target,
                offset,
                isize::MAX as u64
            )
            .is_some());
        }
        let expected_output = call(&r, &f, mode, true);
        assert_output(&expected_output, &expected(&f, mode));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = invoke(
                    black_box(&r),
                    black_box(&f),
                    black_box(&attrs),
                    black_box(&bytes),
                    mode,
                    variant == 0,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_output(&output, &expected_output);
                assert_capacity(&output, cells, attributes, u8_channels);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_U8_WIDE_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"u8_channels\":{u8_channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}", samples[0], samples[1]);
    }
}

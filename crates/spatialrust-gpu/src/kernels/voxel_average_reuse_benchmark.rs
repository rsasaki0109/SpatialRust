use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete voxel average calls with private output reuse; run alone in release mode"]
fn benchmark_complete_voxel_average_reuse_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_AVERAGE_REUSE_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_calls output_drop_asserts_excluded=true shaders_and_dispatches=unchanged", r.adapter_info().backend, r.adapter_info().device_type, std::thread::available_parallelism().map_or(1, |n| n.get()));
    for (cells, attributes, mode) in [
        (16, 4, 0),
        (257, 5, 0),
        (65_537, 9, 0),
        (262_145, 17, 0),
        (65_537, 13, 1),
        (262_145, 17, 1),
        (65_537, 13, 2),
        (262_145, 17, 2),
    ] {
        let f = fixture(&r, cells, attributes, if mode == 0 { 0 } else { 3 }, 1);
        let attrs = f.source[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
        let bytes = f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert!(plan_voxel_average_batch(&r.device().limits(), cells as u32, attributes).is_some());
        if mode > 0 {
            assert!(crate::kernels::voxel_wide_average::plan_wide_voxel_average(
                &r.device().limits(),
                cells as u32,
                attributes,
                cells as u64 * (4 * (3 + attributes)) as u64 + cells.div_ceil(4) as u64 * 12,
                isize::MAX as u64
            )
            .is_some());
        }
        let expected = invoke(&r, &f, mode, true).unwrap();
        assert_bits(&expected, &cpu_output(&f, mode));
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = invoke_with_refs(
                    black_box(&r),
                    black_box(&f),
                    black_box(&attrs),
                    black_box(&bytes),
                    mode,
                    variant == 0,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_bits(&output, &expected);
                assert_capacity(&output, cells, attributes, f.bytes.len(), mode);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_AVERAGE_REUSE_RECORD {{\"cells\":{cells},\"attributes\":{attributes},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}", samples[0], samples[1]);
    }
}

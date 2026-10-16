use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete legacy fallback calls, excluding public selection; run alone in release mode"]
fn benchmark_complete_legacy_voxel_reuse_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    assert_eq!(r.max_gather_channels(), 4);
    println!("VOXEL_LEGACY_REUSE_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_private_fallback_call forced_widths=2,4 public_selection_validation_excluded=true output_drop_asserts_excluded=true shaders=unchanged",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, channels, width) in [
        (16, 5, 2),
        (257, 9, 4),
        (65_537, 5, 2),
        (65_537, 9, 4),
        (262_145, 11, 2),
        (262_145, 17, 4),
        (1_048_577, 5, 2),
        (1_048_577, 9, 4),
    ] {
        let f = fixture(&r, cells, channels, 0);
        let plan =
            plan_voxel_gather_submission(&r.device().limits(), cells as u32, channels, width)
                .unwrap();
        let expected = cpu(&f);
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        r.wait_idle();
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = invoke(black_box(&r), black_box(&f), plan, variant == 0);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_bits(&output, &expected);
                assert!(output
                    .iter()
                    .all(|plane| plane.len() == cells && plane.capacity() == cells));
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_LEGACY_REUSE_RECORD {{\"cells\":{cells},\"channels\":{channels},\"width\":{width},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired active/inactive recorder fixtures, bypassing public selection; run alone in release mode"]
fn benchmark_voxel_gather_inactive_output_fixtures() {
    let r = runtime4().expect("four-channel gather required for this explicit diagnostic");
    println!("VOXEL_GATHER_DUMMY_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=private_recording_fixture public_selection_validation_excluded=true guards_and_input_output_allocation_included=true output_drop_asserts_excluded=true shaders=unchanged",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, channels) in [
        (16, 3),
        (257, 4),
        (65_537, 3),
        (65_537, 4),
        (262_145, 3),
        (262_145, 4),
        (1_048_577, 3),
        (1_048_577, 4),
    ] {
        let f = fixture(&r, cells, channels, 0);
        let expected = expected_flat(&f, 0);
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let result = invoke(black_box(&r), black_box(&f), variant == 0, 0);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result, expected);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_GATHER_DUMMY_RECORD {{\"cells\":{cells},\"channels\":{channels},\"call_old_ms\":{:?},\"call_ms\":{:?}}}", samples[0], samples[1]);
    }
}

use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "complete host-channel calls; run alone in release mode"]
fn benchmark_complete_parallel_channel_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_CHANNEL_PARALLEL_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_resident_segments_host_channels output_drop_asserts_excluded=true threshold_cells=1048576 max_workers=4",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for (cells, channels, mode) in [
        (16, 4, 0),
        (262_144, 7, 1),
        (1_048_575, 2, 0),
        (1_048_576, 1, 1),
        (1_048_576, 2, 0),
        (1_048_577, 7, 1),
        (1_048_577, 9, 0),
        (2_097_153, 3, 1),
    ] {
        let occupancy = 4;
        let f = fixture(&r, cells, channels, occupancy);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let expected = invoke(&r, &refs, &f.segments, mode == 1, true).unwrap();
        assert_bits(&expected, if mode == 1 { &f.mean } else { &f.first });
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = invoke(
                    black_box(&r),
                    black_box(&refs),
                    black_box(&f.segments),
                    mode == 1,
                    variant == 0,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                assert_bits(&output, &expected);
                assert_capacity(&output, cells, channels);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_CHANNEL_PARALLEL_RECORD {{\"cells\":{cells},\"channels\":{channels},\"occupancy\":{occupancy},\"mode\":{mode},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

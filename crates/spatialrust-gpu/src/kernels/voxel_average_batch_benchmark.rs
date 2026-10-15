use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete host-input multi-channel voxel averages; run alone in release mode"]
fn benchmark_complete_voxel_average_batch_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_AVERAGE_BATCH_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 boundary=complete_public_host_input_call output_drop_asserts_excluded=true batch_supported={}",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),plan_voxel_average_batch(&r.device().limits(),1_048_576,7).is_some());
    for (cells, channels, occupancy) in [
        (16, 1, 4),
        (16, 4, 4),
        (4096, 7, 4),
        (65_537, 3, 4),
        (262_144, 2, 4),
        (262_144, 5, 4),
        (1_048_576, 4, 4),
        (1_048_576, 7, 4),
    ] {
        let f = fixture(&r, cells, channels, occupancy);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let expected = baseline_call(&r, &f);
        assert_bits(&expected, &f.mean);
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = if variant == 0 {
                    baseline::baseline_reduce_voxel_average_f32_multi_gpu(
                        black_box(&r),
                        black_box(&refs),
                        black_box(&f.segments),
                    )
                } else {
                    reduce_voxel_average_f32_multi_gpu(
                        black_box(&r),
                        black_box(&refs),
                        black_box(&f.segments),
                    )
                }
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_bits(&output, &expected);
                assert_eq!(capacity(&output), cells * channels);
                if round >= 3 {
                    samples[variant].push(elapsed);
                }
            }
        }
        println!("VOXEL_AVERAGE_BATCH_RECORD {{\"cells\":{cells},\"channels\":{channels},\"occupancy\":{occupancy},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

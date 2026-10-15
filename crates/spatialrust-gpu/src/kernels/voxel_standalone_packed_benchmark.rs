use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "paired complete host-input multi-channel voxel gathers; run alone in release mode"]
fn benchmark_complete_standalone_packed_gather_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    println!("VOXEL_STANDALONE_PACKED_ENV backend={} device_type={} available_workers={} batch_width={} warmups=3 pairs=30 boundary=complete_public_host_input_call output_drop_asserts_excluded=true packed_supported={}",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),r.max_gather_channels(),plan_packed_voxel_submission(&r.device().limits(),1_048_576,7,r.max_gather_channels() as usize,isize::MAX as u64).is_some());
    for (cells, channels, occupancy) in [
        (16, 4, 4),
        (16, 5, 4),
        (4096, 7, 4),
        (65_537, 9, 3),
        (262_144, 5, 4),
        (262_144, 11, 4),
        (1_048_576, 7, 4),
        (1_048_576, 9, 4),
    ] {
        let f = fixture(&r, cells, channels, occupancy);
        let refs: Vec<_> = f.source.iter().map(Vec::as_slice).collect();
        let expected =
            baseline::baseline_gather_voxel_first_f32_multi_gpu(&r, &refs, &f.segments).unwrap();
        assert_bits(&expected, &f.first);
        r.wait_idle();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..33 {
            for offset in 0..2 {
                let variant = (round + offset) % 2;
                let start = Instant::now();
                let output = if variant == 0 {
                    baseline::baseline_gather_voxel_first_f32_multi_gpu(
                        black_box(&r),
                        black_box(&refs),
                        black_box(&f.segments),
                    )
                } else {
                    gather_voxel_first_f32_multi_gpu(
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
        println!("VOXEL_STANDALONE_PACKED_RECORD {{\"cells\":{cells},\"channels\":{channels},\"occupancy\":{occupancy},\"call_old_ms\":{:?},\"call_ms\":{:?}}}",samples[0],samples[1]);
    }
}

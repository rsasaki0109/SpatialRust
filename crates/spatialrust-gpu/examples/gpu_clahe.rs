//! Compare completed grayscale equalization and explicit image placement.
mod clahe_common;

use clahe_common::{Fixture, MODES};
use spatialrust_gpu::{clahe_gpu_profiled, GpuStageProfiler, WgpuPowerPreference, WgpuRuntime};

fn main() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    println!(
        "adapter={:?}, CPU CLAHE pool={} workers",
        runtime.adapter_info(),
        rayon::current_num_threads()
    );
    println!("2 warmups + 10 rotating rounds; GPU resident modes wait for completion. CPU uses production clahe(clip=2.0,grid=8x8).");
    println!("Verification/recycling/drop outside timing; GPU resident modes add an explicit untimed full RGBA8 image readback for verification.");
    for (width, height) in [(640, 480), (3840, 2160)] {
        for hot in [false, true] {
            let mut fixture = Fixture::new(&runtime, width, height, hot);
            for _ in 0..2 {
                for mode in MODES {
                    fixture.sample(&runtime, mode);
                }
            }
            let scratch_before = runtime.scratch_buffer_pool_stats();
            let buffers_before = runtime.buffer_pool().stats();
            let mut samples = vec![Vec::new(); MODES.len()];
            let mut transfers = vec![(0, 0); MODES.len()];
            for round in 0..10 {
                for offset in 0..MODES.len() {
                    let i = (round + offset) % MODES.len();
                    let sample = fixture.sample(&runtime, MODES[i]);
                    samples[i].push(sample.elapsed.as_secs_f64() * 1000.0);
                    transfers[i] = (sample.host_to_device_bytes, sample.device_to_host_bytes);
                }
            }
            println!("{width}x{height}, pattern={}", if hot { "hot" } else { "mixed" });
            for (i, mode) in MODES.iter().enumerate() {
                samples[i].sort_by(f64::total_cmp);
                let median = (samples[i][4] + samples[i][5]) / 2.0;
                println!("{mode:?}: median={median:.6}ms, observed=[{:.6},{:.6}]ms, timed_payload_h2d={}B,d2h={}B", samples[i][0], samples[i][9], transfers[i].0, transfers[i].1);
            }
            println!(
                "warmed CLAHE storage allocations: scratch={}, bins={}, retained_scratch={}B",
                runtime.scratch_buffer_pool_stats().allocation_count
                    - scratch_before.allocation_count,
                runtime.buffer_pool().stats().allocation_count - buffers_before.allocation_count,
                runtime.scratch_buffer_pool_stats().cached_bytes
            );
            println!("texture_pool={:?}", runtime.image_texture_pool_stats());
            if runtime.timestamp_queries_enabled() {
                let mut profiler = GpuStageProfiler::new(&runtime, 3).unwrap();
                let output =
                    clahe_gpu_profiled(&runtime, &fixture.uploaded, 2.0, 8, 8, &mut profiler)
                        .unwrap();
                let report = profiler.collect(&runtime).unwrap();
                println!(
                    "pass_times_us={:?}, explicit_query_readback={}B",
                    report
                        .stages
                        .iter()
                        .map(|s| (s.stage, s.elapsed_ns / 1000.0))
                        .collect::<Vec<_>>(),
                    report.device_to_host_bytes
                );
                output.recycle(&runtime);
            }
            fixture.recycle(&runtime);
        }
    }
}

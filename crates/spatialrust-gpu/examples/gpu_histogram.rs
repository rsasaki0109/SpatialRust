//! Compare completed histogram work and explicit payload transfer placement.
mod histogram_common;

use histogram_common::{Fixture, MODES};
use spatialrust_gpu::{
    histogram_u8_gpu_profiled, GpuStageProfiler, WgpuPowerPreference, WgpuRuntime,
};

fn main() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    println!(
        "adapter={:?}, CPU histogram pool={} workers",
        runtime.adapter_info(),
        rayon::current_num_threads()
    );
    println!("2 warmups + 10 rotating rounds; CPU is production histogram_u8; GPU resident timing waits for completion.");
    println!("Verification/recycling/drop outside timing; Resident adds an explicit untimed 1024-byte bin readback for verification.");
    for (width, height) in [(640, 480), (1920, 1080), (3840, 2160)] {
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
                println!("{mode:?}: median={median:.3}ms, observed=[{:.3},{:.3}]ms, timed_payload_h2d={}B,d2h={}B", samples[i][0], samples[i][9], transfers[i].0, transfers[i].1);
            }
            println!(
                "warmed histogram storage allocations: scratch={}, output={}, retained_scratch={}B",
                runtime.scratch_buffer_pool_stats().allocation_count
                    - scratch_before.allocation_count,
                runtime.buffer_pool().stats().allocation_count - buffers_before.allocation_count,
                runtime.scratch_buffer_pool_stats().cached_bytes
            );
            if runtime.timestamp_queries_enabled() {
                let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
                let output =
                    histogram_u8_gpu_profiled(&runtime, &fixture.uploaded, 0, &mut profiler)
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

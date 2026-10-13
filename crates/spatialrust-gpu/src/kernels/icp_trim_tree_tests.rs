//! Raw rank boundaries for the optional fixed 256-bin sum tree.
use super::*;

#[test]
fn every_bin_and_rank_endpoint_match_frozen_wide_selection() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let current = TrimPipelines::new(runtime.device());
    let previous = frozen_wide_cache(&runtime);
    let mut words = 0;
    for bin in 0..256usize {
        for count in [1, u32::MAX] {
            let mut histogram = vec![0; 256];
            histogram[bin] = count;
            for step in 0..8 {
                for rank in [0, count - 1] {
                    let initial = [count, 1, rank, 0x12000000, 0x34000000, 0, 0, 0];
                    let params = TrimParams { step, ..TrimParams::new(1, 1, 0.5, 1.0, 1) };
                    let actual = digit_snapshot(&runtime, &current, params, &histogram, initial);
                    let old = digit_snapshot(&runtime, &previous, params, &histogram, initial);
                    assert_eq!(actual, old, "bin={bin} count={count} step={step} rank={rank}");
                    let mut expected = initial;
                    if step == 0 {
                        expected[1] = (f64::from(count) * 0.5).floor() as u32;
                        expected[2] = expected[1].saturating_sub(1);
                    }
                    if expected[1] != 0 {
                        expected[if step < 4 { 3 } else { 4 }] |=
                            (bin as u32) << (24 - (step % 4) * 8);
                    }
                    assert_eq!(actual, expected);
                    words += 8;
                }
            }
        }
    }
    eprintln!("GPU_ICP_TREE endpoint_words={words}");
}

#[path = "icp_trim_tree_benchmark.rs"]
mod benchmark;

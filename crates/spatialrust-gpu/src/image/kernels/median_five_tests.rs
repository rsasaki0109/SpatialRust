use super::super::five_reference as old;
use super::*;
use std::sync::OnceLock;

#[test]
fn median_five_exhausts_every_binary_window_and_preserves_three_shader() {
    let pairs = network(5);
    let previous = old::network(5);
    assert_eq!(pairs.len(), 113);
    assert_eq!(previous.len(), 191);
    assert!(pairs.iter().all(|p| p.a < p.b && p.b < 25 && p.ascending));
    assert_ne!(shader(5), old::shader(5));
    let three: Vec<_> = network(3).into_iter().map(|p| (p.a, p.b, p.ascending)).collect();
    let old_three: Vec<_> = old::network(3).into_iter().map(|p| (p.a, p.b, p.ascending)).collect();
    assert_eq!(three, old_three);
    assert_eq!(shader(3), old::shader(3));

    // Each bit is an independent window. Enumerate the lower six input bits in
    // parallel, then enumerate the upper nineteen bits across blocks. AND/OR are
    // exactly min/max for binary values; no sampled or randomized cases remain.
    let low_masks: [u64; 6] = std::array::from_fn(|i| {
        (0..64).fold(0, |mask, lane| mask | (u64::from(lane & (1 << i) != 0) << lane))
    });
    let expected: [u64; 20] = std::array::from_fn(|high_ones| {
        (0..64_u32).fold(0, |mask, lane| {
            mask | (u64::from(high_ones + lane.count_ones() as usize >= 13) << lane)
        })
    });
    let mut checked = 0_u64;
    for high in 0..1_u32 << 19 {
        let mut inputs = [u64::MAX; 32];
        inputs[..6].copy_from_slice(&low_masks);
        for (i, value) in inputs[6..25].iter_mut().enumerate() {
            *value = if high & (1 << i) == 0 { 0 } else { u64::MAX };
        }
        let mut current = inputs;
        let mut prior = inputs;
        for p in &pairs {
            let (low, high) = (current[p.a] & current[p.b], current[p.a] | current[p.b]);
            current[p.a] = low;
            current[p.b] = high;
        }
        for p in &previous {
            let (low, high) = (prior[p.a] & prior[p.b], prior[p.a] | prior[p.b]);
            (prior[p.a], prior[p.b]) = if p.ascending { (low, high) } else { (high, low) };
        }
        assert_eq!(current[12], expected[high.count_ones() as usize], "block {high}");
        assert_eq!(current[12], prior[12], "parent block {high}");
        checked += 64;
    }
    assert_eq!(checked, 33_554_432);
    println!("MEDIAN_FIVE_BINARY windows={checked} old_comparisons={} new_comparisons={} three_shader_identical=true", previous.len(), pairs.len());
}

fn frozen_gpu<const C: usize>(runtime: &WgpuRuntime) -> (usize, usize) {
    let cache = OnceLock::new();
    let mut calls = 0;
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 37), (39, 1), (65, 33), (257, 193), (1025, 257)] {
        for profile in [0, 2, 3, 4] {
            let image = fixture::<C>(w, h, profile);
            let source = GpuImage::upload_u8(runtime, image.view()).unwrap();
            for kernel in [3, 5] {
                for b in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    let mut previous = old::previous(runtime, &cache, &source, kernel, b).unwrap();
                    let mut current = median_blur_gpu(runtime, &source, kernel, b).unwrap();
                    assert_eq!(previous.receipt(), current.receipt());
                    assert_eq!(previous.metadata(), current.metadata());
                    assert_eq!(previous.channels(), current.channels());
                    assert_eq!(
                        previous.readback_u8::<C>(runtime).unwrap(),
                        current.readback_u8::<C>(runtime).unwrap()
                    );
                    assert_eq!(previous.receipt(), current.receipt());
                    previous.recycle(runtime);
                    current.recycle(runtime);
                    calls += 1;
                    bytes += w * h * C;
                }
            }
            source.recycle(runtime);
        }
    }
    (calls, bytes)
}

#[test]
#[ignore = "requires native GPU adapter"]
fn median_five_matches_complete_parent_transfers_metadata_and_both_borders() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cases = [
        frozen_gpu::<1>(&runtime),
        frozen_gpu::<2>(&runtime),
        frozen_gpu::<3>(&runtime),
        frozen_gpu::<4>(&runtime),
    ];
    println!(
        "MEDIAN_FIVE_GPU calls={} bytes={}",
        cases.iter().map(|c| c.0).sum::<usize>(),
        cases.iter().map(|c| c.1).sum::<usize>()
    );
}

#[path = "median_five_benchmark.rs"]
mod benchmark;

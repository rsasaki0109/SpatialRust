use super::super::nine_reference as old;
use super::*;
use std::sync::OnceLock;

fn frozen_selected(values: &[u8], pairs: &[old::Comparator]) -> u8 {
    let mut rows = [255; 16];
    rows[..9].copy_from_slice(values);
    for &old::Comparator { a, b, ascending } in pairs {
        let low = rows[a].min(rows[b]);
        let high = rows[a].max(rows[b]);
        (rows[a], rows[b]) = if ascending { (low, high) } else { (high, low) };
    }
    rows[4]
}

#[test]
fn nineteen_comparisons_cover_every_rank_permutation_and_keep_five_network() {
    let pairs = network(3);
    let frozen_pairs = old::network(3);
    assert_eq!(pairs.len(), 19);
    assert!(old::network(3).len() > pairs.len());
    assert!(pairs.iter().all(|p| p.a < 9 && p.b < 9));
    let five: Vec<_> = network(5).into_iter().map(|p| (p.a, p.b, p.ascending)).collect();
    let old_five: Vec<_> = old::network(5).into_iter().map(|p| (p.a, p.b, p.ascending)).collect();
    assert_eq!(five, old_five);
    assert_eq!(shader(5), old::shader(5));
    let mut values = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut cases = 0;
    loop {
        assert_eq!(selected(&values, &pairs), 4);
        assert_eq!(frozen_selected(&values, &frozen_pairs), 4);
        cases += 1;
        let Some(i) = (0..8).rev().find(|&i| values[i] < values[i + 1]) else {
            break;
        };
        let j = (i + 1..9).rev().find(|&j| values[j] > values[i]).unwrap();
        values.swap(i, j);
        values[i + 1..].reverse();
    }
    assert_eq!(cases, 362_880);
    println!(
        "MEDIAN_NINE_RANK permutations={cases} old_comparisons={} new_comparisons={}",
        old::network(3).len(),
        pairs.len()
    );
}

fn frozen_gpu<const C: usize>(r: &WgpuRuntime) -> (usize, usize) {
    let cache = OnceLock::new();
    let mut calls = 0;
    let mut bytes = 0;
    for (w, h) in [(1, 1), (1, 257), (259, 1), (17, 19), (257, 193), (1025, 257)] {
        for profile in [0, 2, 3, 4] {
            let image = fixture::<C>(w, h, profile);
            let source = GpuImage::upload_u8(r, image.view()).unwrap();
            for kernel in [3, 5] {
                for b in [GpuImageBorder::Replicate, GpuImageBorder::ConstantZero] {
                    let mut before = old::previous(r, &cache, &source, kernel, b).unwrap();
                    let mut after = median_blur_gpu(r, &source, kernel, b).unwrap();
                    assert_eq!(before.receipt(), after.receipt());
                    assert_eq!(before.metadata(), after.metadata());
                    assert_eq!(
                        before.readback_u8::<C>(r).unwrap(),
                        after.readback_u8::<C>(r).unwrap()
                    );
                    before.recycle(r);
                    after.recycle(r);
                    calls += 1;
                    bytes += w * h * C;
                }
            }
            source.recycle(r);
        }
    }
    (calls, bytes)
}

#[test]
#[ignore = "requires native GPU adapter"]
fn frozen_gpu_bytes_receipts_and_five_by_five_control_match() {
    let r = WgpuRuntime::new_headless().unwrap();
    let cases =
        [frozen_gpu::<1>(&r), frozen_gpu::<2>(&r), frozen_gpu::<3>(&r), frozen_gpu::<4>(&r)];
    println!(
        "MEDIAN_NINE_GPU calls={} bytes={}",
        cases.iter().map(|c| c.0).sum::<usize>(),
        cases.iter().map(|c| c.1).sum::<usize>()
    );
}

#[path = "median_nine_benchmark.rs"]
mod benchmark;

use super::{nearest_tests::words, partition_tests::fixture, *};
use std::{hint::black_box, time::Instant};

fn query_batch(
    tree: &impl NearestNeighborIndex,
    queries: &[[f32; 3]],
    out: &mut [Option<Neighbor>],
) {
    for (query, result) in queries.iter().zip(out) {
        let [x, y, z] = black_box(*query);
        *result = black_box(tree.nearest_one(x, y, z));
    }
}

#[test]
#[ignore = "completed scalar query comparison; explicit native invocation only"]
fn paired_previous_scalar_nearest() {
    eprintln!("2 warmups + 20 alternating paired rounds; 8192 completed queries per batch; construction, fixtures, validation and output allocation outside timers");
    eprintln!("points,kind,round,previous_ms,current_ms");
    for points in [512usize, 4096, 32768] {
        for kind in [0usize, 3, 4, 8] {
            let [x, y, z] = fixture(points, kind);
            let previous = previous::KdTree::from_slices(&x, &y, &z);
            let current = KdTree::from_slices(&x, &y, &z);
            assert_eq!(partition_tests::snapshot(&current), previous::snapshot(&previous));
            let queries: Vec<_> = (0..8192)
                .map(|q| {
                    let i = q * 37 % points;
                    let offset = if kind == 0 || q % 2 == 0 { 0.0 } else { 1.0 };
                    [x[i] + offset * 0.002, y[i] - offset * 0.001, z[i] + offset * 0.004]
                })
                .collect();
            let mut out = vec![None; queries.len()];
            query_batch(&previous, &queries, &mut out);
            let expected: Vec<_> = out.iter().copied().map(words).collect();
            for round in 0..22 {
                let mut ms = [0.0; 2];
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = Instant::now();
                    if mode == 0 {
                        query_batch(&previous, &queries, &mut out);
                    } else {
                        query_batch(&current, &queries, &mut out);
                    }
                    ms[mode] = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(out.iter().copied().map(words).collect::<Vec<_>>(), expected);
                }
                if round >= 2 {
                    eprintln!("{points},{kind},{},{:.6},{:.6}", round - 2, ms[0], ms[1]);
                }
            }
        }
    }
}

use super::{
    partition_tests::{fixture, snapshot},
    *,
};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "completed KD-tree construction comparison; explicit native invocation only"]
fn paired_previous_minimum_pivot_kdtree() {
    eprintln!("2 warmups + 20 alternating paired rounds; serial KD-tree construction; validation snapshots and fixture creation outside timers");
    eprintln!("Timers include point ordering, nodes, coordinate copies, axis selection and partitions; exclude returned tree drop.");
    eprintln!("points,kind,round,previous_ms,current_ms");
    for points in [512usize, 4096, 16384, 32768] {
        for kind in [0usize, 1, 3, 8] {
            let [x, y, z] = fixture(points, kind);
            let expected = previous::snapshot(&previous::KdTree::from_slices(&x, &y, &z));
            for _ in 0..2 {
                assert_eq!(snapshot(&KdTree::from_slices(&x, &y, &z)), expected);
                assert_eq!(
                    previous::snapshot(&previous::KdTree::from_slices(&x, &y, &z)),
                    expected
                );
            }
            for round in 0..20 {
                let mut ms = [0.0; 2];
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = Instant::now();
                    if mode == 0 {
                        let tree = previous::KdTree::from_slices(&x, &y, &z);
                        ms[0] = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(previous::snapshot(&tree), expected);
                        black_box(&tree);
                    } else {
                        let tree = KdTree::from_slices(&x, &y, &z);
                        ms[1] = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(snapshot(&tree), expected);
                        black_box(&tree);
                    }
                }
                eprintln!("{points},{kind},{round},{:.6},{:.6}", ms[0], ms[1]);
            }
        }
    }
}

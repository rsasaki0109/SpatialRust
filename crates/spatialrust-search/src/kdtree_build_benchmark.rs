use super::{
    partition_tests::{fixture, snapshot},
    *,
};
use std::time::Instant;

#[test]
#[ignore = "completed KD-tree construction benchmark; run this exact test alone in release mode"]
fn completed_parallel_kdtree_builds() {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("PARALLEL_KDTREE_BUILD_ENV available_workers={available} threshold={PARALLEL_KDTREE_BUILD_MIN_POINTS}");
    for n in [4097, 32768, 65537, 262145] {
        for kind in [0, 3, 8] {
            let [x, y, z] = fixture(n, kind);
            let expected = snapshot(&KdTree::from_slices(&x, &y, &z));
            for _ in 0..3 {
                assert_eq!(snapshot(&KdTree::from_slices(&x, &y, &z)), expected);
                assert_eq!(snapshot(&KdTree::from_slices_parallel(&x, &y, &z)), expected);
            }
            let (mut serial_ms, mut parallel_ms) = (vec![], vec![]);
            for round in 0..20 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    let start = Instant::now();
                    let tree = if mode == 0 {
                        KdTree::from_slices(&x, &y, &z)
                    } else {
                        KdTree::from_slices_parallel(&x, &y, &z)
                    };
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    if mode == 0 {
                        serial_ms.push(ms);
                    } else {
                        parallel_ms.push(ms);
                    }
                    assert_eq!(snapshot(&tree), expected);
                    std::hint::black_box(&tree);
                }
            }
            println!("PARALLEL_KDTREE_BUILD_BENCH n={n} kind={kind} serial_ms={serial_ms:?} parallel_ms={parallel_ms:?}");
        }
    }
}

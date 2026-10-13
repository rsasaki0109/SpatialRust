use super::*;
use crate::kdtree::{
    partition_tests::{fixture, neighbor_words, snapshot},
    previous,
};

#[test]
fn node_ranges_and_worker_policy_are_exact_and_bounded() {
    let mut reference = vec![0usize; 131100];
    for n in 1..reference.len() {
        reference[n] = if n <= LEAF_SIZE { 1 } else { 1 + reference[n / 2] + reference[n - n / 2] };
        assert_eq!(node_count(n), reference[n], "n={n}");
    }
    for n in [0, 1, 16384, 32767, 32768, 32769, 65537, 131072, usize::MAX] {
        for available in [0, 1, 2, 6, usize::MAX] {
            let workers = worker_count(n, available);
            assert!(workers >= 1 && workers <= available.max(1));
            if n < 32768 {
                assert_eq!(workers, 1);
            } else {
                assert!(workers <= n / 16384);
            }
        }
    }
    // The original-index limit also keeps every computed node ID below u32.
    assert!(node_count(u32::MAX as usize) < u32::MAX as usize);
}

fn check_queries(actual: &KdTree, serial: &KdTree) -> usize {
    let mut checked = 0;
    for q in [[0.0; 3], [1.0; 3], [0.25, 0.5, 0.0], [f32::NAN, 0.0, 0.0], [f32::INFINITY, 0.0, 0.0]]
    {
        for k in [0, 1, 3, 17, 65] {
            assert_eq!(
                neighbor_words(actual.nearest_k(q[0], q[1], q[2], k)),
                neighbor_words(serial.nearest_k(q[0], q[1], q[2], k))
            );
            checked += 1;
        }
        for r in [0.0, 0.1, 2.0, f32::INFINITY, f32::NAN] {
            assert_eq!(
                neighbor_words(actual.radius_search(q[0], q[1], q[2], r)),
                neighbor_words(serial.radius_search(q[0], q[1], q[2], r))
            );
            checked += 1;
        }
        for bound in [-1.0, 0.0, 1.0, f32::INFINITY, f32::NAN] {
            let bits = |v: Option<Neighbor>| v.map(|n| (n.index, n.distance_squared.to_bits()));
            assert_eq!(
                bits(actual.nearest_one_within(q[0], q[1], q[2], bound)),
                bits(serial.nearest_one_within(q[0], q[1], q[2], bound))
            );
            checked += 1;
        }
    }
    checked
}

#[test]
fn parallel_trees_match_frozen_layout_words_coordinates_and_queries() {
    let mut layouts = 0;
    let mut queries = 0;
    for n in [0, 1, 15, 16, 17, 31, 32, 33, 257, 1025, 32767, 32768, 32769, 65537, 131073] {
        for kind in 0..9 {
            let [x, y, z] = fixture(n, kind);
            let serial = KdTree::from_slices(&x, &y, &z);
            let expected = snapshot(&serial);
            if n <= 1025 {
                assert_eq!(
                    expected,
                    previous::snapshot(&previous::KdTree::from_slices(&x, &y, &z))
                );
            }
            for workers in [1, 2, 6] {
                let actual = from_slices_with_workers(&x, &y, &z, workers);
                assert_eq!(snapshot(&actual), expected, "n={n} kind={kind} workers={workers}");
                for (a, b) in
                    [actual.x.as_slice(), &actual.y, &actual.z].into_iter().zip([&x, &y, &z])
                {
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                queries += check_queries(&actual, &serial);
                layouts += 1;
            }
        }
    }
    eprintln!("PARALLEL_KDTREE_BUILD exact_layouts={layouts} exact_query_results={queries}");
}

#[test]
fn public_cloud_builder_preserves_inputs_and_concurrent_calls() {
    let [x, y, z] = fixture(65537, 8);
    let before = [x.clone(), y.clone(), z.clone()];
    let expected = snapshot(&KdTree::from_slices(&x, &y, &z));
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        assert_eq!(snapshot(&KdTree::from_slices_parallel(&x, &y, &z)), expected);
                    }
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
    assert_eq!([x, y, z], before);
    let mut b = spatialrust_core::PointCloudBuilder::xyz();
    for p in [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        b.push_point(p).unwrap();
    }
    let cloud = b.build().unwrap();
    assert_eq!(
        snapshot(&KdTree::from_point_cloud_parallel(&cloud).unwrap()),
        snapshot(&KdTree::from_point_cloud(&cloud).unwrap())
    );
    assert!(std::panic::catch_unwind(|| KdTree::from_slices_parallel(&[0.0], &[], &[0.0])).is_err());
    // Independent geometry expectation, including inclusive bound/self ties.
    let t = KdTree::from_slices_parallel(&[0.0, 1.0, 1.0], &[0.0; 3], &[0.0; 3]);
    assert_eq!(t.nearest_one_within(0.5, 0.0, 0.0, 0.25).unwrap().index, 0);
    assert_eq!(t.radius_search(0.0, 0.0, 0.0, 1.0).len(), 3);
}

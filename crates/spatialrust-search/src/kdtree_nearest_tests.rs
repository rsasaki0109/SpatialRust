use super::{partition_tests::fixture, *};

pub(super) fn words(neighbor: Option<Neighbor>) -> Option<(usize, u32)> {
    neighbor.map(|n| (n.index, n.distance_squared.to_bits()))
}

#[test]
fn scalar_nearest_matches_frozen_and_buffered_queries_word_for_word() {
    let mut comparisons = 0;
    for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257, 1025] {
        for kind in 0..9 {
            let [x, y, z] = fixture(len, kind);
            let current = KdTree::from_slices(&x, &y, &z);
            let frozen = previous::KdTree::from_slices(&x, &y, &z);
            let mut queries = vec![
                [0.0; 3],
                [-0.0; 3],
                [1.0; 3],
                [f32::MAX; 3],
                [f32::MIN; 3],
                [f32::from_bits(1); 3],
            ];
            for axis in 0..3 {
                for value in [f32::from_bits(0x7fc00053), f32::INFINITY, f32::NEG_INFINITY] {
                    let mut query = [0.25, 0.5, 0.75];
                    query[axis] = value;
                    queries.push(query);
                }
            }
            for i in 0..64 {
                queries.push([
                    ((i * 43) % 139) as f32 / 9.0 - 4.0,
                    ((i * 61) % 131) as f32 / 13.0 - 3.0,
                    ((i * 29) % 127) as f32 / 7.0 - 5.0,
                ]);
            }
            let mut scratch = Vec::new();
            for [qx, qy, qz] in queries {
                let expected = words(frozen.nearest_one(qx, qy, qz));
                assert_eq!(
                    words(current.nearest_one(qx, qy, qz)),
                    expected,
                    "len={len}, kind={kind}, query={qx:?},{qy:?},{qz:?}"
                );
                current.nearest_k_into(qx, qy, qz, 1, &mut scratch);
                assert_eq!(words(scratch.first().copied()), expected);
                comparisons += 2;
            }
        }
    }
    eprintln!("Verified {comparisons} frozen/buffered scalar neighbor comparisons.");
}

#[test]
fn exhaustive_small_nonfinite_patterns_preserve_scalar_neighbor_words() {
    let symbols = [-0.0, 1.0, f32::NEG_INFINITY, f32::INFINITY, f32::from_bits(0x7fc00037)];
    let queries =
        [0.0, -0.0, 0.5, f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc00053)];
    let mut comparisons = 0;
    for len in 0usize..=6 {
        for mut code in 0..5_usize.pow(len as u32) {
            let values: Vec<f32> = (0..len)
                .map(|_| {
                    let value = symbols[code % 5];
                    code /= 5;
                    value
                })
                .collect();
            let zeros = vec![0.0; len];
            for axis in 0..3 {
                let mut xyz = [&zeros[..]; 3];
                xyz[axis] = &values;
                let tree = KdTree::from_slices(xyz[0], xyz[1], xyz[2]);
                for value in queries {
                    let mut q = [0.0; 3];
                    q[axis] = value;
                    assert_eq!(
                        words(tree.nearest_one(q[0], q[1], q[2])),
                        words(tree.nearest_k(q[0], q[1], q[2], 1).first().copied())
                    );
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!("Verified {comparisons} exhaustive nonfinite scalar queries.");
}

#[test]
fn scalar_nearest_retains_independent_tie_nan_and_overflow_behavior() {
    let tree = KdTree::from_slices(&[-1.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
    assert_eq!(words(tree.nearest_one(0.0, 0.0, 0.0)), Some((0, 1.0_f32.to_bits())));
    assert_eq!(
        words(tree.nearest_one(f32::INFINITY, 0.0, 0.0)),
        Some((0, f32::INFINITY.to_bits()))
    );
    let nan_query = tree.nearest_one(f32::NAN, 0.0, 0.0).unwrap();
    assert_eq!(nan_query.index, 2);
    assert!(nan_query.distance_squared.is_nan());
    let tree = KdTree::from_slices(&[0.0, f32::NAN, 2.0], &[0.0; 3], &[0.0; 3]);
    assert_eq!(words(tree.nearest_one(0.0, 0.0, 0.0)), Some((2, 4.0_f32.to_bits())));
    let tree = KdTree::from_slices(&[0.0, 2.0, f32::NAN], &[0.0; 3], &[0.0; 3]);
    let final_nan = tree.nearest_one(0.0, 0.0, 0.0).unwrap();
    assert_eq!(final_nan.index, 2);
    assert!(final_nan.distance_squared.is_nan());
    let tree = KdTree::from_slices(&[f32::MAX, f32::MIN], &[0.0; 2], &[0.0; 2]);
    assert_eq!(words(tree.nearest_one(0.0, 0.0, 0.0)), Some((0, f32::INFINITY.to_bits())));
    assert_eq!(KdTree::from_slices(&[], &[], &[]).nearest_one(f32::NAN, 0.0, 0.0), None);
}

#[test]
fn concurrent_scalar_queries_share_tree_without_changing_results() {
    let [x, y, z] = fixture(4097, 8);
    let tree = KdTree::from_slices(&x, &y, &z);
    let expected: Vec<_> = (0..2048)
        .map(|i| {
            words(tree.nearest_k(x[i] + 0.002, y[i] - 0.001, z[i] + 0.004, 1).first().copied())
        })
        .collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for i in 0..2048 {
                        assert_eq!(
                            words(tree.nearest_one(x[i] + 0.002, y[i] - 0.001, z[i] + 0.004)),
                            expected[i]
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

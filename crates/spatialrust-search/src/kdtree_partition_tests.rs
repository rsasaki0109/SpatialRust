use super::*;

pub(super) fn fixture(len: usize, kind: usize) -> [Vec<f32>; 3] {
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(len));
    for i in 0..len {
        let p = match kind {
            0 => [1.0, 1.0, 1.0],
            1 => [if i == 0 { 1e8 } else { 1.0 }, 0.0, 0.0],
            2 => [(i % 4) as f32, 0.0, 0.0],
            3 => {
                let x = (i % 128) as f32 * 0.05;
                let y = (i / 128) as f32 * 0.05;
                [x, y, 0.015 * (x * x + y * y) + (i * 37 % 101) as f32 * 0.0008]
            }
            4 => [i as f32 * 0.01, 0.0, 0.0],
            5 => [f32::from_bits(0x7fc00000 + (i % 31) as u32), 0.0, 0.0],
            6 => [if i % 2 == 0 { -0.0 } else { 0.0 }, 0.0, 0.0],
            7 => [[f32::NEG_INFINITY, -1.0, f32::INFINITY, f32::NAN][i % 4], 0.0, 0.0],
            8 => [2654435761_u32, 2246822519, 3266489917]
                .map(|factor| ((i as u32).wrapping_mul(factor) >> 8) as f32 / 16777216.0),
            _ => unreachable!(),
        };
        for (column, value) in xyz.iter_mut().zip(p) {
            column.push(value);
        }
    }
    xyz
}

pub(super) fn snapshot(tree: &KdTree) -> (u32, Vec<u32>, Vec<[u32; 6]>) {
    (
        tree.root,
        tree.points_order.clone(),
        tree.nodes
            .iter()
            .map(|n| [n.split.to_bits(), n.left, n.right, n.start, n.end, u32::from(n.axis)])
            .collect(),
    )
}

pub(super) fn neighbor_words(neighbors: Vec<Neighbor>) -> Vec<(usize, u32)> {
    neighbors.into_iter().map(|n| (n.index, n.distance_squared.to_bits())).collect()
}

#[test]
fn frozen_tree_layout_and_query_words_match_for_ties_and_nonfinite_coordinates() {
    let mut trees = 0;
    let mut query_results = 0;
    for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257, 1025] {
        for kind in 0..9 {
            let [x, y, z] = fixture(len, kind);
            let actual = KdTree::from_slices(&x, &y, &z);
            let expected = previous::KdTree::from_slices(&x, &y, &z);
            assert_eq!(snapshot(&actual), previous::snapshot(&expected), "len={len}, kind={kind}");
            trees += 1;
            for q in [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.25, 0.5, 0.0],
                [f32::NAN, 0.0, 0.0],
                [f32::INFINITY, 0.0, 0.0],
            ] {
                for k in [0, 1, 3, 17, len + 1] {
                    assert_eq!(
                        neighbor_words(actual.nearest_k(q[0], q[1], q[2], k)),
                        neighbor_words(expected.nearest_k(q[0], q[1], q[2], k))
                    );
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    actual.nearest_k_unsorted_into(q[0], q[1], q[2], k, &mut a);
                    expected.nearest_k_unsorted_into(q[0], q[1], q[2], k, &mut b);
                    assert_eq!(neighbor_words(a), neighbor_words(b));
                    query_results += 2;
                }
                for radius in [0.0, 0.1, 2.0, f32::INFINITY, f32::NAN] {
                    assert_eq!(
                        neighbor_words(actual.radius_search(q[0], q[1], q[2], radius)),
                        neighbor_words(expected.radius_search(q[0], q[1], q[2], radius))
                    );
                    query_results += 1;
                }
                for bound in [-1.0, 0.0, 1.0, f32::INFINITY, f32::NAN] {
                    let a = actual
                        .nearest_one_within(q[0], q[1], q[2], bound)
                        .map(|n| (n.index, n.distance_squared.to_bits()));
                    let b = expected
                        .nearest_one_within(q[0], q[1], q[2], bound)
                        .map(|n| (n.index, n.distance_squared.to_bits()));
                    assert_eq!(a, b);
                    query_results += 1;
                }
            }
        }
    }
    eprintln!("Verified {trees} complete frozen tree layouts and {query_results} exact ordered query results.");
}

#[test]
fn every_small_axis_pattern_preserves_original_selection_permutation() {
    // Exhaust every ternary pattern, including mixed NaNs and signed zeros,
    // with all nth positions and nontrivial input permutations/range guards.
    let mut cases = 0;
    for symbols in [
        [-1.0, 1.0, f32::from_bits(0x7fc00037)],
        [-0.0, 0.0, f32::INFINITY],
        [f32::NEG_INFINITY, f32::INFINITY, f32::from_bits(0x7fc00053)],
    ] {
        for len in 1usize..=7 {
            for code in 0..3_usize.pow(len as u32) {
                let mut code = code;
                let values: Vec<f32> = (0..len)
                    .map(|_| {
                        let v = symbols[code % 3];
                        code /= 3;
                        v
                    })
                    .collect();
                let mut padded = vec![-0.0];
                padded.extend(values);
                padded.push(f32::INFINITY);
                let mut order: Vec<u32> = (0..padded.len() as u32).collect();
                order[1..len + 1].reverse();
                for axis in 0..3 {
                    for nth in 1..len + 1 {
                        let xyz = [&padded[..], &padded[..], &padded[..]];
                        let mut actual = order.clone();
                        let mut expected = order.clone();
                        let minimum =
                            padded[1..len + 1].iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        select_nth_by_axis(
                            xyz[0],
                            xyz[1],
                            xyz[2],
                            &mut actual,
                            1,
                            len + 1,
                            (axis, minimum),
                            nth,
                        );
                        previous::select_for_test(
                            xyz[0],
                            xyz[1],
                            xyz[2],
                            &mut expected,
                            1..len + 1,
                            axis,
                            nth,
                        );
                        assert_eq!(actual, expected, "len={len}, axis={axis}, nth={nth}");
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("Verified {cases} exhaustive parent selection permutations.");
}

#[test]
fn large_coincident_cloud_retains_every_index_and_independent_search_distances() {
    let [x, y, z] = fixture(65537, 0);
    let tree = KdTree::from_slices(&x, &y, &z);
    let mut order = tree.points_order.clone();
    order.sort_unstable();
    assert_eq!(order, (0..65537).collect::<Vec<u32>>());
    let neighbor = tree.nearest_one(0.0, 0.0, 0.0).unwrap();
    assert_eq!(neighbor.distance_squared.to_bits(), 3.0_f32.to_bits());
    assert_eq!(tree.radius_search(1.0, 1.0, 1.0, 0.0).len(), 65537);
    assert_eq!(tree.nearest_k(1.0, 1.0, 1.0, 3).len(), 3);
    assert_eq!(tree.nearest_one_within(0.0, 0.0, 0.0, 2.99), None);
}

#[test]
fn concurrent_builds_preserve_parent_tree_snapshots() {
    let [x, y, z] = fixture(1025, 2);
    let expected = previous::snapshot(&previous::KdTree::from_slices(&x, &y, &z));
    std::thread::scope(|scope| {
        let builds: Vec<_> = (0..4)
            .map(|_| {
                let (x, y, z) = (&x, &y, &z);
                scope.spawn(move || snapshot(&KdTree::from_slices(x, y, z)))
            })
            .collect();
        for build in builds {
            assert_eq!(build.join().unwrap(), expected);
        }
    });
}

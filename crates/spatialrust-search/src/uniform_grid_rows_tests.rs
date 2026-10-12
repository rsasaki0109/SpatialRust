use super::*;

#[allow(dead_code)]
pub(super) mod previous {
    include!("../tests/fixtures/uniform_grid_previous.rs");

    pub(super) fn neighbors(
        index: usize,
        xyz: &[Vec<f32>; 3],
        origin: [f32; 3],
        dims: [u32; 3],
        radius: f32,
        sorted: &[u32],
        starts: &[u32],
    ) -> Vec<usize> {
        grid_radius_neighbors(
            index,
            &xyz[0],
            &xyz[1],
            &xyz[2],
            origin,
            dims,
            radius,
            radius * radius,
            sorted,
            starts,
        )
        .collect()
    }
}

pub(super) fn fixture(n: usize, kind: usize) -> [Vec<f32>; 3] {
    let mut xyz = std::array::from_fn(|_| Vec::with_capacity(n));
    let side = (n as f64).cbrt().ceil() as usize;
    for i in 0..n {
        let p = match kind {
            0 => [i % side, (i / side) % side, i / (side * side)].map(|v| v as f32 * 0.75),
            1 => {
                let width = (n as f64).sqrt().ceil() as usize;
                let x = (i % width) as f32 * 0.75;
                let y = (i / width) as f32 * 0.75;
                [x, y, (x + y) * 0.125]
            }
            2 => {
                let group = i / 8;
                let width = ((n.div_ceil(8)) as f64).cbrt().ceil() as usize;
                [
                    (group % width) as f32 * 3.0 + (i & 1) as f32 * 0.25,
                    ((group / width) % width) as f32 * 3.0 + ((i >> 1) & 1) as f32 * 0.25,
                    (group / (width * width)) as f32 * 3.0 + ((i >> 2) & 1) as f32 * 0.25,
                ]
            }
            3 => [i as f32 * 0.75, 0.0, 0.0],
            _ => [0.0; 3],
        };
        for axis in 0..3 {
            xyz[axis].push(p[axis] - 2.0);
        }
    }
    xyz
}

#[test]
fn row_neighbors_preserve_frozen_and_independent_order_at_grid_edges() {
    let mut queries = 0;
    let mut neighbor_words = 0;
    for dx in [1_u32, 2, 4] {
        for dy in [1_u32, 2, 4] {
            for dz in [1_u32, 2, 4] {
                let dims = [dx, dy, dz];
                for radius in [0.25_f32, 1.0, 2.0] {
                    for sparse in [false, true] {
                        let origin = [-3.0, -2.0, -1.0];
                        let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
                        for cell in (0..dx * dy * dz).rev() {
                            if sparse && cell % 3 != 0 {
                                continue;
                            }
                            let c = [cell % dx, (cell / dx) % dy, cell / (dx * dy)];
                            // Exact boundaries, duplicates, and interior samples.
                            for offset in [0.0, 0.0, 0.5, 0.875] {
                                for axis in 0..3 {
                                    xyz[axis]
                                        .push(origin[axis] + (c[axis] as f32 + offset) * radius);
                                }
                            }
                        }
                        // Coordinates outside the prescribed bounds are clamped.
                        for upper in [false, true] {
                            for axis in 0..3 {
                                let c = if upper { dims[axis] as f32 + 0.25 } else { -0.25 };
                                xyz[axis].push(origin[axis] + c * radius);
                            }
                        }
                        let (sorted, starts) =
                            build_grid(&xyz[0], &xyz[1], &xyz[2], origin, dims, radius);
                        for index in 0..xyz[0].len() {
                            let mut iterator = grid_radius_neighbors(
                                index,
                                &xyz[0],
                                &xyz[1],
                                &xyz[2],
                                origin,
                                dims,
                                radius,
                                radius * radius,
                                &sorted,
                                &starts,
                            );
                            let current: Vec<_> = iterator.by_ref().collect();
                            assert_eq!(iterator.next(), None);
                            assert_eq!(iterator.next(), None);
                            assert_eq!(
                                current,
                                previous::neighbors(
                                    index, &xyz, origin, dims, radius, &sorted, &starts,
                                )
                            );
                            // Brute-force membership and a grid key order without CSR traversal.
                            let mut expected: Vec<_> = (0..xyz[0].len())
                                .filter(|&j| {
                                    let d = std::array::from_fn::<_, 3, _>(|a| {
                                        xyz[a][j] - xyz[a][index]
                                    });
                                    j != index
                                        && d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                                            <= radius * radius
                                })
                                .collect();
                            expected.sort_by_key(|&j| {
                                let c = std::array::from_fn::<_, 3, _>(|a| {
                                    (((xyz[a][j] - origin[a]) / radius).floor() as i32)
                                        .clamp(0, dims[a] as i32 - 1)
                                });
                                (c[2], c[1], c[0], j)
                            });
                            assert_eq!(
                                current, expected,
                                "dims={dims:?}, radius={radius}, query={index}"
                            );
                            queries += 1;
                            neighbor_words += current.len();
                        }
                    }
                }
            }
        }
    }
    eprintln!("Verified {queries} ordered queries and {neighbor_words} neighbor indices against frozen and independent oracles.");
}

fn brute_roots(xyz: &[Vec<f32>; 3], radius: f32) -> Vec<u32> {
    let mut labels: Vec<u32> = (0..xyz[0].len() as u32).collect();
    for i in 0..xyz[0].len() {
        for j in 0..i {
            let d = std::array::from_fn::<_, 3, _>(|a| xyz[a][i] - xyz[a][j]);
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= radius * radius {
                let old = labels[i].max(labels[j]);
                let new = labels[i].min(labels[j]);
                for label in &mut labels {
                    if *label == old {
                        *label = new;
                    }
                }
            }
        }
    }
    labels
}

#[test]
fn component_roots_match_brute_force_and_frozen_on_small_clouds() {
    for n in [0_usize, 1, 2, 3, 7, 8, 9, 31, 127, 257] {
        for kind in 0..5 {
            let xyz = fixture(n, kind);
            for radius in [0.25_f32, 0.75, 1.0, 2.0] {
                let actual = euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], radius).unwrap();
                assert_eq!(
                    actual,
                    brute_roots(&xyz, radius),
                    "n={n}, kind={kind}, radius={radius}"
                );
                assert_eq!(
                    actual,
                    previous::euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], radius).unwrap()
                );
            }
        }
    }
}

#[test]
fn nonfinite_and_invalid_inputs_retain_existing_behavior() {
    for values in [
        vec![0.0, -0.0, f32::NAN, 0.5],
        vec![0.0, f32::INFINITY, 0.5],
        vec![0.0, f32::NEG_INFINITY, 0.5],
        vec![f32::MIN, f32::MAX],
    ] {
        let xyz = [values.clone(), vec![0.0; values.len()], vec![0.0; values.len()]];
        for radius in [1.0_f32, f32::INFINITY] {
            let origin = [0.0; 3];
            let dims = [4, 1, 1];
            let (sorted, starts) = build_grid(&xyz[0], &xyz[1], &xyz[2], origin, dims, radius);
            for index in 0..xyz[0].len() {
                let actual: Vec<_> = grid_radius_neighbors(
                    index,
                    &xyz[0],
                    &xyz[1],
                    &xyz[2],
                    origin,
                    dims,
                    radius,
                    radius * radius,
                    &sorted,
                    &starts,
                )
                .collect();
                assert_eq!(
                    actual,
                    previous::neighbors(index, &xyz, origin, dims, radius, &sorted, &starts,)
                );
            }
        }
    }
    for radius in [-1.0, 0.0, f32::NAN] {
        assert!(euclidean_cluster_roots(&[0.0], &[0.0], &[0.0], radius).is_err());
    }
    assert!(euclidean_cluster_roots(&[0.0], &[], &[0.0], 1.0).is_err());
}

#[test]
fn threshold_and_concurrent_calls_preserve_minimum_component_roots() {
    for n in [4095_usize, 4096, 4097, 10003] {
        let xyz = fixture(n, 2);
        let expected: Vec<_> = (0..n).map(|i| (i / 8 * 8) as u32).collect();
        assert_eq!(euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap(), expected);
        assert_eq!(
            previous::euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap(),
            expected
        );
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        assert_eq!(
                            euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap(),
                            expected
                        );
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
}

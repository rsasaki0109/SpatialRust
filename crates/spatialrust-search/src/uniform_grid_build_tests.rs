use super::{build_grid, row_tests::previous};

pub(super) fn fixture(n: usize, dims: [u32; 3], radius: f32) -> ([Vec<f32>; 3], Vec<usize>) {
    let cells = (dims[0] * dims[1] * dims[2]) as usize;
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        // Interleave cells, repeated points, and exact lower cell boundaries.
        let cell = i.wrapping_mul(104729).wrapping_add(i / 3) % cells;
        keys.push(cell);
        let c = [
            cell % dims[0] as usize,
            cell / dims[0] as usize % dims[1] as usize,
            cell / (dims[0] * dims[1]) as usize,
        ];
        for axis in 0..3 {
            let offset = if i % 3 == 0 { 0.0 } else { 0.5 };
            xyz[axis].push(-4.0 + (c[axis] as f32 + offset) * radius);
        }
    }
    (xyz, keys)
}

#[test]
fn reverse_scatter_matches_frozen_and_independent_stable_csr() {
    let mut words = 0;
    let mut cases = 0;
    for dims in [[1, 1, 1], [2, 3, 4], [7, 1, 1], [1, 5, 3], [17, 13, 11], [127, 31, 7]] {
        let cells = (dims[0] * dims[1] * dims[2]) as usize;
        for n in [0_usize, 1, 2, 255, 256, 257, 4095, 4096, 4097, 65537] {
            for radius in [0.25_f32, 1.0, 2.0] {
                let (xyz, keys) = fixture(n, dims, radius);
                let actual = build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, radius);
                assert_eq!(
                    actual,
                    previous::build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], dims, radius)
                );
                // Sort independent integer labels, without float mapping or a histogram.
                let mut order: Vec<_> = (0..n).collect();
                order.sort_by_key(|&i| (keys[i], i));
                let expected_sorted: Vec<_> = order.iter().map(|&i| i as u32).collect();
                let expected_starts: Vec<_> = (0..=cells)
                    .map(|cell| order.partition_point(|&i| keys[i] < cell) as u32)
                    .collect();
                assert_eq!(actual.0, expected_sorted, "n={n}, dims={dims:?}, radius={radius}");
                assert_eq!(actual.1, expected_starts);
                assert_eq!(actual.1[cells], n as u32);
                words += actual.0.len() + actual.1.len();
                cases += 1;
            }
        }
    }
    eprintln!("GRID_BUILD verified {words} CSR words across {cases} frozen/independent cases.");
}

#[test]
fn reverse_scatter_retains_clamping_and_nonfinite_mapping() {
    let values = [
        f32::NEG_INFINITY,
        f32::MIN,
        -10.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        0.25,
        0.3,
        0.5,
        1.0,
        2.0,
        1000.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    for dims in [[1, 1, 1], [3, 4, 5], [127, 1, 7]] {
        let xyz: [Vec<f32>; 3] = std::array::from_fn(|axis| {
            (0..257).map(|i| values[(i * (axis + 3) + axis) % values.len()]).collect()
        });
        for radius in [f32::from_bits(1), 0.3, 0.75, 1.0, f32::MAX, f32::INFINITY] {
            for origin in [[0.0; 3], [-10.0, 2.0, -3.0]] {
                let actual = build_grid(&xyz[0], &xyz[1], &xyz[2], origin, dims, radius);
                assert_eq!(
                    actual,
                    previous::build_grid(&xyz[0], &xyz[1], &xyz[2], origin, dims, radius)
                );
            }
        }
    }
    let result =
        build_grid(&[-3.0, 0.0, 10.0, 0.5, -1.0], &[0.0; 5], &[0.0; 5], [0.0; 3], [3, 1, 1], 1.0);
    assert_eq!(result, (vec![0, 1, 3, 4, 2], vec![0, 4, 4, 5]));
}

#[test]
fn concurrent_grid_builds_keep_owned_results_and_stable_cell_order() {
    let (xyz, _) = fixture(10003, [17, 13, 11], 0.5);
    let expected = previous::build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], [17, 13, 11], 0.5);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let first = build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], [17, 13, 11], 0.5);
                    let second =
                        build_grid(&xyz[0], &xyz[1], &xyz[2], [-4.0; 3], [17, 13, 11], 0.5);
                    assert_eq!(first, expected);
                    assert_eq!(second, expected);
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

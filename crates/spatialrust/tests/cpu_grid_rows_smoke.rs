use spatialrust::search::euclidean_cluster_roots;

#[test]
fn facade_dense_grid_keeps_minimum_roots_for_interleaved_partial_groups() {
    let xyz: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..4097)
            .map(|i| {
                let group = i % 257;
                let cell = [group % 7, (group / 7) % 7, group / 49];
                cell[axis] as f32 * 3.0 + (((i / 257) >> axis) & 1) as f32 * 0.25 - 2.0
            })
            .collect()
    });
    let roots = euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], 1.0).unwrap();
    assert_eq!(roots, (0..4097).map(|i| (i % 257) as u32).collect::<Vec<_>>());
}

#[test]
fn facade_grid_builder_keeps_stable_indices_empty_cells_and_sentinel() {
    let (sorted, starts) = spatialrust::search::build_grid(
        &[3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0],
        &[0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        &[0.0; 7],
        [0.0; 3],
        [4, 2, 1],
        1.0,
    );
    assert_eq!(sorted, [2, 6, 0, 3, 1, 4, 5]);
    assert_eq!(starts, [0, 0, 2, 2, 4, 6, 6, 7, 7]);
}

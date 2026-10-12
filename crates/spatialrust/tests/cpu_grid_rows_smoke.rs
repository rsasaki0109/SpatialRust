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

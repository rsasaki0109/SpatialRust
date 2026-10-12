use super::*;

fn fixture(n: usize, kind: usize) -> (KdTree, Vec<[f32; 3]>) {
    let points: Vec<_> = (0..n)
        .map(|i| {
            let j = n - i - 1;
            match kind {
                0 => [(j % 41) as f32, (j / 41 % 41) as f32, (j / 1681) as f32],
                1 => [j as f32 * 0.5, 0.0, 0.0],
                2 => [0.0; 3],
                _ => [(j % 7) as f32, (j / 7 % 3) as f32, 0.0],
            }
        })
        .collect();
    let xyz: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
    (KdTree::from_slices(&xyz[0], &xyz[1], &xyz[2]), points)
}

#[test]
fn nearest_plan_rejects_arithmetic_device_and_dispatch_boundaries() {
    let limits = wgpu::Limits::default();
    assert!(buffer_bytes(u64::MAX, 16, &limits).is_err());
    assert!(buffer_bytes(u64::from(u32::MAX) + 1, 0, &limits).is_err());
    assert!(
        buffer_bytes(u64::from(limits.max_storage_buffer_binding_size) / 16, 16, &limits).is_ok()
    );
    assert!(buffer_bytes(u64::from(limits.max_storage_buffer_binding_size) / 16 + 1, 16, &limits)
        .is_err());
    assert!(
        check_compute(&limits, u64::from(limits.max_compute_workgroups_per_dimension) * 64).is_ok()
    );
    assert!(check_compute(
        &limits,
        u64::from(limits.max_compute_workgroups_per_dimension) * 64 + 1
    )
    .is_err());
    for field in 0..9 {
        let mut restricted = limits.clone();
        match field {
            0 => restricted.max_compute_workgroup_size_x = 63,
            1 => restricted.max_compute_invocations_per_workgroup = 63,
            2 => restricted.max_storage_buffers_per_shader_stage = 3,
            3 => restricted.max_bindings_per_bind_group = 4,
            4 => restricted.max_bind_groups = 0,
            5 => restricted.max_uniform_buffers_per_shader_stage = 0,
            6 => restricted.max_uniform_buffer_binding_size = 15,
            7 => restricted.max_buffer_size = 31,
            _ => restricted.max_storage_buffer_binding_size = 31,
        }
        assert!(check_compute(&restricted, 1).is_err(), "field={field}");
    }
}

#[test]
fn gpu_nearest_matches_cpu_traversal_ties_gates_and_brute_distance() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut verified = 0;
    for n in [1, 16, 17, 63, 64, 65, 255, 256, 257, 4097] {
        for kind in 0..4 {
            let (tree, points) = fixture(n, kind);
            let target = GpuKdTree::upload(&runtime, &tree).unwrap();
            assert_eq!(
                target.upload_transfers().host_to_device_bytes(),
                (n * 16 + tree.layout().nodes().len() * 32) as u64
            );
            assert_eq!(target.upload_transfers().device_to_host_bytes(), 0);
            let mut queries: Vec<_> = points
                .iter()
                .step_by((n / 129).max(1))
                .flat_map(|&p| [p, [p[0] + 0.25, p[1] - 0.125, p[2]], [p[0] + 0.5, p[1], p[2]]])
                .collect();
            queries.extend([
                [f32::NAN, 0.0, 0.0],
                [0.0, f32::INFINITY, 0.0],
                [0.0, 0.0, f32::NEG_INFINITY],
                [-500.0; 3],
            ]);
            for gate in [
                0.0,
                f32::from_bits(1.0_f32.to_bits() - 1),
                1.0,
                f32::from_bits(1.0_f32.to_bits() + 1),
                f32::INFINITY,
            ] {
                let expected: Vec<_> = queries
                    .iter()
                    .map(|p| tree.nearest_one_within(p[0], p[1], p[2], gate))
                    .collect();
                let (actual, transfers) =
                    target.nearest_one_within_download(&runtime, &queries, gate).unwrap();
                assert_eq!(actual, expected, "n={n} kind={kind} gate={gate}");
                for (q, result) in
                    queries.iter().zip(&actual).filter(|(q, _)| q.iter().all(|v| v.is_finite()))
                {
                    let minimum = points
                        .iter()
                        .map(|p| {
                            let dx = p[0] - q[0];
                            let dy = p[1] - q[1];
                            let dz = p[2] - q[2];
                            dx * dx + dy * dy + dz * dz
                        })
                        .filter(|&d| d <= gate)
                        .min_by(f32::total_cmp);
                    assert_eq!(result.map(|r| r.distance_squared), minimum);
                }
                assert_eq!(transfers.host_to_device_bytes(), queries.len() as u64 * 16);
                assert_eq!(transfers.device_to_host_bytes(), queries.len() as u64 * 8);
                verified += actual.len();
            }
        }
    }
    eprintln!("GPU_NEAREST verified {verified} ordered results against CPU and brute distances");
}

#[test]
fn gpu_nearest_reuses_owned_target_across_large_partial_batches_and_callers() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let (tree, points) = fixture(65537, 0);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    let queries = points;
    let expected: Vec<_> =
        queries.iter().map(|p| tree.nearest_one_within(p[0], p[1], p[2], 0.0)).collect();
    drop(tree);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (runtime, target, queries, expected) = (&runtime, &target, &queries, &expected);
            scope.spawn(move || {
                for _ in 0..2 {
                    let actual =
                        target.nearest_one_within_download(runtime, queries, 0.0).unwrap().0;
                    assert_eq!(&actual, expected);
                }
            });
        }
    });
    assert!(target.nearest_one_within_download(&runtime, &[], 1.0).unwrap().0.is_empty());
    assert_eq!(
        target.nearest_one_within_download(&runtime, &[[0.0; 3]], 0.0).unwrap().0[0],
        expected.last().copied().unwrap()
    );
}

#[test]
fn gpu_nearest_empty_invalid_inputs_overflow_and_runtime_ownership() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let empty = GpuKdTree::upload(&runtime, &KdTree::from_slices(&[], &[], &[])).unwrap();
    let (results, transfers) =
        empty.nearest_one_within_download(&runtime, &[[0.0; 3]], 1.0).unwrap();
    assert_eq!(results, [None]);
    assert_eq!(transfers, TransferStats::default());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            GpuKdTree::upload(&runtime, &KdTree::from_slices(&[value], &[0.0], &[0.0])).is_err()
        );
    }
    let tree = KdTree::from_slices(&[f32::MAX, -f32::MAX], &[0.0; 2], &[0.0; 2]);
    let target = GpuKdTree::upload(&runtime, &tree).unwrap();
    for gate in [-1.0, f32::NAN, f32::NEG_INFINITY] {
        let (results, transfers) =
            target.nearest_one_within_download(&runtime, &[[0.0; 3]], gate).unwrap();
        assert_eq!(results, [None]);
        assert_eq!(transfers, TransferStats::default());
    }
    let actual =
        target.nearest_one_within_download(&runtime, &[[0.0; 3]], f32::INFINITY).unwrap().0;
    assert_eq!(actual, [tree.nearest_one_within(0.0, 0.0, 0.0, f32::INFINITY)]);
    let other = WgpuRuntime::new_headless().unwrap();
    assert!(target.nearest_one_within_download(&other, &[], 1.0).is_err());
    assert!(empty.nearest_one_within_download(&other, &[[0.0; 3]], f32::NAN).is_err());
}

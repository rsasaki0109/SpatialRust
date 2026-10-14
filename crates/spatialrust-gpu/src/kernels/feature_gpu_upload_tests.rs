use super::*;

#[path = "feature_gpu_upload_benchmark.rs"]
mod benchmark;

fn normals(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| match i % 4 {
            0 => [0.6, 0.8, -0.0],
            1 => [-1.0, f32::from_bits(1), f32::from_bits(0x8000_0001)],
            2 => [0.0, -1.0, -0.0],
            _ => [-0.0, 0.0, 1.0],
        })
        .collect()
}

#[test]
#[ignore = "requires native GPU adapter"]
fn gather_input_words_receipts_empty_and_cpu_drop_are_exact() {
    let r = WgpuRuntime::new_headless().unwrap();
    let (tree, _, _) = fixture(0, 0);
    assert_eq!(
        GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &[]).unwrap().upload_transfers(),
        TransferStats::default()
    );
    assert_eq!(
        GpuIssInput::upload_with_gpu_gather(&r, &tree).unwrap().upload_transfers(),
        TransferStats::default()
    );
    assert_eq!(
        GpuMlsInput::upload_with_gpu_gather(&r, &tree).unwrap().upload_transfers(),
        TransferStats::default()
    );
    assert!(r.tree_gather_pipeline.get().is_none() && r.fpfh_sample_pipeline.get().is_none());
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 17, 63, 64, 65, 257, 65_537, 262_145, 524_289] {
        for profile in 0..4 {
            let (tree, points, _) = fixture(n, profile);
            let normals = normals(n);
            let node_bytes = tree.layout().nodes().len() as u64 * 32;
            let old_f = GpuFpfhInput::upload(&r, &tree, &normals).unwrap();
            let f = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).unwrap();
            let old_i = GpuIssInput::upload(&r, &tree).unwrap();
            let i = GpuIssInput::upload_with_gpu_gather(&r, &tree).unwrap();
            let old_m = GpuMlsInput::upload(&r, &tree).unwrap();
            let m = GpuMlsInput::upload_with_gpu_gather(&r, &tree).unwrap();
            drop(tree);
            for (receipt, h2d, d2d) in [
                (f.upload_transfers(), n as u64 * 28 + node_bytes, n as u64 * 64),
                (i.upload_transfers(), n as u64 * 16 + node_bytes, n as u64 * 32),
                (m.upload_transfers(), n as u64 * 16 + node_bytes, n as u64 * 32),
            ] {
                assert_eq!(receipt.host_to_device_bytes(), h2d);
                assert_eq!(receipt.device_to_device_bytes(), d2d);
                assert_eq!(receipt.device_to_host_bytes(), 0);
            }
            assert_eq!(f.point_count(), n as u32);
            assert_eq!(i.point_count(), n as u32);
            assert_eq!(m.point_count(), n as u32);
            compare_query(
                &probe,
                &r,
                fpfh::soa_upload_reference::query_buffer(&old_f),
                fpfh::soa_upload_reference::query_buffer(&f),
                &points,
            );
            compare_query(
                &probe,
                &r,
                iss::soa_upload_reference::query_buffer(&old_i),
                iss::soa_upload_reference::query_buffer(&i),
                &points,
            );
            compare_query(
                &probe,
                &r,
                mls::soa_upload_reference::query_buffer(&old_m),
                mls::soa_upload_reference::query_buffer(&m),
                &points,
            );
            let expected: Vec<u32> = points
                .iter()
                .zip(&normals)
                .flat_map(|(p, n)| {
                    [
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                        0,
                        n[0].to_bits(),
                        n[1].to_bits(),
                        n[2].to_bits(),
                        0,
                    ]
                })
                .collect();
            drop(points);
            drop(normals);
            assert_eq!(
                probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&old_f).unwrap()),
                expected
            );
            assert_eq!(
                probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&f).unwrap()),
                expected
            );
            words += n * 20;
        }
    }
    println!("FEATURE_GPU_INPUT_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn cold_concurrent_sample_gathers_keep_outputs_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    let fixtures: Vec<_> = (0..3).map(|profile| fixture(4097, profile)).collect();
    let normals = normals(4097);
    assert!(r.fpfh_sample_pipeline.get().is_none());
    let inputs = std::thread::scope(|scope| {
        let r = &r;
        let normals = &normals;
        fixtures
            .iter()
            .map(|(tree, _, _)| {
                scope.spawn(move || GpuFpfhInput::upload_with_gpu_gather(r, tree, normals).unwrap())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let probe = Probe::new(&r);
    for ((tree, _, _), input) in fixtures.iter().zip(inputs) {
        let old = GpuFpfhInput::upload(&r, tree, &normals).unwrap();
        assert_eq!(
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&old).unwrap()),
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&input).unwrap())
        );
    }
}

#[test]
#[ignore = "requires native GPU adapter"]
fn gather_rejections_match_frozen_constructors() {
    let r = WgpuRuntime::new_headless().unwrap();
    r.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut cases = 0;
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut axes = [[0.0; 3]; 3];
            axes[axis][1] = value;
            let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
            assert_eq!(
                GpuFpfhInput::upload(&r, &tree, &[[0.0, 0.0, 1.0]; 3]).err().unwrap().to_string(),
                GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &[[0.0, 0.0, 1.0]; 3])
                    .err()
                    .unwrap()
                    .to_string()
            );
            assert_eq!(
                GpuIssInput::upload(&r, &tree).err().unwrap().to_string(),
                GpuIssInput::upload_with_gpu_gather(&r, &tree).err().unwrap().to_string()
            );
            assert_eq!(
                GpuMlsInput::upload(&r, &tree).err().unwrap().to_string(),
                GpuMlsInput::upload_with_gpu_gather(&r, &tree).err().unwrap().to_string()
            );
            cases += 3;
        }
    }
    let huge = KdTree::from_slices(&[-f32::MAX, f32::MAX], &[0.0; 2], &[0.0; 2]);
    assert_eq!(
        GpuFpfhInput::upload(&r, &huge, &[[0.0, 0.0, 1.0]; 2]).err().unwrap().to_string(),
        GpuFpfhInput::upload_with_gpu_gather(&r, &huge, &[[0.0, 0.0, 1.0]; 2])
            .err()
            .unwrap()
            .to_string()
    );
    assert_eq!(
        GpuIssInput::upload(&r, &huge).err().unwrap().to_string(),
        GpuIssInput::upload_with_gpu_gather(&r, &huge).err().unwrap().to_string()
    );
    assert_eq!(
        GpuMlsInput::upload(&r, &huge).err().unwrap().to_string(),
        GpuMlsInput::upload_with_gpu_gather(&r, &huge).err().unwrap().to_string()
    );
    cases += 3;
    let tree = KdTree::from_slices(&[0.0], &[-0.0], &[0.0]);
    for normals in [
        vec![],
        vec![[0.0; 3]],
        vec![[0.0, 0.0, 2.0]],
        vec![[f32::NAN, 0.0, 1.0]],
        vec![[f32::INFINITY, 0.0, 1.0]],
    ] {
        assert_eq!(
            GpuFpfhInput::upload(&r, &tree, &normals).err().unwrap().to_string(),
            GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).err().unwrap().to_string()
        );
        cases += 1;
    }
    assert!(r.tree_gather_pipeline.get().is_none());
    assert!(r.fpfh_sample_pipeline.get().is_none());
    r.wait_idle();
    assert!(pollster::block_on(r.device().pop_error_scope()).is_none());
    println!("FEATURE_GPU_REJECTIONS exact_errors={cases}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn gather_preserves_resident_outputs_status_and_input_drop_ownership() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut calls = 0;
    let mut words = 0;
    for n in [1, 17, 65, 257] {
        for profile in 0..3 {
            let (tree, _points, normals) = fixture(n, profile);
            let old = GpuFpfhInput::upload(&r, &tree, &normals).unwrap();
            let new = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).unwrap();
            let a = old.radius_neighborhoods_resident(&r, 0.35, 64).unwrap();
            let b = new.radius_neighborhoods_resident(&r, 0.35, 64).unwrap();
            assert!(new.descriptors_resident(&r, &a).is_err());
            assert!(new.descriptors_resident(&other, &b).is_err());
            let ao = old.descriptors_resident(&r, &a).unwrap();
            let bo = new.descriptors_resident(&r, &b).unwrap();
            drop(old);
            drop(new);
            assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
            assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
            assert_eq!(
                ao.download(&r)
                    .map(|v| (
                        v.0.into_iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        v.1
                    ))
                    .map_err(|e| e.to_string()),
                bo.download(&r)
                    .map(|v| (
                        v.0.into_iter().map(|row| row.map(f32::to_bits)).collect::<Vec<_>>(),
                        v.1
                    ))
                    .map_err(|e| e.to_string())
            );
            calls += 1;
            words += n * 33 + 4;
            let old = GpuIssInput::upload(&r, &tree).unwrap();
            let new = GpuIssInput::upload_with_gpu_gather(&r, &tree).unwrap();
            let a = old.neighborhoods_resident(&r, 0.35, 0.3, 64, 64).unwrap();
            let b = new.neighborhoods_resident(&r, 0.35, 0.3, 64, 64).unwrap();
            assert!(new.keypoints_resident(&r, &a, 0.975, 0.975, 5).is_err());
            assert!(new.keypoints_resident(&other, &b, 0.975, 0.975, 5).is_err());
            let ao = old.keypoints_resident(&r, &a, 0.975, 0.975, 5).unwrap();
            let bo = new.keypoints_resident(&r, &b, 0.975, 0.975, 5).unwrap();
            drop(old);
            drop(new);
            assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
            assert_eq!(probe.read(&r, ao.scores_buffer()), probe.read(&r, bo.scores_buffer()));
            assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
            assert_eq!(
                ao.download(&r).map_err(|e| e.to_string()),
                bo.download(&r).map_err(|e| e.to_string())
            );
            calls += 1;
            words += n * 5 + 8;
            let old = GpuMlsInput::upload(&r, &tree).unwrap();
            let new = GpuMlsInput::upload_with_gpu_gather(&r, &tree).unwrap();
            drop(tree);
            let a = old.neighborhoods_resident(&r, 0.35, 64).unwrap();
            let b = new.neighborhoods_resident(&r, 0.35, 64).unwrap();
            assert!(new.smooth_resident(&r, &a, 0, 5).is_err());
            assert!(new.smooth_resident(&other, &b, 0, 5).is_err());
            let mut owned = Vec::new();
            for order in 0..=2 {
                owned.push((
                    old.smooth_resident(&r, &a, order, 5).unwrap(),
                    new.smooth_resident(&r, &b, order, 5).unwrap(),
                ));
            }
            drop(old);
            drop(new);
            for (ao, bo) in owned {
                assert_eq!(probe.read(&r, ao.buffer()), probe.read(&r, bo.buffer()));
                assert_eq!(probe.read(&r, ao.status_buffer()), probe.read(&r, bo.status_buffer()));
                match (ao.download(&r), bo.download(&r)) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(
                            a.positions
                                .into_iter()
                                .map(|p| p.map(f32::to_bits))
                                .collect::<Vec<_>>(),
                            b.positions
                                .into_iter()
                                .map(|p| p.map(f32::to_bits))
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(a.modes, b.modes);
                        assert_eq!(a.diagnostics, b.diagnostics);
                        assert_eq!(a.transfers, b.transfers);
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("MLS validity changed"),
                }
                calls += 1;
                words += n * 4 + 8;
            }
        }
    }
    println!("FEATURE_GPU_OUTPUTS calls={calls} exact_words={words}");
}

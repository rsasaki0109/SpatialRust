use super::*;

#[path = "fpfh_fused_upload_benchmark.rs"]
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
fn public_fusion_boundary_retains_large_two_pass_cache_bits_and_receipts() {
    for n in [1_048_575, 1_048_576, 1_048_577] {
        let r = WgpuRuntime::new_headless().unwrap();
        let probe = Probe::new(&r);
        let (tree, points, _) = fixture(n, 0);
        let ns = normals(n);
        let input = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &ns).unwrap();
        assert_eq!(r.fpfh_sample_pipeline.get().is_some(), n >= 1_048_576);
        let previous = fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &ns).unwrap();
        assert_eq!(input.upload_transfers(), previous.upload_transfers());
        assert_eq!(input.point_count(), n as u32);
        drop(tree);
        drop(ns);
        compare_query(
            &probe,
            &r,
            fpfh::soa_upload_reference::query_buffer(&previous),
            fpfh::soa_upload_reference::query_buffer(&input),
            &points,
        );
        drop(points);
        assert_eq!(
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&previous).unwrap()),
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&input).unwrap())
        );
    }
}

#[test]
#[ignore = "requires native GPU adapter"]
fn fused_public_inputs_match_two_pass_bits_receipts_and_cpu_drop() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [0, 1, 63, 64, 65, 257, 65_537, 524_289] {
        for profile in 0..4 {
            let (tree, points, _) = fixture(n, profile);
            let normals = normals(n);
            let old = fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &normals).unwrap();
            let new = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).unwrap();
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            assert_eq!(old.point_count(), new.point_count());
            if n > 0 {
                assert_eq!(
                    new.upload_transfers().host_to_device_bytes(),
                    n as u64 * 28 + tree.layout().nodes().len() as u64 * 32
                );
                assert_eq!(new.upload_transfers().device_to_device_bytes(), n as u64 * 64);
                assert_eq!(new.upload_transfers().device_to_host_bytes(), 0);
            }
            drop(tree);
            compare_query(
                &probe,
                &r,
                fpfh::soa_upload_reference::query_buffer(&old),
                fpfh::soa_upload_reference::query_buffer(&new),
                &points,
            );
            if n > 0 {
                let expected: Vec<_> = points
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
                    probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&old).unwrap()),
                    expected
                );
                assert_eq!(
                    probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&new).unwrap()),
                    expected
                );
            }
            words += n * 12;
        }
    }
    println!("FPFH_FUSED_PUBLIC_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn fused_descriptors_status_runtime_and_owned_outputs_match_two_pass() {
    let r = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [1, 17, 65, 257] {
        for profile in 0..3 {
            let (tree, points, _) = fixture(n, profile);
            let normals = normals(n);
            let old = fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &normals).unwrap();
            let new = GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).unwrap();
            drop(tree);
            drop(points);
            drop(normals);
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
                ao.download(&r).map_err(|e| e.to_string()),
                bo.download(&r).map_err(|e| e.to_string())
            );
            words += n * 33 + 4;
        }
    }
    println!("FPFH_FUSED_OUTPUT_WORDS {words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn fused_errors_and_concurrent_first_call_keep_frozen_contract() {
    let r = WgpuRuntime::new_headless().unwrap();
    r.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for axis in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut axes = [vec![0.0; 65], vec![0.0; 65], vec![0.0; 65]];
            axes[axis][64] = bad;
            let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
            let ns = normals(65);
            assert_eq!(
                fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &ns).err().unwrap(),
                GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &ns).err().unwrap()
            );
        }
    }
    let tree = KdTree::from_slices(&[-f32::MAX, f32::MAX], &[0.0; 2], &[0.0; 2]);
    assert_eq!(
        fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &normals(2)).err().unwrap(),
        GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals(2)).err().unwrap()
    );
    let tree = fixture(1, 0).0;
    for ns in [
        vec![],
        vec![[0.0; 3]],
        vec![[0.0, 0.0, 2.0]],
        vec![[f32::NAN, 0.0, 1.0]],
        vec![[f32::INFINITY, 0.0, 1.0]],
    ] {
        assert_eq!(
            fpfh::two_pass_reference::previous_gpu_upload(&r, &tree, &ns).err().unwrap(),
            GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &ns).err().unwrap()
        );
    }
    assert!(r.tree_gather_pipeline.get().is_none() && r.fpfh_sample_pipeline.get().is_none());
    r.wait_idle();
    assert!(pollster::block_on(r.device().pop_error_scope()).is_none());
    let fixtures: Vec<_> = (0..3).map(|p| fixture(4097, p)).collect();
    let ns = normals(4097);
    let inputs = std::thread::scope(|scope| {
        let r = &r;
        let ns = &ns;
        fixtures
            .iter()
            .map(|(t, _, _)| {
                scope.spawn(move || GpuFpfhInput::upload_with_gpu_gather(r, t, ns).unwrap())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(r.fpfh_sample_pipeline.get().is_none());
    let probe = Probe::new(&r);
    for ((tree, _, _), input) in fixtures.iter().zip(inputs) {
        let old = fpfh::two_pass_reference::previous_gpu_upload(&r, tree, &ns).unwrap();
        assert_eq!(
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&old).unwrap()),
            probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&input).unwrap())
        );
    }
}

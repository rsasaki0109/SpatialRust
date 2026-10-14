use super::*;

#[test]
#[ignore = "requires native GPU adapter"]
fn validation_preserves_both_constructor_routes_and_uploaded_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let probe = Probe::new(&r);
    let mut words = 0;
    for n in [0, 17, 524_287, 524_288, 524_289, 1_048_577] {
        let (tree, points, normals) = fixture(n, if n == 524_288 { 1 } else { 0 });
        for gpu in [false, true] {
            let old =
                fpfh::validation_reference::previous_upload(&r, &tree, &normals, gpu).unwrap();
            let new = if gpu {
                GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals)
            } else {
                GpuFpfhInput::upload(&r, &tree, &normals)
            }
            .unwrap();
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            compare_query(
                &probe,
                &r,
                fpfh::soa_upload_reference::query_buffer(&old),
                fpfh::soa_upload_reference::query_buffer(&new),
                &points,
            );
            if n > 0 {
                let old = probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&old).unwrap());
                let new = probe.read(&r, fpfh::soa_upload_reference::sample_buffer(&new).unwrap());
                assert_eq!(old, new);
                words += new.len();
            }
            let old = iss::validation_reference::previous_upload(&r, &tree, gpu).unwrap();
            let new = if gpu {
                GpuIssInput::upload_with_gpu_gather(&r, &tree)
            } else {
                GpuIssInput::upload(&r, &tree)
            }
            .unwrap();
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            compare_query(
                &probe,
                &r,
                iss::soa_upload_reference::query_buffer(&old),
                iss::soa_upload_reference::query_buffer(&new),
                &points,
            );
            let old = mls::validation_reference::previous_upload(&r, &tree, gpu).unwrap();
            let new = if gpu {
                GpuMlsInput::upload_with_gpu_gather(&r, &tree)
            } else {
                GpuMlsInput::upload(&r, &tree)
            }
            .unwrap();
            assert_eq!(old.upload_transfers(), new.upload_transfers());
            compare_query(
                &probe,
                &r,
                mls::soa_upload_reference::query_buffer(&old),
                mls::soa_upload_reference::query_buffer(&new),
                &points,
            );
            words += n * 12;
        }
    }
    println!("FEATURE_VALIDATION_UPLOAD matched_words={words}");
}

#[test]
#[ignore = "requires native GPU adapter"]
fn large_invalid_inputs_retain_feature_errors_and_normal_length_precedence() {
    let r = WgpuRuntime::new_headless().unwrap();
    let n = 524_289;
    let normals = vec![[0.0, 0.0, 1.0]; n];
    for (axis, value) in [(0, f32::NAN), (1, f32::INFINITY), (2, f32::NEG_INFINITY), (0, f32::MAX)]
    {
        let mut axes: [Vec<_>; 3] = std::array::from_fn(|_| vec![0.0; n]);
        axes[axis][n - 1] = value;
        let tree = KdTree::from_slices(&axes[0], &axes[1], &axes[2]);
        for gpu in [false, true] {
            assert_eq!(
                fpfh::validation_reference::previous_upload(&r, &tree, &normals, gpu).err(),
                if gpu {
                    GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals)
                } else {
                    GpuFpfhInput::upload(&r, &tree, &normals)
                }
                .err()
            );
            assert_eq!(
                iss::validation_reference::previous_upload(&r, &tree, gpu).err(),
                if gpu {
                    GpuIssInput::upload_with_gpu_gather(&r, &tree)
                } else {
                    GpuIssInput::upload(&r, &tree)
                }
                .err()
            );
            assert_eq!(
                mls::validation_reference::previous_upload(&r, &tree, gpu).err(),
                if gpu {
                    GpuMlsInput::upload_with_gpu_gather(&r, &tree)
                } else {
                    GpuMlsInput::upload(&r, &tree)
                }
                .err()
            );
            assert_eq!(
                fpfh::validation_reference::previous_upload(&r, &tree, &[], gpu).err(),
                if gpu {
                    GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &[])
                } else {
                    GpuFpfhInput::upload(&r, &tree, &[])
                }
                .err()
            );
        }
    }
    let (tree, _, mut normals) = fixture(n, 0);
    normals[n - 1] = [0.0; 3];
    assert_eq!(
        fpfh::validation_reference::previous_upload(&r, &tree, &normals, true).err(),
        GpuFpfhInput::upload_with_gpu_gather(&r, &tree, &normals).err()
    );
}

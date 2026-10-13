use super::*;

fn radius_reference(points: &[[f32; 3]], i: usize, radius: f32) -> ([f64; 6], Mat3<f64>, usize) {
    // Independent brute-force inclusive membership, including self/duplicates.
    let neighbors: Vec<_> = points
        .iter()
        .filter(|p| {
            let d: [f32; 3] = std::array::from_fn(|a| p[a] - points[i][a]);
            d.iter().map(|v| v * v).sum::<f32>() <= radius * radius
        })
        .collect();
    let zero = Mat3::from_rows([0.0; 3], [0.0; 3], [0.0; 3]);
    if neighbors.len() < 3 {
        return (
            [f64::from(0.001_f32), f64::from(0.001_f32), f64::from(0.001_f32), 0.0, 0.0, 0.0],
            zero,
            neighbors.len(),
        );
    }
    let anchor = points[i].map(f64::from);
    let scale = neighbors
        .iter()
        .flat_map(|p| (0..3).map(|a| (f64::from(p[a]) - anchor[a]).abs()))
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return ([0.001, 1.0, 1.0, 0.0, 0.0, 0.0], zero, neighbors.len());
    }
    let mean: [f64; 3] = std::array::from_fn(|a| {
        neighbors.iter().map(|p| (f64::from(p[a]) - anchor[a]) / scale).sum::<f64>()
            / neighbors.len() as f64
    });
    let mut c = [[0.0; 3]; 3];
    for p in &neighbors {
        let d: [f64; 3] = std::array::from_fn(|a| (f64::from(p[a]) - anchor[a]) / scale - mean[a]);
        for a in 0..3 {
            for b in 0..3 {
                c[a][b] += d[a] * d[b];
            }
        }
    }
    let raw = Mat3::from_rows(c[0], c[1], c[2]);
    let e = symmetric_eigen3(raw);
    let n: [f64; 3] = std::array::from_fn(|a| e.eigenvectors.m[a][0]);
    let epsilon = f64::from(0.001_f32);
    (
        [
            1.0 + (epsilon - 1.0) * n[0] * n[0],
            1.0 + (epsilon - 1.0) * n[1] * n[1],
            1.0 + (epsilon - 1.0) * n[2] * n[2],
            (epsilon - 1.0) * n[0] * n[1],
            (epsilon - 1.0) * n[0] * n[2],
            (epsilon - 1.0) * n[1] * n[2],
        ],
        raw,
        neighbors.len(),
    )
}

#[test]
fn resident_radius_covariances_match_brute_force_planes_and_spectral_invariants() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    let mut maximum = 0.0_f64;
    for n in [1, 2, 3, 63, 64, 65, 257, 1025] {
        for scale in [1e-4_f32, 1.0, 1e4] {
            let points: Vec<_> = (0..n)
                .map(|i| {
                    let a = hash(i as u32 + 1);
                    let b = hash(i as u32 + 0x10203);
                    [(a + 4.0) * scale, (b - 3.0) * scale, (0.3 * a - 0.2 * b + 2.0) * scale]
                })
                .collect();
            let tree = tree(&points);
            for r in [0.01, 0.8, 20.0] {
                let radius = r * scale;
                let input = GpuGicpInput::upload_with_gpu_radius_covariances(
                    &runtime, &tree, &tree, radius, n as u32, 0.001,
                )
                .unwrap();
                let cov = covariances(&runtime, &input);
                assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                assert_eq!(
                    input.upload_transfers().host_to_device_bytes(),
                    (64 * (n + tree.layout().nodes().len())) as u64
                );
                for i in 0..n {
                    assert_eq!(cov[i], cov[n + i]);
                    let (expected, raw, count) = radius_reference(&points, i, radius);
                    let e = symmetric_eigen3(matrix(cov[i]));
                    let spectrum = symmetric_eigen3(raw).eigenvalues;
                    let weights = if count < 3 { [0.001; 3] } else { [0.001, 1.0, 1.0] };
                    for (a, b) in e.eigenvalues.iter().zip(weights) {
                        assert!((a - b).abs() < 2e-6);
                    }
                    if count >= 3 {
                        let normal: [f64; 3] = std::array::from_fn(|a| e.eigenvectors.m[a][0]);
                        let energy = (0..3)
                            .flat_map(|a| (0..3).map(move |b| normal[a] * raw.m[a][b] * normal[b]))
                            .sum::<f64>();
                        assert!(
                            energy <= spectrum[0] + 1e-10 + 3e-5 * spectrum[2].abs(),
                            "n={n} r={r} scale={scale} i={i} energy={energy}"
                        );
                    }
                    if count < 3
                        || spectrum[1] - spectrum[0] > (spectrum[2].abs() * 1e-4).max(1e-11)
                    {
                        for a in 0..6 {
                            let error = (f64::from(cov[i][a]) - expected[a]).abs();
                            maximum = maximum.max(error);
                            assert!(error < 3e-4, "n={n} r={r} scale={scale} i={i} error={error}");
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!("GPU_RESIDENT_RADIUS_COVARIANCE checked={checked}; maximum centered f64 component error={maximum}");
}

#[test]
fn radius_boundaries_sparse_fallback_duplicates_overflow_and_numeric_errors() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    // Exactly-on-radius samples must participate; the next representable value does not.
    let points = [
        [0.0; 3],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [f32::from_bits(0.5_f32.to_bits() + 1), 0.0, 0.0],
    ];
    let t = tree(&points);
    let input =
        GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, &t, &t, 0.5, 4, 0.001).unwrap();
    assert_eq!(covariances(&runtime, &input)[0], [1.0, 1.0, 0.001, 0.0, 0.0, 0.0]);
    for epsilon in [1e-4, 0.001, 1.0] {
        let t = tree(&[[0.0; 3], [0.25, 0.0, 0.0]]);
        let input =
            GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, &t, &t, 0.5, 2, epsilon)
                .unwrap();
        assert!(covariances(&runtime, &input)
            .iter()
            .all(|c| *c == [epsilon, epsilon, epsilon, 0.0, 0.0, 0.0]));
    }
    let duplicate = tree(&[[0.0; 3]; 65]);
    let singleton = tree(&[[0.0; 3]]);
    let good = GpuGicpInput::upload_with_gpu_radius_covariances(
        &runtime, &duplicate, &singleton, 0.5, 65, 0.001,
    )
    .unwrap();
    let c = covariances(&runtime, &good);
    assert_eq!(c[0], [0.001, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert_eq!(c[65], [0.001, 0.001, 0.001, 0.0, 0.0, 0.0]);
    for second in [false, true] {
        let (s, t) = if second { (&singleton, &duplicate) } else { (&duplicate, &singleton) };
        let bad = GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, s, t, 0.5, 64, 0.001)
            .unwrap();
        let w = words(&runtime, &bad.covariances);
        assert_eq!(w[0], 8);
        assert!(w[1..].iter().all(|v| *v == 0));
        let eq = bad.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
        let w = words(&runtime, eq.buffer());
        assert_eq!(w[0], 16);
        assert!(w[1..].iter().all(|v| *v == 0));
        assert!(eq.download(&runtime).is_err());
    }
    let tiny = tree(&[[0.0; 3], [1e-22, 0.0, 0.0], [0.0, 1e-22, 0.0]]);
    let bad = GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, &tiny, &t, 0.5, 4, 0.001)
        .unwrap();
    let w = words(&runtime, &bad.covariances);
    assert_eq!(w[0], 2);
    assert!(w[1..].iter().all(|v| *v == 0));
    for r in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e20, 1e-22] {
        assert!(GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, &t, &t, r, 4, 0.001)
            .is_err());
    }
    for capacity in [0, u32::MAX] {
        assert!(GpuGicpInput::upload_with_gpu_radius_covariances(
            &runtime, &t, &t, 0.5, capacity, 0.001
        )
        .is_err());
    }
    assert!(GpuGicpInput::upload_with_gpu_radius_covariances(
        &runtime,
        &tree(&[]),
        &t,
        0.5,
        4,
        0.001
    )
    .is_err());
    assert!(GpuGicpInput::upload_with_gpu_radius_covariances(
        &runtime,
        &tree(&[[f32::NAN, 0.0, 0.0]]),
        &t,
        0.5,
        4,
        0.001
    )
    .is_err());
    assert_eq!(bounded_covariance_plan(1, 2, 128, 0.001, &wgpu::Limits::default()).unwrap(), 128);
    assert!(bounded_covariance_plan(65537, 65537, 1024, 0.001, &wgpu::Limits::default()).is_err());
}

#[test]
fn radius_covariances_match_unique_legacy_grid_planes_and_feed_owned_equations() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let points: Vec<_> = (0..257)
        .map(|i| {
            let a = hash(i + 1);
            let b = hash(i + 0x10203);
            [a, b, 0.3 * a - 0.2 * b + 2.0]
        })
        .collect();
    let t = tree(&points);
    let input =
        GpuGicpInput::upload_with_gpu_radius_covariances(&runtime, &t, &t, 20.0, 257, 0.001)
            .unwrap();
    let c = covariances(&runtime, &input);
    let (x, y, z) = t.layout().positions();
    let old =
        crate::estimate_plane_covariances_grid_gpu_with_receipt(&runtime, x, y, z, 20.0, 0.001)
            .unwrap()
            .0;
    for (a, b) in c[..257].iter().zip(&old) {
        for axis in 0..6 {
            assert!((a[axis] - b[axis]).abs() < 3e-5);
        }
    }
    let uploaded = GpuGicpInput::upload(&runtime, &points, &c[..257], &t, &c[257..]).unwrap();
    let pose = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.01, -0.005, 0.002));
    let result = input.equations_resident(&runtime, pose, 0.25).unwrap();
    let actual = result.download(&runtime).unwrap();
    let expected =
        uploaded.equations_resident(&runtime, pose, 0.25).unwrap().download(&runtime).unwrap();
    assert_eq!(actual.squared_error_sum, expected.squared_error_sum);
    for a in 0..6 {
        assert!(
            (actual.gradient[a] * actual.weight_scale
                - expected.gradient[a] * expected.weight_scale)
                .abs()
                < 2e-4 * (1.0 + (expected.gradient[a] * expected.weight_scale).abs())
        );
        for b in 0..6 {
            assert!(
                (actual.hessian[a][b] * actual.weight_scale
                    - expected.hessian[a][b] * expected.weight_scale)
                    .abs()
                    < 2e-4 * (1.0 + (expected.hessian[a][b] * expected.weight_scale).abs())
            );
        }
    }
    assert!(input.equations_resident(&other, pose, 0.25).is_err());
    drop(input);
    assert_eq!(result.download(&runtime).unwrap(), actual);
    assert!(result.download(&other).is_err());
}

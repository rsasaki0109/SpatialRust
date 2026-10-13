//! Exact serial per-point f64 oracle from e7cabc2, preserving neighbor order.
use super::*;
use crate::gicp::{covariances, regularize};
use spatialrust_math::{Mat3, Vec3};
use spatialrust_search::NearestNeighborIndex;

fn previous_covariances(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tree: &KdTree,
    k: usize,
    epsilon: f64,
) -> Vec<M3> {
    let len = x.len();
    let mut out = Vec::with_capacity(len);
    for index in 0..len {
        let neighbors = tree.nearest_k(x[index], y[index], z[index], k);
        let mut acc = spatialrust_math::CovarianceAccumulator3::new();
        for neighbor in &neighbors {
            acc.push(Vec3::new(x[neighbor.index], y[neighbor.index], z[neighbor.index]));
        }
        let cov = acc
            .covariance()
            .unwrap_or_else(|| Mat3::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]));
        out.push(regularize(cov, epsilon));
    }
    out
}

#[test]
fn bounded_packed_rows_match_frozen_f64_covariances_and_serial_cpu_rows() {
    let mut rows_checked = 0;
    for n in [0, 1, 63, 4095, 16384, 32767, 32768, 32769, 65537] {
        for duplicates in [false, true] {
            let points: Vec<_> = (0..n)
                .map(|i| {
                    let j = if duplicates { i / 2 } else { i };
                    [
                        4.0 + (j % 37) as f32 * 0.125,
                        -3.0 + (j / 37 % 37) as f32 * 0.125,
                        2.0 + (j / 1369) as f32 * 0.125,
                    ]
                })
                .collect();
            let c: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
            let t = KdTree::from_slices(&c[0], &c[1], &c[2]);
            let reference = previous_covariances(&c[0], &c[1], &c[2], &t, 6, 0.001);
            assert_eq!(
                covariances(&c[0], &c[1], &c[2], &t, 6, 0.001)
                    .iter()
                    .flatten()
                    .flatten()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference.iter().flatten().flatten().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let expected: Vec<_> =
                reference.into_iter().map(|m| packed(m).map(f32::to_bits)).collect();
            for workers in [1, 2, 6] {
                let actual = prepare_with_workers(&c[0], &c[1], &c[2], &t, 6, 0.001, workers);
                assert_eq!(
                    actual.iter().map(|m| m.map(f32::to_bits)).collect::<Vec<_>>(),
                    expected,
                    "n={n} duplicates={duplicates} workers={workers}"
                );
                rows_checked += actual.len();
            }
        }
    }
    for scale in [1e-4_f32, 1.0, 1e4] {
        for epsilon in [1e-4, 0.001, 0.1] {
            for k in [3, 6, 20] {
                let x = vec![4.0 * scale; 65];
                let y = vec![-3.0 * scale; 65];
                let z = vec![2.0 * scale; 65];
                let t = KdTree::from_slices(&x, &y, &z);
                let expected: Vec<_> = previous_covariances(&x, &y, &z, &t, k, epsilon)
                    .into_iter()
                    .map(|m| packed(m).map(f32::to_bits))
                    .collect();
                assert_eq!(
                    prepare(&x, &y, &z, &t, k, epsilon)
                        .into_iter()
                        .map(|m| m.map(f32::to_bits))
                        .collect::<Vec<_>>(),
                    expected
                );
                rows_checked += 65;
            }
        }
    }
    eprintln!("GICP_PACKED_PREPARATION exact_rows={rows_checked}");
}

#[test]
fn preparation_worker_count_stays_bounded_and_concurrent_rows_keep_order() {
    for available in [0, 1, 2, 6, usize::MAX] {
        for n in [0, 4095, 16383, 16384, 32767] {
            assert_eq!(worker_count(n, available), 1);
        }
        assert_eq!(worker_count(32768, available), available.clamp(1, 2));
        assert_eq!(worker_count(65537, available), available.clamp(1, 4));
    }
    let n = 32769;
    let x: Vec<_> = (0..n).map(|i| (i % 37) as f32 * 0.125).collect();
    let y: Vec<_> = (0..n).map(|i| (i / 37 % 37) as f32 * 0.125).collect();
    let z: Vec<_> = (0..n).map(|i| (i / 1369) as f32 * 0.125).collect();
    let t = KdTree::from_slices(&x, &y, &z);
    let expected = prepare_with_workers(&x, &y, &z, &t, 6, 0.001, 1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| prepare_with_workers(&x, &y, &z, &t, 6, 0.001, 2)))
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), expected);
        }
    });
}

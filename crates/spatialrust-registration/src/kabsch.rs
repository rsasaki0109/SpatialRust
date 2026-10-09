use spatialrust_math::{Isometry3, Quat, Vec3};

/// Estimates the least-squares proper rigid transform mapping paired source to target.
///
/// Uses f64 centroids and Horn's unit-quaternion formulation with a normalized
/// symmetric eigensystem. Returns None for short, mismatched or nonfinite inputs,
/// or a transform that cannot be represented in f32. Collinear/coincident inputs
/// can admit multiple equally valid rotations; the returned pose is not a
/// certificate of observability or of correct correspondences.
#[must_use]
pub fn estimate_rigid_transform(
    source: &[Vec3<f32>],
    target: &[Vec3<f32>],
) -> Option<Isometry3<f32>> {
    if source.len() != target.len() || source.len() < 3 {
        return None;
    }
    let mut mean_source = [0.0_f64; 3];
    let mut mean_target = [0.0_f64; 3];
    for (src, dst) in source.iter().zip(target) {
        for (i, (s, t)) in [src.x, src.y, src.z].into_iter().zip([dst.x, dst.y, dst.z]).enumerate()
        {
            if !s.is_finite() || !t.is_finite() {
                return None;
            }
            mean_source[i] += f64::from(s);
            mean_target[i] += f64::from(t);
        }
    }
    for i in 0..3 {
        mean_source[i] /= source.len() as f64;
        mean_target[i] /= source.len() as f64;
    }
    let mut h = [[0.0_f64; 3]; 3];
    for (src, dst) in source.iter().zip(target) {
        let s = [f64::from(src.x), f64::from(src.y), f64::from(src.z)];
        let t = [f64::from(dst.x), f64::from(dst.y), f64::from(dst.z)];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += (s[i] - mean_source[i]) * (t[j] - mean_target[j]);
            }
        }
    }
    let scale = h.iter().flatten().fold(0.0_f64, |largest, v| largest.max(v.abs()));
    let rotation = if scale == 0.0 {
        Quat::<f32>::identity()
    } else {
        for row in &mut h {
            for v in row {
                *v /= scale;
            }
        }
        let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = h;
        // Quaternion coordinates are [w, x, y, z]. The largest eigenvector
        // maximizes trace(R H) and always represents a proper rotation.
        let n = [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ];
        let q = largest_eigenvector(n)?;
        Quat::new(q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32).normalize()
    };
    // Use the actual returned f32 rotation for the centroid correction.
    let r = rotation.to_mat3();
    let mut t = [0.0_f32; 3];
    for i in 0..3 {
        t[i] = (mean_target[i] - (0..3).map(|j| f64::from(r.m[i][j]) * mean_source[j]).sum::<f64>())
            as f32;
        if !t[i].is_finite() {
            return None;
        }
    }
    Some(Isometry3::new(rotation, Vec3::new(t[0], t[1], t[2])))
}

fn largest_eigenvector(mut a: [[f64; 4]; 4]) -> Option<[f64; 4]> {
    let mut vectors = [[0.0_f64; 4]; 4];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let (mut p, mut q, mut largest) = (0, 1, 0.0_f64);
        for (i, row) in a.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > largest {
                    p = i;
                    q = j;
                    largest = value.abs();
                }
            }
        }
        let diagonal_scale = (0..4).fold(0.0_f64, |s, i| s.max(a[i][i].abs()));
        if largest <= 16.0 * f64::EPSILON * diagonal_scale.max(1.0) {
            let mut index = 0;
            for i in 1..4 {
                if a[i][i] > a[index][index] {
                    index = i;
                }
            }
            return Some(std::array::from_fn(|i| vectors[i][index]));
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let tangent = 1.0_f64.copysign(tau) / (tau.abs() + tau.hypot(1.0));
        let cosine = 1.0 / tangent.hypot(1.0);
        let sine = tangent * cosine;
        a[p][p] -= tangent * a[p][q];
        a[q][q] += tangent * a[p][q];
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in [0, 1, 2, 3] {
            if k != p && k != q {
                let (kp, kq) = (a[k][p], a[k][q]);
                a[k][p] = cosine * kp - sine * kq;
                a[p][k] = a[k][p];
                a[k][q] = sine * kp + cosine * kq;
                a[q][k] = a[k][q];
            }
        }
        for row in &mut vectors {
            let (vp, vq) = (row[p], row[q]);
            row[p] = cosine * vp - sine * vq;
            row[q] = sine * vp + cosine * vq;
        }
    }
    None // Do not silently use an eigensystem that failed to converge.
}

#[cfg(test)]
mod tests {
    use super::estimate_rigid_transform;
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    #[test]
    fn recovers_pure_translation() {
        let source: Vec<Vec3<f32>> = (0..20).map(|i| Vec3::new(i as f32 * 0.1, 0.0, 0.0)).collect();
        let offset = Vec3::new(0.5, -0.2, 0.1);
        let target: Vec<Vec3<f32>> = source.iter().map(|point| *point + offset).collect();

        let transform = estimate_rigid_transform(&source, &target).unwrap();
        assert!((transform.translation().x - offset.x).abs() < 1e-4);
        assert!((transform.translation().y - offset.y).abs() < 1e-4);
        assert!((transform.translation().z - offset.z).abs() < 1e-4);
    }

    #[test]
    fn recovers_known_rigid_transform() {
        let target: Vec<Vec3<f32>> = (0..4)
            .flat_map(|x| {
                (0..4).flat_map(move |y| {
                    (0..3).map(move |z| Vec3::new(x as f32, y as f32, z as f32 * 0.2))
                })
            })
            .collect();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.2),
            Vec3::new(0.3, -0.1, 0.05),
        );
        let source: Vec<Vec3<f32>> =
            target.iter().map(|point| misalignment.transform_point(*point)).collect();

        let estimated = estimate_rigid_transform(&source, &target).unwrap();
        let composed = estimated.compose(misalignment);
        let probe = Vec3::new(1.0, 2.0, 0.0);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 1e-3);
        assert!((restored.y - probe.y).abs() < 1e-3);
        assert!((restored.z - probe.z).abs() < 1e-3);
    }

    #[test]
    fn rigid_estimation_is_scale_invariant_for_volume_and_plane() {
        for scale in [1e-4_f32, 1.0, 1e4] {
            for planar in [false, true] {
                let source: Vec<_> = (0..4)
                    .flat_map(|x| {
                        (0..5).map(move |y| {
                            Vec3::new(
                                (x as f32 - 1.5) * scale,
                                (y as f32 - 2.0) * scale,
                                if planar { 0.0 } else { ((x * y % 3) as f32 - 1.0) * scale },
                            )
                        })
                    })
                    .collect();
                let truth = Isometry3::new(
                    Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.45),
                    Vec3::new(0.1 * scale, -0.2 * scale, 0.3 * scale),
                );
                let target: Vec<_> = source.iter().map(|p| truth.transform_point(*p)).collect();
                let estimate = estimate_rigid_transform(&source, &target).unwrap();
                for (p, expected) in source.iter().zip(&target) {
                    let restored = estimate.transform_point(*p);
                    let error = restored - *expected;
                    assert!(
                        error.x.hypot(error.y).hypot(error.z) / scale < 1e-5,
                        "scale={scale}, planar={planar}, estimated={estimate:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn many_points_at_large_origin_retain_translation_precision() {
        let source: Vec<_> = (0..20000)
            .map(|i| {
                Vec3::new(
                    10000.0 + ((i % 11) as f32 - 5.0) * 0.25,
                    -20000.0 + ((i % 7) as f32 - 3.0) * 0.5,
                    30000.0 + ((i % 5) as f32 - 2.0) * 0.25,
                )
            })
            .collect();
        let shift = Vec3::new(0.125, -0.25, 0.5);
        let target: Vec<_> = source.iter().map(|p| *p + shift).collect();
        let estimated = estimate_rigid_transform(&source, &target).unwrap();
        let t = estimated.translation();
        assert!((t.x - shift.x).abs() < 1e-5);
        assert!((t.y - shift.y).abs() < 1e-5);
        assert!((t.z - shift.z).abs() < 1e-5);
        for (source, target) in source.iter().zip(&target) {
            assert_eq!(estimated.transform_point(*source), *target);
        }
    }

    #[test]
    fn rejects_nonfinite_and_unrepresentable_results() {
        let source = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = source;
            invalid[1].z = bad;
            assert!(estimate_rigid_transform(&invalid, &source).is_none());
            assert!(estimate_rigid_transform(&source, &invalid).is_none());
        }
        assert!(estimate_rigid_transform(&source[..2], &source[..2]).is_none());
        assert!(estimate_rigid_transform(&source[..2], &source).is_none());
        let negative = [Vec3::new(-f32::MAX, 0.0, 0.0); 3];
        let positive = [Vec3::new(f32::MAX, 0.0, 0.0); 3];
        assert!(estimate_rigid_transform(&negative, &positive).is_none());
    }

    #[test]
    fn returns_proper_rotation_for_reflection_and_half_turn() {
        let source = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(-1.0, -2.0, -3.0),
        ];
        let reflected: Vec<_> = source.iter().map(|p| Vec3::new(-p.x, p.y, p.z)).collect();
        let estimated = estimate_rigid_transform(&source, &reflected).unwrap();
        let r = estimated.rotation().to_mat3().m;
        let determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((determinant - 1.0).abs() < 1e-5);
        let truth = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), std::f32::consts::PI),
            Vec3::new(0.0, 0.0, 0.0),
        );
        let target: Vec<_> = source.iter().map(|p| truth.transform_point(*p)).collect();
        let estimated = estimate_rigid_transform(&source, &target).unwrap();
        for (p, expected) in source.iter().zip(target) {
            let error = estimated.transform_point(*p) - expected;
            assert!(error.x.hypot(error.y).hypot(error.z) < 1e-5);
        }
    }
}

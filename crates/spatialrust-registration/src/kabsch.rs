use spatialrust_math::{Isometry3, Mat3, Quat, Vec3, symmetric_eigen3};

/// Estimates the rigid transform that best maps `source` onto `target`.
#[must_use]
pub fn estimate_rigid_transform(source: &[Vec3<f32>], target: &[Vec3<f32>]) -> Option<Isometry3<f32>> {
    if source.len() != target.len() || source.len() < 3 {
        return None;
    }

    let count = source.len() as f64;
    let mut mean_source = Vec3::new(0.0_f32, 0.0, 0.0);
    let mut mean_target = Vec3::new(0.0_f32, 0.0, 0.0);
    for (src, dst) in source.iter().zip(target) {
        mean_source = mean_source + *src;
        mean_target = mean_target + *dst;
    }
    mean_source = scale_vec3(mean_source, 1.0 / count as f32);
    mean_target = scale_vec3(mean_target, 1.0 / count as f32);

    let mut h = Mat3::<f64>::from_rows([0.0; 3], [0.0; 3], [0.0; 3]);
    for (src, dst) in source.iter().zip(target) {
        let ps = subtract(*src, mean_source);
        let pt = subtract(*dst, mean_target);
        h.m[0][0] += f64::from(ps.x) * f64::from(pt.x);
        h.m[0][1] += f64::from(ps.x) * f64::from(pt.y);
        h.m[0][2] += f64::from(ps.x) * f64::from(pt.z);
        h.m[1][0] += f64::from(ps.y) * f64::from(pt.x);
        h.m[1][1] += f64::from(ps.y) * f64::from(pt.y);
        h.m[1][2] += f64::from(ps.y) * f64::from(pt.z);
        h.m[2][0] += f64::from(ps.z) * f64::from(pt.x);
        h.m[2][1] += f64::from(ps.z) * f64::from(pt.y);
        h.m[2][2] += f64::from(ps.z) * f64::from(pt.z);
    }

    let mut rotation_matrix = rotation_from_cross_covariance(h);
    if mat3_det_f64(rotation_matrix) < 0.0 {
        let mut v = eigenvectors_from_cross_covariance(h);
        for row in 0..3 {
            v.m[row][2] = -v.m[row][2];
        }
        rotation_matrix = rotation_from_eigenvectors(h, v);
    }

    let rotation_f32 = mat3_f64_to_f32(rotation_matrix);
    let rotated_mean = rotation_f32.mul_vec3(mean_source);
    let translation = subtract(mean_target, rotated_mean);
    Some(Isometry3::new(quat_from_mat3(rotation_f32), translation))
}

fn rotation_from_cross_covariance(h: Mat3<f64>) -> Mat3<f64> {
    rotation_from_eigenvectors(h, eigenvectors_from_cross_covariance(h))
}

fn rotation_from_eigenvectors(h: Mat3<f64>, v: Mat3<f64>) -> Mat3<f64> {
    let eigen = symmetric_eigen3(mul_mat3_f64(h.transpose(), h));
    let hv = mul_mat3_f64(h, v);
    let mut u = Mat3::<f64>::identity();
    for column in 0..3 {
        let scale = inv_sqrt(eigen.eigenvalues[column]);
        for row in 0..3 {
            u.m[row][column] = hv.m[row][column] * scale;
        }
    }
    mul_mat3_f64(v, u.transpose())
}

fn eigenvectors_from_cross_covariance(h: Mat3<f64>) -> Mat3<f64> {
    symmetric_eigen3(mul_mat3_f64(h.transpose(), h)).eigenvectors
}

fn inv_sqrt(value: f64) -> f64 {
    if value > 1e-12 { 1.0 / value.sqrt() } else { 0.0 }
}

fn subtract(a: Vec3<f32>, b: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn scale_vec3(v: Vec3<f32>, scale: f32) -> Vec3<f32> {
    Vec3::new(v.x * scale, v.y * scale, v.z * scale)
}

fn mul_mat3_f64(a: Mat3<f64>, b: Mat3<f64>) -> Mat3<f64> {
    Mat3::from_rows(
        [
            a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0],
            a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1],
            a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2],
        ],
        [
            a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0],
            a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1],
            a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2],
        ],
        [
            a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0],
            a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1],
            a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2],
        ],
    )
}

fn mat3_det_f64(m: Mat3<f64>) -> f64 {
    m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])
        - m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])
        + m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0])
}

fn mat3_f64_to_f32(m: Mat3<f64>) -> Mat3<f32> {
    Mat3::from_rows(
        [m.m[0][0] as f32, m.m[0][1] as f32, m.m[0][2] as f32],
        [m.m[1][0] as f32, m.m[1][1] as f32, m.m[1][2] as f32],
        [m.m[2][0] as f32, m.m[2][1] as f32, m.m[2][2] as f32],
    )
}

fn quat_from_mat3(m: Mat3<f32>) -> Quat<f32> {
    let trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::new(
            (m.m[2][1] - m.m[1][2]) / s,
            (m.m[0][2] - m.m[2][0]) / s,
            (m.m[1][0] - m.m[0][1]) / s,
            0.25 * s,
        )
        .normalize()
    } else if m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2] {
        let s = (1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]).sqrt() * 2.0;
        Quat::new(
            0.25 * s,
            (m.m[0][1] + m.m[1][0]) / s,
            (m.m[0][2] + m.m[2][0]) / s,
            (m.m[2][1] - m.m[1][2]) / s,
        )
        .normalize()
    } else if m.m[1][1] > m.m[2][2] {
        let s = (1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]).sqrt() * 2.0;
        Quat::new(
            (m.m[0][1] + m.m[1][0]) / s,
            0.25 * s,
            (m.m[1][2] + m.m[2][1]) / s,
            (m.m[0][2] - m.m[2][0]) / s,
        )
        .normalize()
    } else {
        let s = (1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]).sqrt() * 2.0;
        Quat::new(
            (m.m[0][2] + m.m[2][0]) / s,
            (m.m[1][2] + m.m[2][1]) / s,
            0.25 * s,
            (m.m[1][0] - m.m[0][1]) / s,
        )
        .normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::estimate_rigid_transform;
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    #[test]
    fn recovers_pure_translation() {
        let source: Vec<Vec3<f32>> = (0..20)
            .map(|i| Vec3::new(i as f32 * 0.1, 0.0, 0.0))
            .collect();
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
            .flat_map(|x| (0..4).flat_map(move |y| (0..3).map(move |z| Vec3::new(x as f32, y as f32, z as f32 * 0.2))))
            .collect();
        let misalignment = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.2),
            Vec3::new(0.3, -0.1, 0.05),
        );
        let source: Vec<Vec3<f32>> = target
            .iter()
            .map(|point| misalignment.transform_point(*point))
            .collect();

        let estimated = estimate_rigid_transform(&source, &target).unwrap();
        let composed = estimated.compose(misalignment);
        let probe = Vec3::new(1.0, 2.0, 0.0);
        let restored = composed.transform_point(probe);
        assert!((restored.x - probe.x).abs() < 1e-3);
        assert!((restored.y - probe.y).abs() < 1e-3);
        assert!((restored.z - probe.z).abs() < 1e-3);
    }
}

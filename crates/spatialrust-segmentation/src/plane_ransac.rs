//! Shared RANSAC plane helpers for CPU and GPU segmenters.

use spatialrust_math::{symmetric_eigen3, Mat3, Vec3};

use crate::plane::PlaneModel;

pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        let high = self.next_u64() >> 32;
        ((high * upper as u64) >> 32) as usize
    }
}

pub(crate) fn sample_indices(rng: &mut Rng, len: usize) -> Option<[usize; 3]> {
    if len < 3 {
        return None;
    }

    let mut indices = [0usize; 3];
    indices[0] = rng.next_usize(len);
    indices[1] = rng.next_usize(len);
    while indices[1] == indices[0] {
        indices[1] = rng.next_usize(len);
    }
    indices[2] = rng.next_usize(len);
    while indices[2] == indices[0] || indices[2] == indices[1] {
        indices[2] = rng.next_usize(len);
    }
    Some(indices)
}

#[cfg(feature = "segment-ransac-plane-gpu")]
pub(crate) fn generate_hypotheses(len: usize, max_iterations: usize, seed: u64) -> Vec<[usize; 3]> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(max_iterations);
    for _ in 0..max_iterations {
        if let Some(sample) = sample_indices(&mut rng, len) {
            out.push(sample);
        }
    }
    out
}

pub(crate) fn plane_from_indices(x: &[f32], y: &[f32], z: &[f32], indices: [usize; 3]) -> Option<PlaneModel> {
    let points = [
        Vec3::new(x[indices[0]], y[indices[0]], z[indices[0]]),
        Vec3::new(x[indices[1]], y[indices[1]], z[indices[1]]),
        Vec3::new(x[indices[2]], y[indices[2]], z[indices[2]]),
    ];
    plane_from_points(points[0], points[1], points[2])
}

pub(crate) fn plane_from_points(p0: Vec3<f32>, p1: Vec3<f32>, p2: Vec3<f32>) -> Option<PlaneModel> {
    let v1 = p1 - p0;
    let v2 = p2 - p0;
    let mut normal = v1.cross(v2);
    if normal.length_squared() < 1e-12 {
        return None;
    }
    normal = normal.normalize();
    let d = -normal.dot(p0);
    Some(PlaneModel { normal, d })
}

pub(crate) fn collect_inliers(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    model: &PlaneModel,
    threshold: f32,
) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter_map(|(index, &px)| {
            (model.distance_xyz(px, y[index], z[index]) <= threshold).then_some(index)
        })
        .collect()
}

pub(crate) fn refine_plane_from_inliers(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    inliers: &[usize],
) -> Option<PlaneModel> {
    if inliers.len() < 3 {
        return None;
    }

    let count = inliers.len() as f64;
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut mean_z = 0.0_f64;
    for &index in inliers {
        mean_x += f64::from(x[index]);
        mean_y += f64::from(y[index]);
        mean_z += f64::from(z[index]);
    }
    mean_x /= count;
    mean_y /= count;
    mean_z /= count;

    let mut c00 = 0.0_f64;
    let mut c11 = 0.0_f64;
    let mut c22 = 0.0_f64;
    let mut c01 = 0.0_f64;
    let mut c02 = 0.0_f64;
    let mut c12 = 0.0_f64;
    for &index in inliers {
        let dx = f64::from(x[index]) - mean_x;
        let dy = f64::from(y[index]) - mean_y;
        let dz = f64::from(z[index]) - mean_z;
        c00 += dx * dx;
        c11 += dy * dy;
        c22 += dz * dz;
        c01 += dx * dy;
        c02 += dx * dz;
        c12 += dy * dz;
    }
    let inv = 1.0 / count;
    let covariance = Mat3::<f64>::from_rows(
        [c00 * inv, c01 * inv, c02 * inv],
        [c01 * inv, c11 * inv, c12 * inv],
        [c02 * inv, c12 * inv, c22 * inv],
    );

    let eigen = symmetric_eigen3(covariance);
    let normal = Vec3::new(
        eigen.eigenvectors.m[0][0] as f32,
        eigen.eigenvectors.m[1][0] as f32,
        eigen.eigenvectors.m[2][0] as f32,
    )
    .normalize();
    let centroid = Vec3::new(mean_x as f32, mean_y as f32, mean_z as f32);
    let d = -normal.dot(centroid);
    Some(PlaneModel { normal, d })
}

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::{symmetric_eigen3, Mat3, Vec3};

use crate::cloud::extract_mask;
use crate::segmenter::PointCloudSegmenter;

/// Plane model in Hessian form: `normal · p + d = 0` with unit normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneModel {
    /// Unit-length plane normal.
    pub normal: Vec3<f32>,
    /// Plane offset term.
    pub d: f32,
}

impl PlaneModel {
    /// Returns the signed distance from a point to the plane.
    #[must_use]
    pub fn signed_distance(&self, point: Vec3<f32>) -> f32 {
        self.normal.dot(point) + self.d
    }

    /// Returns the absolute distance from a point to the plane.
    #[must_use]
    pub fn distance(&self, point: Vec3<f32>) -> f32 {
        self.signed_distance(point).abs()
    }

    /// Returns the absolute distance from XYZ coordinates to the plane.
    #[must_use]
    pub fn distance_xyz(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.normal.x * x + self.normal.y * y + self.normal.z * z + self.d).abs()
    }
}

/// Configuration for RANSAC plane segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacPlaneConfig {
    /// Maximum distance from the plane for inlier classification.
    pub distance_threshold: f32,
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Minimum number of inliers required to accept a model.
    pub min_inliers: usize,
    /// Seed for deterministic sampling in tests.
    pub seed: u64,
}

impl Default for RansacPlaneConfig {
    fn default() -> Self {
        Self { distance_threshold: 0.01, max_iterations: 1_000, min_inliers: 3, seed: 42 }
    }
}

impl RansacPlaneConfig {
    /// Creates a config with the given distance threshold.
    #[must_use]
    pub const fn with_distance_threshold(distance_threshold: f32) -> Self {
        Self { distance_threshold, max_iterations: 1_000, min_inliers: 3, seed: 42 }
    }
}

/// Result of RANSAC plane segmentation.
#[derive(Clone, Debug, PartialEq)]
pub struct RansacPlaneSegmentation {
    /// Fitted plane model refined from inliers.
    pub model: PlaneModel,
    /// Points classified as inliers.
    pub inliers: PointCloud,
    /// Points classified as outliers.
    pub outliers: PointCloud,
    /// Number of inlier points.
    pub inlier_count: usize,
}

/// RANSAC-based dominant plane segmenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacPlaneSegmenter {
    config: RansacPlaneConfig,
}

impl RansacPlaneSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: RansacPlaneConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> RansacPlaneConfig {
        self.config
    }

    /// Segments the dominant plane and returns inlier/outlier clouds.
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<RansacPlaneSegmentation> {
        if input.is_empty() {
            return Err(SpatialError::InvalidArgument(
                "cannot segment plane from empty point cloud".to_owned(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let len = input.len();
        if len < 3 {
            return Err(SpatialError::InvalidArgument(
                "plane segmentation requires at least three points".to_owned(),
            ));
        }

        let mut rng = Rng::new(self.config.seed);
        let mut best_inliers = Vec::new();
        let mut best_model = None;

        for _ in 0..self.config.max_iterations {
            let Some(sample) = sample_indices(&mut rng, len) else {
                continue;
            };
            let Some(candidate) = plane_from_indices(x, y, z, sample) else {
                continue;
            };

            let inliers = collect_inliers(x, y, z, &candidate, self.config.distance_threshold);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_model = Some(candidate);
            }
        }

        if best_inliers.len() < self.config.min_inliers {
            return Err(SpatialError::InvalidArgument(format!(
                "RANSAC found only {} inliers, minimum is {}",
                best_inliers.len(),
                self.config.min_inliers
            )));
        }

        let model =
            refine_plane_from_inliers(x, y, z, &best_inliers).or(best_model).ok_or_else(|| {
                SpatialError::InvalidArgument("failed to refine plane model".to_owned())
            })?;

        let mut inlier_mask = vec![false; len];
        for index in &best_inliers {
            inlier_mask[*index] = true;
        }
        let mut outlier_mask = inlier_mask.clone();
        for selected in &mut outlier_mask {
            *selected = !*selected;
        }

        let inliers = extract_mask(input, &inlier_mask)?;
        let outliers = extract_mask(input, &outlier_mask)?;

        Ok(RansacPlaneSegmentation { inlier_count: best_inliers.len(), model, inliers, outliers })
    }

    /// Returns only the outlier cloud after removing the dominant plane.
    pub fn extract_outliers(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        self.segment(input).map(|result| result.outliers)
    }
}

impl PointCloudSegmenter for RansacPlaneSegmenter {
    fn name(&self) -> &'static str {
        "RansacPlaneSegmenter"
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        // Use the high, well-mixed bits of the LCG (its low bits have a short
        // period) and map them into `0..upper` with a multiply-shift, which
        // keeps the sampling uniform enough for RANSAC.
        let high = self.next_u64() >> 32;
        ((high * upper as u64) >> 32) as usize
    }
}

fn sample_indices(rng: &mut Rng, len: usize) -> Option<[usize; 3]> {
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

fn plane_from_indices(x: &[f32], y: &[f32], z: &[f32], indices: [usize; 3]) -> Option<PlaneModel> {
    let points = [
        Vec3::new(x[indices[0]], y[indices[0]], z[indices[0]]),
        Vec3::new(x[indices[1]], y[indices[1]], z[indices[1]]),
        Vec3::new(x[indices[2]], y[indices[2]], z[indices[2]]),
    ];
    plane_from_points(points[0], points[1], points[2])
}

fn plane_from_points(p0: Vec3<f32>, p1: Vec3<f32>, p2: Vec3<f32>) -> Option<PlaneModel> {
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

fn collect_inliers(
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

fn refine_plane_from_inliers(
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

#[cfg(test)]
mod tests {
    use super::{PlaneModel, RansacPlaneConfig, RansacPlaneSegmenter};
    use spatialrust_core::{HasPositions3, PointCloudBuilder, StandardSchemas};
    use spatialrust_math::Vec3;

    fn plane_with_outliers() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..10 {
            for y in 0..10 {
                builder.push_point([x as f32, y as f32, 0.0]).unwrap();
            }
        }
        builder.push_point([0.0, 0.0, 5.0]).unwrap();
        builder.push_point([1.0, 1.0, 5.0]).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn segments_dominant_plane() {
        let input = plane_with_outliers();
        let segmenter = RansacPlaneSegmenter::new(RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 50,
            seed: 7,
        });
        let result = segmenter.segment(&input).unwrap();
        assert_eq!(result.inlier_count, 100);
        assert_eq!(result.outliers.len(), 2);
        assert!(result.model.normal.z.abs() > 0.9);
    }

    #[test]
    fn plane_distance_matches_point() {
        let model = PlaneModel { normal: Vec3::new(0.0, 0.0, 1.0), d: 0.0 };
        assert!((model.distance(Vec3::new(0.0, 0.0, 1.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn extract_outliers_removes_plane() {
        let input = plane_with_outliers();
        let segmenter = RansacPlaneSegmenter::new(RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 50,
            seed: 7,
        });
        let outliers = segmenter.extract_outliers(&input).unwrap();
        let (_, _, z) = outliers.positions3().unwrap();
        assert!(z.iter().all(|value| *value > 1.0));
    }
}

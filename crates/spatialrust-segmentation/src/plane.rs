#[cfg(feature = "segment-ransac-plane-gpu")]
use spatialrust_core::DeviceKind;
use spatialrust_core::{ExecutionPolicy, HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_math::Vec3;

use crate::cloud::extract_mask;
use crate::plane_ransac::{
    collect_inliers, plane_from_indices, refine_plane_from_inliers, sample_indices, Rng,
};
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

/// Minimum point count before GPU RANSAC plane scoring is selected under `Auto`.
///
/// Full-cloud bench on the public PCL `table_scene_lms400` sample (460k points,
/// 1000 iterations) showed ~11× GPU speedup. After MVP-style voxel downsampling
/// (leaf=0.05) + normals the same scene is ~2k points and GPU remains ~2.7×
/// faster in local release measurements.
pub const DEFAULT_GPU_MIN_POINTS_PLANE: usize = 2_000;

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
    /// Minimum input point count before GPU execution is considered under `Auto`.
    ///
    /// `None` always uses GPU when requested.
    pub gpu_min_points: Option<usize>,
}

impl Default for RansacPlaneConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.01,
            max_iterations: 1_000,
            min_inliers: 3,
            seed: 42,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_PLANE),
        }
    }
}

impl RansacPlaneConfig {
    /// Creates a config with the given distance threshold.
    #[must_use]
    pub const fn with_distance_threshold(distance_threshold: f32) -> Self {
        Self {
            distance_threshold,
            max_iterations: 1_000,
            min_inliers: 3,
            seed: 42,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_PLANE),
        }
    }

    /// Disables the GPU point-count heuristic so GPU is always used when requested.
    #[must_use]
    pub const fn without_gpu_min_points(mut self) -> Self {
        self.gpu_min_points = None;
        self
    }

    /// Returns the point-count threshold used by [`ExecutionPolicy::Auto`].
    #[must_use]
    pub const fn effective_gpu_min_points(&self) -> Option<usize> {
        self.gpu_min_points
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
        self.segment_with_policy(input, ExecutionPolicy::CpuSingle)
    }

    /// Segments the dominant plane using the given execution policy.
    ///
    /// With the `segment-ransac-plane-gpu` feature, [`ExecutionPolicy::Auto`] and
    /// [`ExecutionPolicy::Gpu`] run hypothesis scoring on wgpu when the input meets
    /// [`RansacPlaneConfig::effective_gpu_min_points`].
    pub fn segment_with_policy(
        &self,
        input: &PointCloud,
        policy: ExecutionPolicy,
    ) -> SpatialResult<RansacPlaneSegmentation> {
        #[cfg(feature = "segment-ransac-plane-gpu")]
        {
            let resolved = self.resolve_policy(input, policy);
            if matches!(resolved, ExecutionPolicy::Gpu(DeviceKind::Wgpu)) {
                return crate::plane_gpu::GpuRansacPlaneSegmenter::new(self.config).segment(input);
            }
        }

        let _ = policy;
        self.segment_cpu(input)
    }

    #[cfg(feature = "segment-ransac-plane-gpu")]
    fn should_use_gpu(&self, input: &PointCloud) -> bool {
        self.config.effective_gpu_min_points().map_or(true, |min_points| input.len() >= min_points)
    }

    #[cfg(feature = "segment-ransac-plane-gpu")]
    fn resolve_policy(&self, input: &PointCloud, policy: ExecutionPolicy) -> ExecutionPolicy {
        match policy {
            ExecutionPolicy::Auto => {
                if self.should_use_gpu(input) {
                    ExecutionPolicy::Gpu(DeviceKind::Wgpu)
                } else {
                    ExecutionPolicy::CpuSingle
                }
            }
            ExecutionPolicy::Gpu(DeviceKind::Wgpu) if !self.should_use_gpu(input) => {
                ExecutionPolicy::CpuSingle
            }
            other => other,
        }
    }

    fn segment_cpu(&self, input: &PointCloud) -> SpatialResult<RansacPlaneSegmentation> {
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

        finalize_plane_segmentation(input, x, y, z, &self.config, best_inliers, best_model)
    }

    /// Returns only the outlier cloud after removing the dominant plane.
    pub fn extract_outliers(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        self.segment(input).map(|result| result.outliers)
    }
}

pub(crate) fn finalize_plane_segmentation(
    input: &PointCloud,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    config: &RansacPlaneConfig,
    best_inliers: Vec<usize>,
    best_model: Option<PlaneModel>,
) -> SpatialResult<RansacPlaneSegmentation> {
    if best_inliers.len() < config.min_inliers {
        return Err(SpatialError::InvalidArgument(format!(
            "RANSAC found only {} inliers, minimum is {}",
            best_inliers.len(),
            config.min_inliers
        )));
    }

    let model = refine_plane_from_inliers(x, y, z, &best_inliers)
        .or(best_model)
        .ok_or_else(|| SpatialError::InvalidArgument("failed to refine plane model".to_owned()))?;

    let len = input.len();
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

impl PointCloudSegmenter for RansacPlaneSegmenter {
    fn name(&self) -> &'static str {
        "RansacPlaneSegmenter"
    }
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
            ..Default::default()
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
            ..Default::default()
        });
        let outliers = segmenter.extract_outliers(&input).unwrap();
        let (_, _, z) = outliers.positions3().unwrap();
        assert!(z.iter().all(|value| *value > 1.0));
    }
}

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_gpu::{score_ransac_plane_hypotheses_gpu, WgpuRuntime};
use spatialrust_math::Vec3;

use crate::plane::{
    finalize_plane_segmentation, PlaneModel, RansacPlaneConfig, RansacPlaneSegmentation,
};
use crate::plane_ransac::{collect_inliers, generate_hypotheses};
use crate::segmenter::PointCloudSegmenter;

/// GPU-accelerated RANSAC plane segmenter.
///
/// Hypothesis scoring (inlier counting) runs on wgpu; refinement and cloud
/// extraction reuse the CPU helpers so results match [`crate::RansacPlaneSegmenter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRansacPlaneSegmenter {
    config: RansacPlaneConfig,
}

impl GpuRansacPlaneSegmenter {
    /// Creates a GPU segmenter from config.
    #[must_use]
    pub const fn new(config: RansacPlaneConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> RansacPlaneConfig {
        self.config
    }

    /// Segments the dominant plane using GPU hypothesis scoring.
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

        let runtime = WgpuRuntime::shared()?;
        let hypotheses_usize =
            generate_hypotheses(len, self.config.max_iterations, self.config.seed);
        let hypotheses_u32: Vec<[u32; 3]> = hypotheses_usize
            .iter()
            .map(|indices| [indices[0] as u32, indices[1] as u32, indices[2] as u32])
            .collect();

        let scores = score_ransac_plane_hypotheses_gpu(
            &runtime,
            x,
            y,
            z,
            &hypotheses_u32,
            self.config.distance_threshold,
        )?;

        let (best_index, best_count) = scores
            .iter()
            .enumerate()
            .map(|(index, score)| (index, score.inlier_count as usize))
            .max_by_key(|(_, count)| *count)
            .unwrap_or((0, 0));

        let best_score = &scores[best_index];
        let best_model = PlaneModel {
            normal: Vec3::new(best_score.normal[0], best_score.normal[1], best_score.normal[2]),
            d: best_score.d,
        };
        let best_inliers = if best_count > 0 {
            collect_inliers(x, y, z, &best_model, self.config.distance_threshold)
        } else {
            Vec::new()
        };

        finalize_plane_segmentation(
            input,
            x,
            y,
            z,
            &self.config,
            best_inliers,
            Some(best_model),
        )
    }
}

impl PointCloudSegmenter for GpuRansacPlaneSegmenter {
    fn name(&self) -> &'static str {
        "GpuRansacPlaneSegmenter"
    }
}

#[cfg(test)]
mod tests {
    use super::GpuRansacPlaneSegmenter;
    use crate::plane::{RansacPlaneConfig, RansacPlaneSegmenter};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_gpu::WgpuRuntime;

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
    fn gpu_matches_cpu_on_planar_patch() {
        if WgpuRuntime::shared().is_err() {
            return;
        }

        let input = plane_with_outliers();
        let config = RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 50,
            seed: 7,
            ..Default::default()
        };
        let cpu = RansacPlaneSegmenter::new(config).segment(&input).unwrap();
        let gpu = GpuRansacPlaneSegmenter::new(config).segment(&input).unwrap();
        assert_eq!(cpu.inlier_count, gpu.inlier_count);
        assert_eq!(cpu.outliers.len(), gpu.outliers.len());
        assert!((cpu.model.normal.z - gpu.model.normal.z).abs() < 1e-3);
    }
}

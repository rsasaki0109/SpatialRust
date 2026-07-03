//! MVP point cloud processing pipeline.
//!
//! Chains voxel downsampling, normal estimation, plane segmentation, clustering,
//! and optional ICP registration.

use spatialrust_core::{ExecutionPolicy, PointCloud, SpatialResult};
use spatialrust_features::{NormalEstimationConfig, NormalEstimator};
use spatialrust_filtering::{VoxelGridDownsample, VoxelGridDownsampleConfig};
use spatialrust_math::Isometry3;
use spatialrust_registration::{
    transform_point_cloud, GicpConfig, GicpRegistration, IcpConfig, IcpRegistration,
    PointCloudRegistration, PointToPlaneIcp, PointToPlaneIcpConfig, RegistrationResult,
};
use spatialrust_segmentation::{
    EuclideanClusterConfig, EuclideanClusterExtractor, EuclideanClusterResult, RansacPlaneConfig,
    RansacPlaneSegmentation, RansacPlaneSegmenter,
};

/// Registration backend used by the MVP pipeline's optional alignment step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MvpRegistrationMethod {
    /// Classic point-to-point ICP (target = downsampled cloud).
    #[default]
    PointToPoint,
    /// Point-to-plane ICP using the estimated normals (target = cloud with normals).
    PointToPlane,
    /// Generalized ICP (plane-to-plane, target = downsampled cloud).
    Gicp,
}

/// Configuration for optional ICP in the MVP pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct MvpIcpConfig {
    /// Shared registration settings (iterations, correspondence distance, thresholds).
    pub icp: IcpConfig,
    /// Optional transform applied to the reference cloud to synthesize a source scan.
    pub source_transform: Option<Isometry3<f32>>,
    /// Registration backend to use.
    pub method: MvpRegistrationMethod,
}

impl Default for MvpIcpConfig {
    fn default() -> Self {
        Self {
            icp: IcpConfig::default(),
            source_transform: None,
            method: MvpRegistrationMethod::PointToPoint,
        }
    }
}

/// Full configuration for the MVP pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct MvpPipelineConfig {
    /// Voxel downsampling settings.
    pub voxel: VoxelGridDownsampleConfig,
    /// Normal estimation settings.
    pub normals: NormalEstimationConfig,
    /// Dominant plane segmentation settings.
    pub plane: RansacPlaneConfig,
    /// Euclidean clustering settings for non-plane points.
    pub cluster: EuclideanClusterConfig,
    /// Optional ICP registration against the downsampled reference cloud.
    pub icp: Option<MvpIcpConfig>,
    /// Execution policy for the voxel downsampling stage.
    pub voxel_policy: ExecutionPolicy,
    /// Execution policy for the RANSAC plane segmentation stage.
    pub plane_policy: ExecutionPolicy,
    /// Execution policy for normal estimation.
    pub normal_policy: ExecutionPolicy,
    /// Optional multiplier for GPU normal grid radius (`leaf_size * scale`) under GPU Auto/Gpu.
    pub normal_gpu_radius_scale: f32,
    /// Execution policy for Euclidean clustering.
    pub cluster_policy: ExecutionPolicy,
}

impl Default for MvpPipelineConfig {
    fn default() -> Self {
        Self {
            voxel: VoxelGridDownsampleConfig::centroid(0.05),
            normals: NormalEstimationConfig::default(),
            plane: RansacPlaneConfig::default(),
            cluster: EuclideanClusterConfig::default(),
            icp: None,
            voxel_policy: ExecutionPolicy::Auto,
            plane_policy: ExecutionPolicy::Auto,
            normal_policy: ExecutionPolicy::Auto,
            normal_gpu_radius_scale: 2.0,
            cluster_policy: ExecutionPolicy::Auto,
        }
    }
}

impl MvpPipelineConfig {
    /// Creates a config with the given voxel leaf size.
    #[must_use]
    pub fn with_voxel_leaf_size(leaf_size: f32) -> Self {
        Self { voxel: VoxelGridDownsampleConfig::centroid(leaf_size), ..Self::default() }
    }
}

/// Output of a completed MVP pipeline run.
#[derive(Clone, Debug, PartialEq)]
pub struct MvpPipelineResult {
    /// Cloud after voxel downsampling.
    pub downsampled: PointCloud,
    /// Cloud with estimated normals and curvature.
    pub with_normals: PointCloud,
    /// Plane segmentation result.
    pub plane: RansacPlaneSegmentation,
    /// Clustering result on plane outliers.
    pub clusters: EuclideanClusterResult,
    /// Optional ICP registration result.
    pub registration: Option<RegistrationResult>,
    /// Primary pipeline output (labeled cluster cloud).
    pub output: PointCloud,
}

/// Builder-style MVP pipeline runner.
#[derive(Clone, Debug, PartialEq)]
pub struct MvpPipeline {
    config: MvpPipelineConfig,
}

impl MvpPipeline {
    /// Creates a pipeline from config.
    #[must_use]
    pub fn new(config: MvpPipelineConfig) -> Self {
        Self { config }
    }

    /// Returns the pipeline config.
    #[must_use]
    pub fn config(&self) -> &MvpPipelineConfig {
        &self.config
    }

    /// Runs the full MVP pipeline on the input cloud.
    pub fn run(&self, input: &PointCloud) -> SpatialResult<MvpPipelineResult> {
        let downsampled = VoxelGridDownsample::new(self.config.voxel)
            .filter_with_policy(input, self.config.voxel_policy)?;

        let normal_config = {
            #[cfg(feature = "pipeline-mvp-gpu")]
            {
                let mut config = self.config.normals;
                let estimator = NormalEstimator::new(config);
                if config.search_radius.is_none()
                    && estimator.selects_gpu_backend(&downsampled, self.config.normal_policy)
                {
                    let radius = (self.config.voxel.leaf_size
                        * self.config.normal_gpu_radius_scale)
                        .max(1e-4);
                    if estimator.gpu_grid_fits(&downsampled, radius) {
                        config.search_radius = Some(radius);
                    }
                }
                config
            }
            #[cfg(not(feature = "pipeline-mvp-gpu"))]
            {
                self.config.normals
            }
        };
        let with_normals = NormalEstimator::new(normal_config)
            .estimate_with_policy(&downsampled, self.config.normal_policy)?;

        let plane = RansacPlaneSegmenter::new(self.config.plane)
            .segment_with_policy(&with_normals, self.config.plane_policy)?;

        let clusters = EuclideanClusterExtractor::new(self.config.cluster)
            .extract_with_policy(&plane.outliers, self.config.cluster_policy)?;

        let registration = if let Some(icp_config) = &self.config.icp {
            // Point-to-plane aligns against the normal-bearing cloud; the others
            // use the plain downsampled cloud as the reference target.
            let target = match icp_config.method {
                MvpRegistrationMethod::PointToPlane => &with_normals,
                MvpRegistrationMethod::PointToPoint | MvpRegistrationMethod::Gicp => &downsampled,
            };
            let source = if let Some(transform) = icp_config.source_transform {
                transform_point_cloud(target, transform)?
            } else {
                target.clone()
            };
            let result = match icp_config.method {
                MvpRegistrationMethod::PointToPoint => {
                    IcpRegistration::new(icp_config.icp).align(&source, target)?
                }
                MvpRegistrationMethod::PointToPlane => {
                    PointToPlaneIcp::new(point_to_plane_config(&icp_config.icp))
                        .align(&source, target)?
                }
                MvpRegistrationMethod::Gicp => {
                    GicpRegistration::new(gicp_config(&icp_config.icp)).align(&source, target)?
                }
            };
            Some(result)
        } else {
            None
        };

        Ok(MvpPipelineResult {
            output: clusters.cloud.clone(),
            downsampled,
            with_normals,
            plane,
            clusters,
            registration,
        })
    }
}

/// Maps the shared ICP settings onto a point-to-plane configuration.
fn point_to_plane_config(icp: &IcpConfig) -> PointToPlaneIcpConfig {
    PointToPlaneIcpConfig {
        max_iterations: icp.max_iterations,
        max_correspondence_distance: icp.max_correspondence_distance,
        transformation_epsilon: icp.transformation_epsilon,
        fitness_epsilon: icp.fitness_epsilon,
        min_correspondences: icp.min_correspondences,
        initial_guess: icp.initial_guess,
    }
}

/// Maps the shared ICP settings onto a GICP configuration.
fn gicp_config(icp: &IcpConfig) -> GicpConfig {
    GicpConfig {
        max_iterations: icp.max_iterations,
        max_correspondence_distance: icp.max_correspondence_distance,
        transformation_epsilon: icp.transformation_epsilon,
        fitness_epsilon: icp.fitness_epsilon,
        min_correspondences: icp.min_correspondences,
        initial_guess: icp.initial_guess,
        ..GicpConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{MvpIcpConfig, MvpPipeline, MvpPipelineConfig};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};
    use spatialrust_math::{Isometry3, Quat, Vec3};
    use spatialrust_registration::IcpConfig;
    use spatialrust_segmentation::{EuclideanClusterConfig, RansacPlaneConfig};

    fn sample_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for x in 0..10 {
            for y in 0..10 {
                builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
            }
        }
        builder.push_point([0.0, 0.0, 0.5]).unwrap();
        builder.push_point([0.1, 0.0, 0.5]).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn runs_voxel_normals_plane_and_cluster() {
        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            voxel: spatialrust_filtering::VoxelGridDownsampleConfig::centroid(0.2),
            normals: spatialrust_features::NormalEstimationConfig {
                k_neighbors: 8,
                min_neighbors: 3,
                viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
                ..Default::default()
            },
            plane: RansacPlaneConfig {
                distance_threshold: 0.05,
                max_iterations: 500,
                min_inliers: 10,
                seed: 17,
                ..Default::default()
            },
            cluster: EuclideanClusterConfig {
                cluster_tolerance: 0.3,
                min_cluster_size: 1,
                max_cluster_size: usize::MAX,
                ..Default::default()
            },
            icp: None,
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(!result.downsampled.is_empty());
        assert!(result.with_normals.field("normal_x").is_ok());
        assert!(result.plane.inlier_count >= 10);
        assert!(result.clusters.cluster_count >= 1);
        assert!(result.output.field("label").is_ok());
        assert!(result.registration.is_none());
    }

    #[test]
    fn runs_optional_icp_step() {
        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            voxel: spatialrust_filtering::VoxelGridDownsampleConfig::centroid(0.2),
            normals: spatialrust_features::NormalEstimationConfig {
                k_neighbors: 8,
                min_neighbors: 3,
                viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
                ..Default::default()
            },
            plane: RansacPlaneConfig {
                distance_threshold: 0.05,
                max_iterations: 500,
                min_inliers: 10,
                seed: 17,
                ..Default::default()
            },
            cluster: EuclideanClusterConfig {
                cluster_tolerance: 0.3,
                min_cluster_size: 1,
                max_cluster_size: usize::MAX,
                ..Default::default()
            },
            icp: Some(MvpIcpConfig {
                icp: IcpConfig {
                    max_correspondence_distance: 0.2,
                    max_iterations: 30,
                    ..Default::default()
                },
                source_transform: Some(Isometry3::new(
                    Quat::<f32>::identity(),
                    Vec3::new(0.03, -0.01, 0.0),
                )),
                ..Default::default()
            }),
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        let registration = result.registration.expect("expected icp result");
        assert!(registration.converged);
    }

    /// Three perpendicular faces, giving point-to-plane/GICP full 6-DoF constraint.
    fn corner_cloud() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..12 {
            for j in 0..12 {
                let (a, b) = (i as f32 * 0.06, j as f32 * 0.06);
                builder.push_point([a, b, 0.0]).unwrap();
                builder.push_point([a, 0.0, b + 0.03]).unwrap();
                builder.push_point([0.0, a + 0.03, b + 0.03]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    fn run_with_method(method: super::MvpRegistrationMethod) -> super::RegistrationResult {
        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            voxel: spatialrust_filtering::VoxelGridDownsampleConfig::centroid(0.05),
            normals: spatialrust_features::NormalEstimationConfig {
                k_neighbors: 10,
                min_neighbors: 3,
                ..Default::default()
            },
            plane: RansacPlaneConfig {
                distance_threshold: 0.02,
                max_iterations: 500,
                min_inliers: 10,
                seed: 17,
                ..Default::default()
            },
            cluster: EuclideanClusterConfig {
                cluster_tolerance: 0.3,
                min_cluster_size: 1,
                max_cluster_size: usize::MAX,
                ..Default::default()
            },
            icp: Some(MvpIcpConfig {
                icp: IcpConfig {
                    max_correspondence_distance: 0.3,
                    max_iterations: 40,
                    min_correspondences: 6,
                    ..Default::default()
                },
                source_transform: Some(Isometry3::new(
                    Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.05),
                    Vec3::new(0.01, -0.008, 0.012),
                )),
                method,
            }),
            ..Default::default()
        });
        pipeline.run(&corner_cloud()).unwrap().registration.expect("expected registration")
    }

    #[test]
    fn runs_point_to_plane_registration() {
        let result = run_with_method(super::MvpRegistrationMethod::PointToPlane);
        assert!(result.fitness.is_finite());
        assert!(result.fitness < 1e-2, "fitness too high: {}", result.fitness);
    }

    #[test]
    fn runs_gicp_registration() {
        let result = run_with_method(super::MvpRegistrationMethod::Gicp);
        assert!(result.fitness.is_finite());
        assert!(result.fitness < 1e-1, "fitness too high: {}", result.fitness);
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    #[test]
    fn runs_with_gpu_voxel_policy() {
        use spatialrust_core::{DeviceKind, ExecutionPolicy};

        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            voxel: spatialrust_filtering::VoxelGridDownsampleConfig::centroid(0.2),
            voxel_policy: ExecutionPolicy::Gpu(DeviceKind::Wgpu),
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(!result.downsampled.is_empty());
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    #[test]
    fn runs_with_gpu_plane_policy() {
        use spatialrust_core::{DeviceKind, ExecutionPolicy};

        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            plane_policy: ExecutionPolicy::Gpu(DeviceKind::Wgpu),
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(result.plane.inlier_count >= 10);
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    #[test]
    fn runs_with_gpu_normal_policy() {
        use spatialrust_core::{DeviceKind, ExecutionPolicy};

        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            normal_policy: ExecutionPolicy::Gpu(DeviceKind::Wgpu),
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(result.with_normals.field("normal_x").is_ok());
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    #[test]
    fn auto_normal_policy_derives_gpu_radius_from_voxel_leaf() {
        use spatialrust_core::ExecutionPolicy;
        use spatialrust_features::NormalEstimationConfig;

        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            voxel: spatialrust_filtering::VoxelGridDownsampleConfig::centroid(0.05),
            normals: NormalEstimationConfig { search_radius: None, ..Default::default() },
            normal_policy: ExecutionPolicy::Auto,
            normal_gpu_radius_scale: 2.0,
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(result.with_normals.field("normal_x").is_ok());
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    #[test]
    fn runs_with_gpu_cluster_policy() {
        use spatialrust_core::{DeviceKind, ExecutionPolicy};

        let pipeline = MvpPipeline::new(MvpPipelineConfig {
            cluster_policy: ExecutionPolicy::Gpu(DeviceKind::Wgpu),
            ..Default::default()
        });

        let result = pipeline.run(&sample_cloud()).unwrap();
        assert!(result.clusters.cluster_count >= 1);
    }
}

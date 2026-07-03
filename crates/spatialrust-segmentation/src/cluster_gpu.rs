use spatialrust_core::{HasPositions3, PointCloud, SpatialResult};
use spatialrust_gpu::{euclidean_cluster_roots_gpu, WgpuRuntime};

use crate::cluster::{
    finalize_euclidean_clusters, EuclideanClusterConfig,
    EuclideanClusterResult,
};
use crate::segmenter::PointCloudSegmenter;

/// GPU-accelerated Euclidean cluster extractor.
///
/// Connected components are found with uniform-grid label propagation on wgpu;
/// cluster size filtering and label remapping reuse the CPU helpers so results
/// match [`EuclideanClusterExtractor`] partition semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuEuclideanClusterExtractor {
    config: EuclideanClusterConfig,
}

impl GpuEuclideanClusterExtractor {
    /// Creates a GPU extractor from config.
    #[must_use]
    pub const fn new(config: EuclideanClusterConfig) -> Self {
        Self { config }
    }

    /// Returns the extractor config.
    #[must_use]
    pub const fn config(&self) -> EuclideanClusterConfig {
        self.config
    }

    /// Clusters the input cloud on the GPU.
    pub fn extract(&self, input: &PointCloud) -> SpatialResult<EuclideanClusterResult> {
        if input.is_empty() {
            return Ok(EuclideanClusterResult {
                cloud: input.clone(),
                cluster_count: 0,
                cluster_sizes: Vec::new(),
            });
        }

        let (x, y, z) = input.positions3()?;
        let runtime = WgpuRuntime::shared()?;
        let roots = euclidean_cluster_roots_gpu(&runtime, x, y, z, self.config.cluster_tolerance)?;
        finalize_euclidean_clusters(input, &roots, self.config)
    }
}

impl PointCloudSegmenter for GpuEuclideanClusterExtractor {
    fn name(&self) -> &'static str {
        "GpuEuclideanClusterExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::GpuEuclideanClusterExtractor;
    use crate::cluster::EuclideanClusterConfig;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn three_clusters() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for center in [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)] {
            for dx in 0..3 {
                for dy in 0..3 {
                    builder
                        .push_point([center.0 + dx as f32, center.1 + dy as f32, center.2])
                        .unwrap();
                }
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn gpu_finds_three_separated_clusters() {
        let input = three_clusters();
        let extractor = GpuEuclideanClusterExtractor::new(EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
            ..Default::default()
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 3);
        assert!(result.cluster_sizes.iter().all(|&size| size == 9));
    }
}

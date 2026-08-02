use spatialrust_core::{HasPositions3, PointCloud, SpatialResult, TransferStats};
use spatialrust_gpu::{euclidean_cluster_roots_gpu_with_receipt, WgpuRuntime};

use crate::cluster::{finalize_euclidean_clusters, EuclideanClusterConfig, EuclideanClusterResult};
use crate::segmenter::PointCloudSegmenter;

/// GPU-grid-backed Euclidean cluster extractor.
///
/// Sparse grid key generation, sorting, and compaction run on wgpu. Connected
/// component labeling and cluster-size filtering remain deterministic host-side
/// stages, matching [`crate::EuclideanClusterExtractor`] semantics.
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

    /// Clusters the input cloud using the grid union-find backend.
    pub fn extract(&self, input: &PointCloud) -> SpatialResult<EuclideanClusterResult> {
        self.extract_with_receipt(input).map(|(result, _)| result)
    }

    /// Clusters the input cloud and returns GPU grid transfer accounting.
    pub fn extract_with_receipt(
        &self,
        input: &PointCloud,
    ) -> SpatialResult<(EuclideanClusterResult, TransferStats)> {
        if input.is_empty() {
            return Ok((
                EuclideanClusterResult {
                    cloud: input.clone(),
                    cluster_count: 0,
                    cluster_sizes: Vec::new(),
                },
                TransferStats::default(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let runtime = WgpuRuntime::shared()?;
        let (roots, transfers) = euclidean_cluster_roots_gpu_with_receipt(
            &runtime,
            x,
            y,
            z,
            self.config.cluster_tolerance,
        )?;
        Ok((finalize_euclidean_clusters(input, &roots, self.config)?, transfers))
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
            gpu_min_points: None,
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 3);
        assert!(result.cluster_sizes.iter().all(|&size| size == 9));
    }

    fn long_chain(len: usize, spacing: f32) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for index in 0..len {
            builder.push_point([index as f32 * spacing, 0.0, 0.0]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn gpu_matches_cpu_on_long_chain() {
        use crate::cluster::EuclideanClusterExtractor;

        let input = long_chain(300, 1.0);
        let config = EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        };
        let cpu = EuclideanClusterExtractor::new(config).extract(&input).unwrap();
        let gpu = GpuEuclideanClusterExtractor::new(config).extract(&input).unwrap();
        assert_eq!(cpu.cluster_count, 1);
        assert_eq!(gpu.cluster_count, cpu.cluster_count);
        assert_eq!(gpu.cluster_sizes, cpu.cluster_sizes);
    }

    #[test]
    fn gpu_matches_cpu_on_three_clusters() {
        use crate::cluster::EuclideanClusterExtractor;

        let input = three_clusters();
        let config = EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
            gpu_min_points: None,
        };
        let cpu = EuclideanClusterExtractor::new(config).extract(&input).unwrap();
        let gpu = GpuEuclideanClusterExtractor::new(config).extract(&input).unwrap();
        assert_eq!(gpu.cluster_count, cpu.cluster_count);
        assert_eq!(gpu.cluster_sizes, cpu.cluster_sizes);
    }
}

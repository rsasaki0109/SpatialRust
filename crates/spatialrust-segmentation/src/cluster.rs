use std::collections::VecDeque;

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_search::{KdTree, RadiusSearchIndex};

use crate::cloud::with_labels;
use crate::segmenter::PointCloudSegmenter;

/// Configuration for Euclidean clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EuclideanClusterConfig {
    /// Maximum distance between points in the same cluster.
    pub cluster_tolerance: f32,
    /// Minimum number of points required to form a cluster.
    pub min_cluster_size: usize,
    /// Maximum number of points allowed in a cluster.
    pub max_cluster_size: usize,
}

impl Default for EuclideanClusterConfig {
    fn default() -> Self {
        Self { cluster_tolerance: 0.02, min_cluster_size: 1, max_cluster_size: usize::MAX }
    }
}

impl EuclideanClusterConfig {
    /// Creates a config with the given tolerance and minimum cluster size.
    #[must_use]
    pub const fn with_tolerance(cluster_tolerance: f32, min_cluster_size: usize) -> Self {
        Self { cluster_tolerance, min_cluster_size, max_cluster_size: usize::MAX }
    }
}

/// Result of Euclidean clustering.
#[derive(Clone, Debug, PartialEq)]
pub struct EuclideanClusterResult {
    /// Input points annotated with cluster labels.
    pub cloud: PointCloud,
    /// Number of valid clusters found.
    pub cluster_count: usize,
    /// Size of each cluster in label order.
    pub cluster_sizes: Vec<usize>,
}

/// Euclidean region-growing cluster extractor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EuclideanClusterExtractor {
    config: EuclideanClusterConfig,
}

impl EuclideanClusterExtractor {
    /// Creates an extractor from config.
    #[must_use]
    pub const fn new(config: EuclideanClusterConfig) -> Self {
        Self { config }
    }

    /// Returns the extractor config.
    #[must_use]
    pub const fn config(&self) -> EuclideanClusterConfig {
        self.config
    }

    /// Clusters the input cloud and adds a `label` field.
    pub fn extract(&self, input: &PointCloud) -> SpatialResult<EuclideanClusterResult> {
        if input.is_empty() {
            return Ok(EuclideanClusterResult {
                cloud: input.clone(),
                cluster_count: 0,
                cluster_sizes: Vec::new(),
            });
        }
        if self.config.cluster_tolerance < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "cluster_tolerance must be non-negative".to_owned(),
            ));
        }
        if self.config.min_cluster_size == 0 {
            return Err(SpatialError::InvalidArgument(
                "min_cluster_size must be greater than zero".to_owned(),
            ));
        }
        if self.config.max_cluster_size < self.config.min_cluster_size {
            return Err(SpatialError::InvalidArgument(
                "max_cluster_size must be >= min_cluster_size".to_owned(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let tree = KdTree::from_slices(x, y, z);
        let len = input.len();
        let mut processed = vec![false; len];
        let mut labels = vec![-1_i32; len];
        let mut cluster_sizes = Vec::new();
        let mut cluster_id = 0_i32;

        for seed in 0..len {
            if processed[seed] {
                continue;
            }

            let mut queue = VecDeque::from([seed]);
            let mut cluster_indices = Vec::new();
            processed[seed] = true;

            while let Some(index) = queue.pop_front() {
                cluster_indices.push(index);
                let neighbors =
                    tree.radius_search(x[index], y[index], z[index], self.config.cluster_tolerance);
                for neighbor in neighbors {
                    let candidate = neighbor.index;
                    if processed[candidate] {
                        continue;
                    }
                    processed[candidate] = true;
                    queue.push_back(candidate);
                }
            }

            if cluster_indices.len() >= self.config.min_cluster_size
                && cluster_indices.len() <= self.config.max_cluster_size
            {
                let cluster_size = cluster_indices.len();
                for index in cluster_indices {
                    labels[index] = cluster_id;
                }
                cluster_sizes.push(cluster_size);
                cluster_id += 1;
            }
        }

        Ok(EuclideanClusterResult {
            cloud: with_labels(input, "label", labels)?,
            cluster_count: cluster_sizes.len(),
            cluster_sizes,
        })
    }
}

impl PointCloudSegmenter for EuclideanClusterExtractor {
    fn name(&self) -> &'static str {
        "EuclideanClusterExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::{EuclideanClusterConfig, EuclideanClusterExtractor};
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
    fn finds_three_separated_clusters() {
        let input = three_clusters();
        let extractor = EuclideanClusterExtractor::new(EuclideanClusterConfig {
            cluster_tolerance: 1.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 3);
        assert!(result.cluster_sizes.iter().all(|&size| size == 9));
        assert!(result.cloud.field("label").is_ok());
    }

    #[test]
    fn rejects_clusters_smaller_than_minimum() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([0.1, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();

        let extractor = EuclideanClusterExtractor::new(EuclideanClusterConfig {
            cluster_tolerance: 0.5,
            min_cluster_size: 3,
            max_cluster_size: usize::MAX,
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 0);
    }
}

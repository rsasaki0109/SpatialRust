use std::collections::VecDeque;

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_search::{KdTree, RadiusSearchIndex};

use crate::cloud::with_labels;
use crate::segmenter::PointCloudSegmenter;

/// Configuration for DBSCAN density-based clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DbscanConfig {
    /// Neighborhood radius (`eps`); points within this distance are connected.
    pub eps: f32,
    /// Minimum neighbors (including the point itself) for a point to be a core
    /// point. Core points seed and expand clusters; non-core points within a
    /// core point's neighborhood become border points, everything else is noise.
    pub min_points: usize,
}

impl Default for DbscanConfig {
    fn default() -> Self {
        Self { eps: 0.5, min_points: 10 }
    }
}

impl DbscanConfig {
    /// Creates a config from the neighborhood radius and core-point threshold.
    #[must_use]
    pub const fn new(eps: f32, min_points: usize) -> Self {
        Self { eps, min_points }
    }
}

/// Result of DBSCAN clustering.
#[derive(Clone, Debug, PartialEq)]
pub struct DbscanResult {
    /// Input points annotated with cluster labels (`label` field, `-1` = noise).
    pub cloud: PointCloud,
    /// Number of clusters found.
    pub cluster_count: usize,
    /// Size of each cluster in label order.
    pub cluster_sizes: Vec<usize>,
    /// Number of points classified as noise.
    pub noise_count: usize,
}

/// DBSCAN density-based segmenter.
///
/// Groups points into clusters of high density (at least `min_points` within
/// `eps`), expanding from core points through density-reachable neighbors, and
/// labels low-density points as noise (`-1`). Unlike Euclidean clustering it
/// separates touching clusters connected only through sparse bridges and is
/// robust to scattered outliers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DbscanSegmenter {
    config: DbscanConfig,
}

impl DbscanSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: DbscanConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> DbscanConfig {
        self.config
    }

    /// Clusters the input cloud and adds a `label` field (`-1` for noise).
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<DbscanResult> {
        if input.is_empty() {
            return Ok(DbscanResult {
                cloud: input.clone(),
                cluster_count: 0,
                cluster_sizes: Vec::new(),
                noise_count: 0,
            });
        }
        if self.config.eps <= 0.0 || self.config.eps.is_nan() {
            return Err(SpatialError::InvalidArgument("eps must be positive".to_owned()));
        }
        if self.config.min_points == 0 {
            return Err(SpatialError::InvalidArgument(
                "min_points must be greater than zero".to_owned(),
            ));
        }

        let (x, y, z) = input.positions3()?;
        let tree = KdTree::from_slices(x, y, z);
        let len = input.len();

        // -1 = noise/unvisited sentinel reused as the final label; `visited`
        // tracks which points have had their neighborhood expanded.
        let mut labels = vec![-1_i32; len];
        let mut visited = vec![false; len];
        let mut cluster_sizes = Vec::new();
        let mut cluster_id = 0_i32;

        for seed in 0..len {
            if visited[seed] {
                continue;
            }
            visited[seed] = true;

            let seed_neighbors = tree.radius_search(x[seed], y[seed], z[seed], self.config.eps);
            if seed_neighbors.len() < self.config.min_points {
                // Not a core point (yet); leave as noise. It may later be
                // claimed as a border point by a neighboring core point.
                continue;
            }

            // Start a new cluster and flood-fill density-reachable points.
            labels[seed] = cluster_id;
            let mut size = 1_usize;
            let mut queue: VecDeque<usize> =
                seed_neighbors.iter().map(|n| n.index).filter(|&i| i != seed).collect();

            while let Some(current) = queue.pop_front() {
                if labels[current] == -1 {
                    // Was noise; claim it as a border (or core) point.
                    labels[current] = cluster_id;
                    size += 1;
                }
                if visited[current] {
                    continue;
                }
                visited[current] = true;

                let neighbors =
                    tree.radius_search(x[current], y[current], z[current], self.config.eps);
                if neighbors.len() >= self.config.min_points {
                    // Core point: its neighbors are density-reachable too.
                    for neighbor in neighbors {
                        if !visited[neighbor.index] || labels[neighbor.index] == -1 {
                            queue.push_back(neighbor.index);
                        }
                    }
                }
            }

            cluster_sizes.push(size);
            cluster_id += 1;
        }

        let noise_count = labels.iter().filter(|&&l| l == -1).count();
        Ok(DbscanResult {
            cloud: with_labels(input, "label", labels)?,
            cluster_count: cluster_sizes.len(),
            cluster_sizes,
            noise_count,
        })
    }
}

impl PointCloudSegmenter for DbscanSegmenter {
    fn name(&self) -> &'static str {
        "DbscanSegmenter"
    }
}

#[cfg(test)]
mod tests {
    use super::{DbscanConfig, DbscanSegmenter};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    /// Two dense 3x3 blobs plus a lone noise point far from both.
    fn two_blobs_with_noise() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for center in [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] {
            for dx in 0..3 {
                for dy in 0..3 {
                    builder
                        .push_point([center.0 + dx as f32 * 0.2, center.1 + dy as f32 * 0.2, 0.0])
                        .unwrap();
                }
            }
        }
        builder.push_point([100.0, 100.0, 100.0]).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn finds_two_clusters_and_isolates_noise() {
        let input = two_blobs_with_noise();
        let result = DbscanSegmenter::new(DbscanConfig::new(0.5, 4)).segment(&input).unwrap();
        assert_eq!(result.cluster_count, 2);
        assert_eq!(result.noise_count, 1);
        assert!(result.cluster_sizes.iter().all(|&s| s == 9));
    }

    #[test]
    fn all_noise_when_min_points_too_high() {
        let input = two_blobs_with_noise();
        // 19 points but require 50 neighbors -> nothing is a core point.
        let result = DbscanSegmenter::new(DbscanConfig::new(0.5, 50)).segment(&input).unwrap();
        assert_eq!(result.cluster_count, 0);
        assert_eq!(result.noise_count, input.len());
    }

    #[test]
    fn invalid_params_error() {
        let input = two_blobs_with_noise();
        assert!(DbscanSegmenter::new(DbscanConfig::new(0.0, 4)).segment(&input).is_err());
        assert!(DbscanSegmenter::new(DbscanConfig::new(0.5, 0)).segment(&input).is_err());
    }
}

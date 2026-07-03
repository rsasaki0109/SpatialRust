use std::collections::{HashMap, VecDeque};

#[cfg(feature = "segment-euclidean-gpu")]
use spatialrust_core::DeviceKind;
use spatialrust_core::{ExecutionPolicy, HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_search::{KdTree, RadiusSearchIndex};

use crate::cloud::with_labels;
use crate::segmenter::PointCloudSegmenter;

/// Minimum point count before GPU Euclidean clustering is selected under `Auto`.
pub const DEFAULT_GPU_MIN_POINTS_EUCLIDEAN: usize = 2_000;

/// Configuration for Euclidean clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EuclideanClusterConfig {
    /// Maximum distance between points in the same cluster.
    pub cluster_tolerance: f32,
    /// Minimum number of points required to form a cluster.
    pub min_cluster_size: usize,
    /// Maximum number of points allowed in a cluster.
    pub max_cluster_size: usize,
    /// Minimum input point count before GPU execution is considered under `Auto`.
    ///
    /// `None` always uses GPU when requested.
    pub gpu_min_points: Option<usize>,
}

impl Default for EuclideanClusterConfig {
    fn default() -> Self {
        Self {
            cluster_tolerance: 0.02,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_EUCLIDEAN),
        }
    }
}

impl EuclideanClusterConfig {
    /// Creates a config with the given tolerance and minimum cluster size.
    #[must_use]
    pub const fn with_tolerance(cluster_tolerance: f32, min_cluster_size: usize) -> Self {
        Self {
            cluster_tolerance,
            min_cluster_size,
            max_cluster_size: usize::MAX,
            gpu_min_points: Some(DEFAULT_GPU_MIN_POINTS_EUCLIDEAN),
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
        validate_cluster_config(self.config)?;
        if input.is_empty() {
            return Ok(EuclideanClusterResult {
                cloud: input.clone(),
                cluster_count: 0,
                cluster_sizes: Vec::new(),
            });
        }
        let roots = extract_cpu_roots(input, self.config)?;
        finalize_euclidean_clusters(input, &roots, self.config)
    }

    /// Clusters the input cloud using the given execution policy.
    ///
    /// With the `segment-euclidean-gpu` feature, [`ExecutionPolicy::Auto`] and
    /// [`ExecutionPolicy::Gpu`] run connected-component labeling on wgpu when the
    /// input meets [`EuclideanClusterConfig::effective_gpu_min_points`].
    pub fn extract_with_policy(
        &self,
        input: &PointCloud,
        policy: ExecutionPolicy,
    ) -> SpatialResult<EuclideanClusterResult> {
        #[cfg(feature = "segment-euclidean-gpu")]
        {
            let resolved = self.resolve_policy(input, policy);
            if matches!(resolved, ExecutionPolicy::Gpu(DeviceKind::Wgpu))
                && self.gpu_grid_fits(input)
            {
                return crate::cluster_gpu::GpuEuclideanClusterExtractor::new(self.config)
                    .extract(input);
            }
        }

        let _ = policy;
        self.extract(input)
    }

    #[cfg(feature = "segment-euclidean-gpu")]
    fn gpu_grid_fits(&self, input: &PointCloud) -> bool {
        let Ok((x, y, z)) = input.positions3() else {
            return false;
        };
        spatialrust_gpu::uniform_grid_fits(x, y, z, self.config.cluster_tolerance)
    }

    #[cfg(feature = "segment-euclidean-gpu")]
    fn should_use_gpu(&self, input: &PointCloud) -> bool {
        self.config.effective_gpu_min_points().map_or(true, |min_points| input.len() >= min_points)
    }

    #[cfg(feature = "segment-euclidean-gpu")]
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
}

impl PointCloudSegmenter for EuclideanClusterExtractor {
    fn name(&self) -> &'static str {
        "EuclideanClusterExtractor"
    }
}

/// Builds labeled output from per-point connected-component roots.
pub(crate) fn finalize_euclidean_clusters(
    input: &PointCloud,
    component_roots: &[u32],
    config: EuclideanClusterConfig,
) -> SpatialResult<EuclideanClusterResult> {
    if component_roots.len() != input.len() {
        return Err(SpatialError::InvalidArgument(
            "component root labels must match point count".to_owned(),
        ));
    }

    let mut sizes: HashMap<u32, usize> = HashMap::new();
    for &root in component_roots {
        *sizes.entry(root).or_insert(0) += 1;
    }

    let mut valid_roots: Vec<u32> = sizes
        .iter()
        .filter(|(_, &size)| size >= config.min_cluster_size && size <= config.max_cluster_size)
        .map(|(&root, _)| root)
        .collect();
    valid_roots.sort_unstable();

    let mut remap: HashMap<u32, i32> = HashMap::new();
    for (cluster_id, root) in valid_roots.iter().enumerate() {
        remap.insert(*root, cluster_id as i32);
    }

    let mut labels = vec![-1_i32; input.len()];
    let mut cluster_sizes = Vec::with_capacity(valid_roots.len());
    for root in &valid_roots {
        cluster_sizes.push(sizes[root]);
        for (index, &point_root) in component_roots.iter().enumerate() {
            if point_root == *root {
                labels[index] = remap[root];
            }
        }
    }

    Ok(EuclideanClusterResult {
        cloud: with_labels(input, "label", labels)?,
        cluster_count: cluster_sizes.len(),
        cluster_sizes,
    })
}

pub(crate) fn extract_cpu_roots(
    input: &PointCloud,
    config: EuclideanClusterConfig,
) -> SpatialResult<Vec<u32>> {
    let (x, y, z) = input.positions3()?;
    let tree = KdTree::from_slices(x, y, z);
    let len = input.len();
    let mut processed = vec![false; len];
    let mut roots = vec![0u32; len];

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
                tree.radius_search(x[index], y[index], z[index], config.cluster_tolerance);
            for neighbor in neighbors {
                let candidate = neighbor.index;
                if processed[candidate] {
                    continue;
                }
                processed[candidate] = true;
                queue.push_back(candidate);
            }
        }

        let root = *cluster_indices.iter().min().unwrap_or(&seed) as u32;
        for index in cluster_indices {
            roots[index] = root;
        }
    }

    Ok(roots)
}

fn validate_cluster_config(config: EuclideanClusterConfig) -> SpatialResult<()> {
    if config.cluster_tolerance < 0.0 {
        return Err(SpatialError::InvalidArgument(
            "cluster_tolerance must be non-negative".to_owned(),
        ));
    }
    if config.min_cluster_size == 0 {
        return Err(SpatialError::InvalidArgument(
            "min_cluster_size must be greater than zero".to_owned(),
        ));
    }
    if config.max_cluster_size < config.min_cluster_size {
        return Err(SpatialError::InvalidArgument(
            "max_cluster_size must be >= min_cluster_size".to_owned(),
        ));
    }
    Ok(())
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
            ..Default::default()
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
            ..Default::default()
        });
        let result = extractor.extract(&input).unwrap();
        assert_eq!(result.cluster_count, 0);
    }
}

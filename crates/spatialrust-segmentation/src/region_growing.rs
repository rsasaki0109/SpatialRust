use std::collections::VecDeque;

use spatialrust_core::{
    HasNormals3, HasPositions3, PointBuffer, PointCloud, SpatialError, SpatialResult,
};
use spatialrust_search::{KdTree, NearestNeighborIndex};

use crate::cloud::with_labels;
use crate::segmenter::PointCloudSegmenter;

/// Configuration for normal-based region growing segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionGrowingConfig {
    /// Number of nearest neighbors considered per point.
    pub k_neighbors: usize,
    /// Maximum angle (radians) between point normals for them to join a region.
    pub smoothness_threshold: f32,
    /// Maximum curvature for a point to act as a growth seed.
    ///
    /// Only applied when the input cloud carries a `curvature` field; flatter
    /// points (low curvature) are grown first and propagate the region.
    pub curvature_threshold: f32,
    /// Minimum number of points required to keep a region.
    pub min_cluster_size: usize,
    /// Maximum number of points allowed in a region.
    pub max_cluster_size: usize,
}

impl Default for RegionGrowingConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 30,
            // ~3 degrees, matching common region-growing defaults.
            smoothness_threshold: 0.052_359_88,
            curvature_threshold: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        }
    }
}

impl RegionGrowingConfig {
    /// Creates a config with the given smoothness angle (radians) and neighbor count.
    #[must_use]
    pub const fn with_smoothness(smoothness_threshold: f32, k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            smoothness_threshold,
            curvature_threshold: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        }
    }
}

/// Result of region growing segmentation.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionGrowingResult {
    /// Input points annotated with region labels (`label` field, `-1` = unassigned).
    pub cloud: PointCloud,
    /// Number of accepted regions.
    pub cluster_count: usize,
    /// Size of each region in label order.
    pub cluster_sizes: Vec<usize>,
}

/// Normal-based region growing segmenter.
///
/// Grows smooth regions by connecting neighboring points whose normals differ by
/// less than [`RegionGrowingConfig::smoothness_threshold`], seeding growth from
/// the flattest (lowest-curvature) points first. The input cloud must carry
/// normals (e.g. from normal estimation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionGrowingSegmenter {
    config: RegionGrowingConfig,
}

impl RegionGrowingSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: RegionGrowingConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> RegionGrowingConfig {
        self.config
    }

    /// Segments the input cloud into smooth regions, adding a `label` field.
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<RegionGrowingResult> {
        if input.is_empty() {
            return Ok(RegionGrowingResult {
                cloud: input.clone(),
                cluster_count: 0,
                cluster_sizes: Vec::new(),
            });
        }
        if self.config.k_neighbors == 0 {
            return Err(SpatialError::InvalidArgument(
                "k_neighbors must be greater than zero".to_owned(),
            ));
        }
        if self.config.smoothness_threshold < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "smoothness_threshold must be non-negative".to_owned(),
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
        let (nx, ny, nz) = input.normals3()?;
        let curvature = match input.field("curvature") {
            Ok(PointBuffer::F32(values)) => Some(values.as_slice()),
            _ => None,
        };

        let len = input.len();
        let tree = KdTree::from_slices(x, y, z);
        let cos_threshold = self.config.smoothness_threshold.cos();

        // Seed order: flattest points first when curvature is available.
        let mut order: Vec<usize> = (0..len).collect();
        if let Some(curv) = curvature {
            order.sort_by(|&a, &b| curv[a].total_cmp(&curv[b]));
        }

        let mut processed = vec![false; len];
        let mut labels = vec![-1_i32; len];
        let mut cluster_sizes = Vec::new();
        let mut cluster_id = 0_i32;

        for &start in &order {
            if processed[start] {
                continue;
            }

            let mut seeds = VecDeque::from([start]);
            // Each accepted point is pushed to `region` exactly once, at discovery.
            let mut region = vec![start];
            processed[start] = true;

            while let Some(current) = seeds.pop_front() {
                let neighbors =
                    tree.nearest_k(x[current], y[current], z[current], self.config.k_neighbors + 1);
                for neighbor in neighbors {
                    let candidate = neighbor.index;
                    if candidate == current || processed[candidate] {
                        continue;
                    }
                    // Smoothness test: treat antiparallel normals as aligned.
                    let dot = (nx[current] * nx[candidate]
                        + ny[current] * ny[candidate]
                        + nz[current] * nz[candidate])
                        .abs();
                    if dot < cos_threshold {
                        continue;
                    }
                    processed[candidate] = true;
                    region.push(candidate);
                    // Flat points keep the region growing; rough points stay leaves.
                    let flat = match curvature {
                        Some(curv) => curv[candidate] <= self.config.curvature_threshold,
                        None => true,
                    };
                    if flat {
                        seeds.push_back(candidate);
                    }
                }
            }

            if region.len() >= self.config.min_cluster_size
                && region.len() <= self.config.max_cluster_size
            {
                for index in &region {
                    labels[*index] = cluster_id;
                }
                cluster_sizes.push(region.len());
                cluster_id += 1;
            }
        }

        Ok(RegionGrowingResult {
            cloud: with_labels(input, "label", labels)?,
            cluster_count: cluster_sizes.len(),
            cluster_sizes,
        })
    }
}

impl PointCloudSegmenter for RegionGrowingSegmenter {
    fn name(&self) -> &'static str {
        "RegionGrowingSegmenter"
    }
}

#[cfg(test)]
mod tests {
    use super::{RegionGrowingConfig, RegionGrowingSegmenter};
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};

    fn schema_with_normals() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }

    /// A floor (normal +Z) and a wall (normal +Y) meeting along the y=0 edge.
    fn floor_and_wall() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(schema_with_normals());
        for i in 0..5 {
            for j in 0..5 {
                let (xf, yf) = (i as f32, j as f32);
                // floor: z = 0, normal up
                builder.push_point([xf, yf, 0.0, 0.0, 0.0, 1.0]).unwrap();
                // wall: y = 0, rising in z, normal +Y
                builder.push_point([xf, 0.0, yf + 1.0, 0.0, 1.0, 0.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn perpendicular_faces_split_into_two_regions() {
        let input = floor_and_wall();
        let segmenter = RegionGrowingSegmenter::new(RegionGrowingConfig::with_smoothness(
            10.0_f32.to_radians(),
            8,
        ));
        let result = segmenter.segment(&input).unwrap();
        assert_eq!(result.cluster_count, 2);
        assert!(result.cloud.field("label").is_ok());
    }

    #[test]
    fn coplanar_points_form_single_region() {
        let mut builder = PointCloudBuilder::new(schema_with_normals());
        for i in 0..5 {
            for j in 0..5 {
                builder.push_point([i as f32, j as f32, 0.0, 0.0, 0.0, 1.0]).unwrap();
            }
        }
        let input = builder.build().unwrap();

        let segmenter = RegionGrowingSegmenter::new(RegionGrowingConfig::with_smoothness(
            10.0_f32.to_radians(),
            8,
        ));
        let result = segmenter.segment(&input).unwrap();
        assert_eq!(result.cluster_count, 1);
        assert_eq!(result.cluster_sizes, vec![25]);
    }

    #[test]
    fn requires_normals() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let input = builder.build().unwrap();
        let segmenter = RegionGrowingSegmenter::new(RegionGrowingConfig::default());
        assert!(segmenter.segment(&input).is_err());
    }
}

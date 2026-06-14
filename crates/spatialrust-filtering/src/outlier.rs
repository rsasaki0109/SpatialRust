//! Neighborhood-based outlier removal filters.
//!
//! Both filters use a KD-tree over the input positions and drop points whose
//! local neighborhood looks sparse, which removes scanner speckle and stray
//! returns before downstream estimation (normals, registration, segmentation).

use spatialrust_core::{
    HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError, SpatialResult,
};
use spatialrust_search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};

use crate::filter::PointCloudFilter;

/// Configuration for [`StatisticalOutlierRemoval`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatisticalOutlierConfig {
    /// Number of nearest neighbors averaged per point.
    pub k_neighbors: usize,
    /// Standard-deviation multiplier; points whose mean neighbor distance
    /// exceeds `global_mean + std_mul * global_std` are removed.
    pub std_mul: f32,
}

impl Default for StatisticalOutlierConfig {
    fn default() -> Self {
        Self { k_neighbors: 16, std_mul: 1.0 }
    }
}

impl StatisticalOutlierConfig {
    /// Creates a config from the neighbor count and std multiplier.
    #[must_use]
    pub const fn new(k_neighbors: usize, std_mul: f32) -> Self {
        Self { k_neighbors, std_mul }
    }
}

/// Statistical Outlier Removal (SOR).
///
/// For each point the mean distance to its `k` nearest neighbors is computed.
/// Assuming those means are roughly Gaussian, points whose mean distance is
/// more than `std_mul` standard deviations above the global mean are dropped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatisticalOutlierRemoval {
    config: StatisticalOutlierConfig,
}

impl StatisticalOutlierRemoval {
    /// Creates a filter from config.
    #[must_use]
    pub const fn new(config: StatisticalOutlierConfig) -> Self {
        Self { config }
    }

    /// Returns the filter config.
    #[must_use]
    pub const fn config(&self) -> StatisticalOutlierConfig {
        self.config
    }

    /// Computes the keep mask without materializing the filtered cloud.
    pub fn keep_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        if self.config.k_neighbors == 0 {
            return Err(SpatialError::InvalidArgument(
                "k_neighbors must be greater than zero".to_owned(),
            ));
        }
        let len = input.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let (x, y, z) = input.positions3()?;
        let tree = KdTree::from_slices(x, y, z);

        // Mean distance to the k nearest neighbors (excluding the point itself).
        let mut mean_dist = vec![0.0_f32; len];
        for i in 0..len {
            let neighbors = tree.nearest_k(x[i], y[i], z[i], self.config.k_neighbors + 1);
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for neighbor in neighbors {
                if neighbor.index == i {
                    continue;
                }
                sum += neighbor.distance_squared.sqrt();
                count += 1;
            }
            mean_dist[i] = if count == 0 { 0.0 } else { sum / count as f32 };
        }

        let n = len as f64;
        let mean: f64 = mean_dist.iter().map(|&d| d as f64).sum::<f64>() / n;
        let variance: f64 = mean_dist.iter().map(|&d| (d as f64 - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let threshold = mean + self.config.std_mul as f64 * std;

        Ok(mean_dist.iter().map(|&d| d as f64 <= threshold).collect())
    }
}

impl PointCloudFilter for StatisticalOutlierRemoval {
    fn name(&self) -> &'static str {
        "StatisticalOutlierRemoval"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let mask = self.keep_mask(input)?;
        gather_mask(input, &mask)
    }
}

/// Configuration for [`RadiusOutlierRemoval`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiusOutlierConfig {
    /// Search radius (not squared) defining a point's neighborhood.
    pub radius: f32,
    /// Minimum neighbors (excluding the point itself) required to keep a point.
    pub min_neighbors: usize,
}

impl Default for RadiusOutlierConfig {
    fn default() -> Self {
        Self { radius: 0.5, min_neighbors: 4 }
    }
}

impl RadiusOutlierConfig {
    /// Creates a config from the radius and minimum neighbor count.
    #[must_use]
    pub const fn new(radius: f32, min_neighbors: usize) -> Self {
        Self { radius, min_neighbors }
    }
}

/// Radius Outlier Removal (ROR).
///
/// Drops every point that has fewer than `min_neighbors` other points within
/// `radius`. Unlike SOR this uses an absolute density threshold, so it is robust
/// when outliers are clustered rather than isolated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiusOutlierRemoval {
    config: RadiusOutlierConfig,
}

impl RadiusOutlierRemoval {
    /// Creates a filter from config.
    #[must_use]
    pub const fn new(config: RadiusOutlierConfig) -> Self {
        Self { config }
    }

    /// Returns the filter config.
    #[must_use]
    pub const fn config(&self) -> RadiusOutlierConfig {
        self.config
    }

    /// Computes the keep mask without materializing the filtered cloud.
    pub fn keep_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        if self.config.radius <= 0.0 || self.config.radius.is_nan() {
            return Err(SpatialError::InvalidArgument("radius must be positive".to_owned()));
        }
        let len = input.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let (x, y, z) = input.positions3()?;
        let tree = KdTree::from_slices(x, y, z);

        let mut keep = vec![false; len];
        for i in 0..len {
            let neighbors = tree.radius_search(x[i], y[i], z[i], self.config.radius);
            // radius_search includes the query point itself when it is indexed.
            let others = neighbors.iter().filter(|n| n.index != i).count();
            keep[i] = others >= self.config.min_neighbors;
        }
        Ok(keep)
    }
}

impl PointCloudFilter for RadiusOutlierRemoval {
    fn name(&self) -> &'static str {
        "RadiusOutlierRemoval"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let mask = self.keep_mask(input)?;
        gather_mask(input, &mask)
    }
}

/// Builds a new cloud from the points where `mask` is true, preserving schema.
fn gather_mask(input: &PointCloud, mask: &[bool]) -> SpatialResult<PointCloud> {
    let indices: Vec<usize> =
        mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, &indices));
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn gather_buffer(source: &PointBuffer, indices: &[usize]) -> PointBuffer {
    match source {
        PointBuffer::F32(v) => PointBuffer::from_f32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::F64(v) => PointBuffer::F64(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U8(v) => PointBuffer::U8(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U16(v) => PointBuffer::U16(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U32(v) => PointBuffer::U32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::I32(v) => PointBuffer::I32(indices.iter().map(|&i| v[i]).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema};

    fn cloud_from_xyz(points: &[[f32; 3]]) -> PointCloud {
        let schema = PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32));
        let mut buffers = PointBufferSet::new();
        buffers
            .insert("x".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[0]).collect()));
        buffers
            .insert("y".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[1]).collect()));
        buffers
            .insert("z".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[2]).collect()));
        PointCloud::try_from_parts(schema, buffers, Default::default()).unwrap()
    }

    /// A dense unit-spaced grid plus one far-away speckle point.
    fn grid_with_outlier() -> (PointCloud, usize) {
        let mut points = Vec::new();
        for ix in 0..6 {
            for iy in 0..6 {
                points.push([ix as f32, iy as f32, 0.0]);
            }
        }
        let outlier_index = points.len();
        points.push([100.0, 100.0, 100.0]);
        (cloud_from_xyz(&points), outlier_index)
    }

    #[test]
    fn sor_removes_isolated_speckle() {
        let (cloud, outlier) = grid_with_outlier();
        let filter = StatisticalOutlierRemoval::new(StatisticalOutlierConfig::new(8, 1.0));
        let mask = filter.keep_mask(&cloud).unwrap();
        assert!(!mask[outlier], "the far speckle must be dropped");
        // Every dense grid point should survive.
        assert!(mask[..outlier].iter().all(|&k| k));
        let out = filter.filter(&cloud).unwrap();
        assert_eq!(out.len(), cloud.len() - 1);
    }

    #[test]
    fn ror_removes_isolated_speckle() {
        let (cloud, outlier) = grid_with_outlier();
        let filter = RadiusOutlierRemoval::new(RadiusOutlierConfig::new(1.5, 2));
        let mask = filter.keep_mask(&cloud).unwrap();
        assert!(!mask[outlier], "the far speckle has no neighbors in radius");
        assert!(mask[..outlier].iter().all(|&k| k));
    }

    #[test]
    fn empty_cloud_is_passthrough() {
        let cloud = cloud_from_xyz(&[]);
        let sor = StatisticalOutlierRemoval::new(StatisticalOutlierConfig::default());
        assert_eq!(sor.filter(&cloud).unwrap().len(), 0);
        let ror = RadiusOutlierRemoval::new(RadiusOutlierConfig::default());
        assert_eq!(ror.filter(&cloud).unwrap().len(), 0);
    }

    #[test]
    fn invalid_params_error() {
        let cloud = cloud_from_xyz(&[[0.0, 0.0, 0.0]]);
        assert!(StatisticalOutlierRemoval::new(StatisticalOutlierConfig::new(0, 1.0))
            .keep_mask(&cloud)
            .is_err());
        assert!(RadiusOutlierRemoval::new(RadiusOutlierConfig::new(0.0, 1))
            .keep_mask(&cloud)
            .is_err());
    }
}

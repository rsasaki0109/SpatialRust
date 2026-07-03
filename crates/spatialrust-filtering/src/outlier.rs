//! Neighborhood-based outlier removal filters.
//!
//! Both filters use a KD-tree over the input positions and drop points whose
//! local neighborhood looks sparse, which removes scanner speckle and stray
//! returns before downstream estimation (normals, registration, segmentation).

use spatialrust_core::{
    DType, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet, PointCloud, PointField,
    SpatialError, SpatialResult,
};
use spatialrust_search::{parallel_worker_count, KdTree};

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
        fill_mean_neighbor_distances(self.config.k_neighbors, &tree, x, y, z, &mut mean_dist);

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

fn fill_mean_neighbor_distances(
    k_neighbors: usize,
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    mean_dist: &mut [f32],
) {
    let worker_count = parallel_worker_count(mean_dist.len());
    if worker_count == 1 {
        fill_mean_neighbor_distances_chunk(k_neighbors, tree, x, y, z, 0, mean_dist);
        return;
    }

    let chunk_size = mean_dist.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in mean_dist.chunks_mut(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                fill_mean_neighbor_distances_chunk(k_neighbors, tree, x, y, z, start, chunk);
            });
        }
    });
}

fn fill_mean_neighbor_distances_chunk(
    k_neighbors: usize,
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    start: usize,
    mean_dist: &mut [f32],
) {
    let mut neighbors = Vec::with_capacity(k_neighbors.saturating_add(1));
    for (offset, mean) in mean_dist.iter_mut().enumerate() {
        let i = start + offset;
        tree.nearest_k_unsorted_into(
            x[i],
            y[i],
            z[i],
            k_neighbors.saturating_add(1),
            &mut neighbors,
        );
        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for neighbor in &neighbors {
            if neighbor.index == i {
                continue;
            }
            sum += neighbor.distance_squared.sqrt();
            count += 1;
        }
        *mean = if count == 0 { 0.0 } else { sum / count as f32 };
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

        // The query point itself is in the tree, so requiring `min_neighbors`
        // *other* points within radius means reaching `min_neighbors + 1` total.
        // `radius_reaches` early-exits at that threshold without allocating.
        let target = self.config.min_neighbors + 1;
        let mut keep = vec![false; len];
        fill_radius_reaches_mask(&tree, x, y, z, self.config.radius, target, &mut keep);
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

fn fill_radius_reaches_mask(
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
    target: usize,
    keep: &mut [bool],
) {
    let worker_count = parallel_worker_count(keep.len());
    if worker_count == 1 {
        fill_radius_reaches_mask_chunk(tree, x, y, z, radius, target, 0, keep);
        return;
    }

    let chunk_size = keep.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in keep.chunks_mut(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                fill_radius_reaches_mask_chunk(tree, x, y, z, radius, target, start, chunk);
            });
        }
    });
}

fn fill_radius_reaches_mask_chunk(
    tree: &KdTree,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
    target: usize,
    start: usize,
    keep: &mut [bool],
) {
    for (offset, keep_point) in keep.iter_mut().enumerate() {
        let i = start + offset;
        *keep_point = tree.radius_reaches(x[i], y[i], z[i], radius, target);
    }
}

/// Builds a new cloud from the points where `mask` is true, preserving schema.
fn gather_mask(input: &PointCloud, mask: &[bool]) -> SpatialResult<PointCloud> {
    if let Some(output) = gather_xyz_mask(input, mask)? {
        return Ok(output);
    }

    let indices: Vec<usize> =
        mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, &indices));
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn gather_xyz_mask(input: &PointCloud, mask: &[bool]) -> SpatialResult<Option<PointCloud>> {
    let schema = input.schema();
    if schema.len() != 3 {
        return Ok(None);
    }

    let Some(x_field) = xyz_f32_field(input, FieldSemantic::PositionX) else {
        return Ok(None);
    };
    let Some(y_field) = xyz_f32_field(input, FieldSemantic::PositionY) else {
        return Ok(None);
    };
    let Some(z_field) = xyz_f32_field(input, FieldSemantic::PositionZ) else {
        return Ok(None);
    };

    let (x, y, z) = input.positions3()?;
    let output_len = mask.iter().filter(|&&keep| keep).count();
    let mut out_x = Vec::with_capacity(output_len);
    let mut out_y = Vec::with_capacity(output_len);
    let mut out_z = Vec::with_capacity(output_len);

    for (index, &keep) in mask.iter().enumerate() {
        if keep {
            out_x.push(x[index]);
            out_y.push(y[index]);
            out_z.push(z[index]);
        }
    }

    let mut buffers = PointBufferSet::new();
    buffers.insert(x_field.name.clone(), PointBuffer::from_f32(out_x));
    buffers.insert(y_field.name.clone(), PointBuffer::from_f32(out_y));
    buffers.insert(z_field.name.clone(), PointBuffer::from_f32(out_z));
    PointCloud::try_from_parts(schema.clone(), buffers, input.metadata().clone()).map(Some)
}

fn xyz_f32_field(input: &PointCloud, semantic: FieldSemantic) -> Option<&PointField> {
    let field = input.schema().find_semantic(semantic)?;
    (field.dtype == DType::F32 && field.components == 1).then_some(field)
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

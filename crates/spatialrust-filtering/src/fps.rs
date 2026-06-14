//! Farthest Point Sampling (FPS).
//!
//! Greedily selects a subset that is spread as evenly as possible over the
//! cloud: each new point is the one farthest from everything chosen so far. This
//! is the standard downsampling for learned point-cloud models (PointNet++ and
//! friends), where uniform spatial coverage matters more than a fixed grid.

use spatialrust_core::{
    HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError, SpatialResult,
};

use crate::filter::PointCloudFilter;

/// Configuration for [`FarthestPointSampling`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarthestPointSamplingConfig {
    /// Number of points to keep.
    pub sample_size: usize,
    /// Index of the first seed point (the rest are chosen deterministically).
    pub seed_index: usize,
}

impl Default for FarthestPointSamplingConfig {
    fn default() -> Self {
        Self { sample_size: 1024, seed_index: 0 }
    }
}

impl FarthestPointSamplingConfig {
    /// Creates a config keeping `sample_size` points, seeded from index 0.
    #[must_use]
    pub const fn new(sample_size: usize) -> Self {
        Self { sample_size, seed_index: 0 }
    }
}

/// Farthest Point Sampling downsampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarthestPointSampling {
    config: FarthestPointSamplingConfig,
}

impl FarthestPointSampling {
    /// Creates a sampler from config.
    #[must_use]
    pub const fn new(config: FarthestPointSamplingConfig) -> Self {
        Self { config }
    }

    /// Returns the sampler config.
    #[must_use]
    pub const fn config(&self) -> FarthestPointSamplingConfig {
        self.config
    }

    /// Returns the selected point indices in selection order.
    pub fn select(&self, input: &PointCloud) -> SpatialResult<Vec<usize>> {
        if self.config.sample_size == 0 {
            return Err(SpatialError::InvalidArgument(
                "sample_size must be greater than zero".to_owned(),
            ));
        }
        let len = input.len();
        if self.config.seed_index >= len.max(1) {
            return Err(SpatialError::InvalidArgument("seed_index is out of range".to_owned()));
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        if self.config.sample_size >= len {
            return Ok((0..len).collect());
        }

        let (x, y, z) = input.positions3()?;

        // `min_dist[i]` = squared distance from point i to the nearest selected
        // point. Seed it from the first chosen point, then repeatedly take the
        // current maximum and relax the array against the new selection.
        let mut selected = Vec::with_capacity(self.config.sample_size);
        let mut min_dist = vec![f32::INFINITY; len];
        let mut current = self.config.seed_index;
        selected.push(current);

        for _ in 1..self.config.sample_size {
            let (cx, cy, cz) = (x[current], y[current], z[current]);
            let mut best = 0_usize;
            let mut best_dist = -1.0_f32;
            for i in 0..len {
                let dx = x[i] - cx;
                let dy = y[i] - cy;
                let dz = z[i] - cz;
                let d = dx * dx + dy * dy + dz * dz;
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
                if min_dist[i] > best_dist {
                    best_dist = min_dist[i];
                    best = i;
                }
            }
            current = best;
            selected.push(current);
        }

        Ok(selected)
    }
}

impl PointCloudFilter for FarthestPointSampling {
    fn name(&self) -> &'static str {
        "FarthestPointSampling"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let indices = self.select(input)?;
        gather_indices(input, &indices)
    }
}

/// Gathers the selected indices into a new cloud, preserving schema.
fn gather_indices(input: &PointCloud, indices: &[usize]) -> SpatialResult<PointCloud> {
    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, indices));
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
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn grid(n: usize) -> PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..n {
            for j in 0..n {
                builder.push_point([i as f32, j as f32, 0.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn selects_requested_count() {
        let cloud = grid(10);
        let out = FarthestPointSampling::new(FarthestPointSamplingConfig::new(16))
            .filter(&cloud)
            .unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn samples_are_spread_out() {
        // The four corners of a 10x10 grid should be among the first selections
        // because FPS maximizes spacing.
        let cloud = grid(10);
        let indices =
            FarthestPointSampling::new(FarthestPointSamplingConfig::new(4)).select(&cloud).unwrap();
        // Index 0 = (0,0). The next pick must be the opposite corner (9,9)=99.
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 99);
    }

    #[test]
    fn oversampling_returns_all_points() {
        let cloud = grid(3);
        let out = FarthestPointSampling::new(FarthestPointSamplingConfig::new(100))
            .filter(&cloud)
            .unwrap();
        assert_eq!(out.len(), cloud.len());
    }

    #[test]
    fn rejects_bad_params() {
        let cloud = grid(3);
        assert!(FarthestPointSampling::new(FarthestPointSamplingConfig::new(0))
            .select(&cloud)
            .is_err());
        assert!(FarthestPointSampling::new(FarthestPointSamplingConfig {
            sample_size: 4,
            seed_index: 999
        })
        .select(&cloud)
        .is_err());
    }
}

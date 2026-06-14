//! Sequential multi-plane extraction (floor, walls, ceiling, …).
//!
//! Indoor and structured scenes contain several dominant planes, not one. This
//! repeatedly fits the dominant plane with RANSAC, labels and removes its
//! inliers, and continues on the remainder — the standard way to decompose a
//! room into its planar surfaces.

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};

use crate::cloud::{extract_indices, with_labels};
use crate::plane::{PlaneModel, RansacPlaneConfig, RansacPlaneSegmenter};
use crate::segmenter::PointCloudSegmenter;

/// Configuration for [`MultiPlaneSegmenter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiPlaneConfig {
    /// Maximum number of planes to extract.
    pub max_planes: usize,
    /// Maximum distance from a plane for inlier classification.
    pub distance_threshold: f32,
    /// Minimum inliers to accept a plane; extraction stops below this.
    pub min_inliers: usize,
    /// RANSAC iterations per plane.
    pub max_iterations: usize,
    /// Seed for deterministic sampling.
    pub seed: u64,
}

impl Default for MultiPlaneConfig {
    fn default() -> Self {
        Self {
            max_planes: 4,
            distance_threshold: 0.02,
            min_inliers: 100,
            max_iterations: 1_000,
            seed: 42,
        }
    }
}

impl MultiPlaneConfig {
    /// Creates a config from the plane count and distance threshold.
    #[must_use]
    pub const fn new(max_planes: usize, distance_threshold: f32) -> Self {
        Self { max_planes, distance_threshold, min_inliers: 100, max_iterations: 1_000, seed: 42 }
    }
}

/// Result of multi-plane segmentation.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiPlaneSegmentation {
    /// Fitted plane models, in extraction order (most dominant first).
    pub planes: Vec<PlaneModel>,
    /// Input cloud with a `label` field: plane index `0..planes.len()`, or `-1`
    /// for points not assigned to any plane.
    pub labeled: PointCloud,
    /// Number of points assigned to each plane, in plane order.
    pub plane_sizes: Vec<usize>,
}

/// Sequential RANSAC multi-plane segmenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiPlaneSegmenter {
    config: MultiPlaneConfig,
}

impl MultiPlaneSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: MultiPlaneConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> MultiPlaneConfig {
        self.config
    }

    /// Extracts up to `max_planes` dominant planes and labels each point.
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<MultiPlaneSegmentation> {
        if self.config.distance_threshold < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "distance_threshold must be non-negative".to_owned(),
            ));
        }
        let len = input.len();
        let (x, y, z) = input.positions3()?;

        let mut labels = vec![-1_i32; len];
        let mut remaining: Vec<usize> = (0..len).collect();
        let mut planes = Vec::new();
        let mut plane_sizes = Vec::new();

        for plane_index in 0..self.config.max_planes {
            if remaining.len() < 3 || remaining.len() < self.config.min_inliers {
                break;
            }

            // Fit the dominant plane of whatever points are left.
            let sub = extract_indices(input, &remaining)?;
            let config = RansacPlaneConfig {
                distance_threshold: self.config.distance_threshold,
                max_iterations: self.config.max_iterations,
                min_inliers: self.config.min_inliers,
                // Vary the seed per plane so successive fits explore differently.
                seed: self.config.seed.wrapping_add(plane_index as u64),
            };
            let Ok(result) = RansacPlaneSegmenter::new(config).segment(&sub) else {
                // No plane with enough inliers remains.
                break;
            };

            // Re-classify the *remaining original* points against the fitted
            // model so labels map back to the input cloud.
            let model = result.model;
            let mut next_remaining = Vec::with_capacity(remaining.len());
            let mut assigned = 0_usize;
            for &orig in &remaining {
                if model.distance_xyz(x[orig], y[orig], z[orig]) <= self.config.distance_threshold {
                    labels[orig] = plane_index as i32;
                    assigned += 1;
                } else {
                    next_remaining.push(orig);
                }
            }

            if assigned < self.config.min_inliers {
                // The model did not actually cover enough original points; undo.
                for &orig in &remaining {
                    if labels[orig] == plane_index as i32 {
                        labels[orig] = -1;
                    }
                }
                break;
            }

            planes.push(model);
            plane_sizes.push(assigned);
            remaining = next_remaining;
        }

        Ok(MultiPlaneSegmentation {
            labeled: with_labels(input, "label", labels)?,
            planes,
            plane_sizes,
        })
    }
}

impl PointCloudSegmenter for MultiPlaneSegmenter {
    fn name(&self) -> &'static str {
        "MultiPlaneSegmenter"
    }
}

#[cfg(test)]
mod tests {
    use super::{MultiPlaneConfig, MultiPlaneSegmenter};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    /// A floor (z=0), a wall (y=0), and a ceiling (z=2): three planes.
    fn room() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..20 {
            for j in 0..20 {
                let (a, b) = (i as f32 * 0.1, j as f32 * 0.1);
                builder.push_point([a, b, 0.0]).unwrap(); // floor
                builder.push_point([a, 0.0, b]).unwrap(); // wall
                builder.push_point([a, b, 2.0]).unwrap(); // ceiling
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn extracts_three_room_planes() {
        let cloud = room();
        let seg = MultiPlaneSegmenter::new(MultiPlaneConfig {
            max_planes: 4,
            distance_threshold: 0.02,
            min_inliers: 100,
            max_iterations: 500,
            seed: 7,
        })
        .segment(&cloud)
        .unwrap();

        assert_eq!(seg.planes.len(), 3, "expected floor, wall, ceiling");
        assert!(seg.labeled.field("label").is_ok());
        // The three planes together cover almost every point.
        let covered: usize = seg.plane_sizes.iter().sum();
        assert!(covered as f32 > cloud.len() as f32 * 0.95);
    }

    #[test]
    fn stops_when_no_dominant_plane_remains() {
        // Pure noise has no plane with enough inliers.
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        let mut seed = 99_u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for _ in 0..400 {
            builder.push_point([rng() * 5.0, rng() * 5.0, rng() * 5.0]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let seg = MultiPlaneSegmenter::new(MultiPlaneConfig::new(3, 0.01)).segment(&cloud).unwrap();
        assert!(seg.planes.is_empty(), "noise should yield no planes");
    }

    #[test]
    fn rejects_bad_threshold() {
        let cloud = room();
        let config = MultiPlaneConfig { distance_threshold: -1.0, ..MultiPlaneConfig::default() };
        assert!(MultiPlaneSegmenter::new(config).segment(&cloud).is_err());
    }
}

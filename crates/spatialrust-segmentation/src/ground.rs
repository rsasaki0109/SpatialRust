//! Grid-based ground segmentation for outdoor (LiDAR) scans.
//!
//! The cloud is binned into a 2D grid over the ground plane; each cell's minimum
//! height seeds a ground-elevation estimate, which is then eroded against its
//! neighbors (a morphological opening) so that isolated high cells — rooftops,
//! vehicle tops — are pulled down to the surrounding ground level rather than
//! seeding their own false ground. A point is ground if it sits within
//! `height_threshold` of that estimate.

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};

use crate::cloud::{extract_mask, with_labels};
use crate::segmenter::PointCloudSegmenter;

/// Which axis points "up" (its minimum defines local ground height).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpAxis {
    /// +X is up.
    X,
    /// +Y is up.
    Y,
    /// +Z is up (the common case).
    Z,
}

/// Configuration for [`GroundSegmenter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundConfig {
    /// Side length of each grid cell in the ground plane.
    pub cell_size: f32,
    /// Max height above the local ground estimate for a point to be ground.
    pub height_threshold: f32,
    /// Erosion window radius in cells (0 disables erosion).
    pub erosion_cells: usize,
    /// Which axis is "up".
    pub up_axis: UpAxis,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self { cell_size: 0.5, height_threshold: 0.2, erosion_cells: 1, up_axis: UpAxis::Z }
    }
}

impl GroundConfig {
    /// Creates a config from the cell size and height threshold (Z up).
    #[must_use]
    pub const fn new(cell_size: f32, height_threshold: f32) -> Self {
        Self { cell_size, height_threshold, erosion_cells: 1, up_axis: UpAxis::Z }
    }
}

/// Result of ground segmentation.
#[derive(Clone, Debug, PartialEq)]
pub struct GroundSegmentation {
    /// Points classified as ground.
    pub ground: PointCloud,
    /// Points classified as non-ground (objects, vegetation, structures).
    pub non_ground: PointCloud,
    /// Input cloud with a `label` field (1 = ground, 0 = non-ground).
    pub labeled: PointCloud,
    /// Number of ground points.
    pub ground_count: usize,
}

/// Grid-based ground segmenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSegmenter {
    config: GroundConfig,
}

impl GroundSegmenter {
    /// Creates a segmenter from config.
    #[must_use]
    pub const fn new(config: GroundConfig) -> Self {
        Self { config }
    }

    /// Returns the segmenter config.
    #[must_use]
    pub const fn config(&self) -> GroundConfig {
        self.config
    }

    /// Computes the per-point ground mask (`true` = ground).
    pub fn ground_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        if self.config.cell_size <= 0.0 || self.config.cell_size.is_nan() {
            return Err(SpatialError::InvalidArgument("cell_size must be positive".to_owned()));
        }
        let len = input.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        let (x, y, z) = input.positions3()?;
        // (plane_a, plane_b) span the ground; `up` is the height axis.
        let (pa, pb, up): (&[f32], &[f32], &[f32]) = match self.config.up_axis {
            UpAxis::X => (y, z, x),
            UpAxis::Y => (x, z, y),
            UpAxis::Z => (x, y, z),
        };

        let inv_cell = 1.0 / self.config.cell_size;
        let min_a = pa.iter().copied().fold(f32::INFINITY, f32::min);
        let min_b = pb.iter().copied().fold(f32::INFINITY, f32::min);
        let max_a = pa.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_b = pb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cols = (((max_a - min_a) * inv_cell) as usize) + 1;
        let rows = (((max_b - min_b) * inv_cell) as usize) + 1;

        let cell_of = |i: usize| {
            let c = (((pa[i] - min_a) * inv_cell) as usize).min(cols - 1);
            let r = (((pb[i] - min_b) * inv_cell) as usize).min(rows - 1);
            r * cols + c
        };

        // Per-cell minimum height.
        let mut cell_min = vec![f32::INFINITY; cols * rows];
        for (i, &up_i) in up.iter().enumerate() {
            let cell = cell_of(i);
            if up_i < cell_min[cell] {
                cell_min[cell] = up_i;
            }
        }

        // Morphological erosion: each cell's ground reference is the minimum of
        // its neighborhood, so a lone high cell adopts the surrounding ground.
        let ground_ref = if self.config.erosion_cells == 0 {
            cell_min.clone()
        } else {
            erode(&cell_min, cols, rows, self.config.erosion_cells)
        };

        let mut mask = vec![false; len];
        for (i, &up_i) in up.iter().enumerate() {
            let reference = ground_ref[cell_of(i)];
            if reference.is_finite() && up_i - reference <= self.config.height_threshold {
                mask[i] = true;
            }
        }
        Ok(mask)
    }

    /// Segments the cloud into ground and non-ground partitions.
    pub fn segment(&self, input: &PointCloud) -> SpatialResult<GroundSegmentation> {
        let mask = self.ground_mask(input)?;
        let non_ground_mask: Vec<bool> = mask.iter().map(|&g| !g).collect();
        let labels: Vec<i32> = mask.iter().map(|&g| i32::from(g)).collect();
        let ground_count = mask.iter().filter(|&&g| g).count();

        Ok(GroundSegmentation {
            ground: extract_mask(input, &mask)?,
            non_ground: extract_mask(input, &non_ground_mask)?,
            labeled: with_labels(input, "label", labels)?,
            ground_count,
        })
    }
}

impl PointCloudSegmenter for GroundSegmenter {
    fn name(&self) -> &'static str {
        "GroundSegmenter"
    }
}

/// Greyscale morphological erosion of a grid of heights by `radius` cells.
fn erode(cells: &[f32], cols: usize, rows: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![f32::INFINITY; cells.len()];
    let r = radius as isize;
    for row in 0..rows {
        for col in 0..cols {
            let mut m = f32::INFINITY;
            for dr in -r..=r {
                for dc in -r..=r {
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let value = cells[nr as usize * cols + nc as usize];
                        if value < m {
                            m = value;
                        }
                    }
                }
            }
            out[row * cols + col] = m;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{GroundConfig, GroundSegmenter};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    /// A flat ground grid plus a raised "building" block above part of it.
    fn ground_with_building() -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        // Ground: 20x20 at z ~ 0.
        for i in 0..20 {
            for j in 0..20 {
                builder.push_point([i as f32 * 0.5, j as f32 * 0.5, 0.0]).unwrap();
            }
        }
        // Building roof: a 5x5 block at z = 3 over one corner.
        for i in 0..5 {
            for j in 0..5 {
                builder.push_point([i as f32 * 0.5, j as f32 * 0.5, 3.0]).unwrap();
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn separates_ground_from_building() {
        let cloud = ground_with_building();
        let seg = GroundSegmenter::new(GroundConfig::new(0.6, 0.3)).segment(&cloud).unwrap();
        // 400 ground points, 25 roof points.
        assert_eq!(seg.ground_count, 400);
        assert_eq!(seg.ground.len(), 400);
        assert_eq!(seg.non_ground.len(), 25);
        assert!(seg.labeled.field("label").is_ok());
    }

    #[test]
    fn sloped_ground_stays_ground() {
        // A ramp rising along x: each cell's local min tracks the slope, so the
        // whole ramp should be classified as ground.
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..40 {
            for j in 0..10 {
                let x = i as f32 * 0.5;
                builder.push_point([x, j as f32 * 0.5, x * 0.2]).unwrap();
            }
        }
        let cloud = builder.build().unwrap();
        let seg = GroundSegmenter::new(GroundConfig::new(0.6, 0.3)).segment(&cloud).unwrap();
        // Almost everything is ground (the gentle slope fits the cell threshold).
        assert!(seg.ground_count as f32 > cloud.len() as f32 * 0.95);
    }

    #[test]
    fn rejects_bad_params() {
        let cloud = ground_with_building();
        assert!(GroundSegmenter::new(GroundConfig::new(0.0, 0.3)).ground_mask(&cloud).is_err());
    }
}

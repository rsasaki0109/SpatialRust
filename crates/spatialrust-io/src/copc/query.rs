//! COPC spatial query types and helpers.

use copc_streaming::Aabb;

use crate::error::{copc_format, IoError};

/// Axis-aligned bounds for COPC spatial queries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopcBounds {
    /// Minimum corner `[x, y, z]`.
    pub min: [f64; 3],
    /// Maximum corner `[x, y, z]`.
    pub max: [f64; 3],
}

impl CopcBounds {
    /// Creates bounds from minimum and maximum corners.
    #[must_use]
    pub const fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Creates bounds from separate axis ranges.
    #[must_use]
    pub fn from_ranges(x: (f64, f64), y: (f64, f64), z: (f64, f64)) -> Self {
        Self { min: [x.0, y.0, z.0], max: [x.1, y.1, z.1] }
    }

    /// Validates that min <= max on every axis.
    pub fn validate(&self) -> Result<(), IoError> {
        for axis in 0..3 {
            if self.min[axis] > self.max[axis] {
                return Err(copc_format(format!(
                    "invalid COPC bounds on axis {axis}: min {} > max {}",
                    self.min[axis], self.max[axis]
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn to_aabb(self) -> Aabb {
        Aabb { min: self.min, max: self.max }
    }
}

/// Spatial query parameters for partial COPC reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopcQuery {
    /// Region of interest.
    pub bounds: CopcBounds,
    /// Optional target point spacing in file units (typically meters).
    pub max_resolution: Option<f64>,
    /// Optional explicit maximum octree level. Overrides `max_resolution` when set.
    pub max_level: Option<i32>,
}

impl CopcQuery {
    /// Creates a bounds-only query at full available detail inside the region.
    #[must_use]
    pub fn bounds(bounds: CopcBounds) -> Self {
        Self { bounds, max_resolution: None, max_level: None }
    }

    /// Creates a bounds query limited by target point spacing.
    #[must_use]
    pub fn with_resolution(bounds: CopcBounds, max_resolution: f64) -> Self {
        Self { bounds, max_resolution: Some(max_resolution), max_level: None }
    }

    /// Creates a bounds query limited by explicit octree level.
    #[must_use]
    pub fn with_level(bounds: CopcBounds, max_level: i32) -> Self {
        Self { bounds, max_resolution: None, max_level: Some(max_level) }
    }

    /// Validates query parameters.
    pub fn validate(&self) -> Result<(), IoError> {
        self.bounds.validate()?;
        if let Some(resolution) = self.max_resolution {
            if !resolution.is_finite() || resolution <= 0.0 {
                return Err(copc_format(
                    "max_resolution must be a positive finite value".to_owned(),
                ));
            }
        }
        if let Some(level) = self.max_level {
            if level < 0 {
                return Err(copc_format("max_level must be non-negative".to_owned()));
            }
        }
        Ok(())
    }

    pub(crate) fn max_level_for_spacing(&self, base_spacing: f64) -> Option<i32> {
        if self.max_level.is_some() {
            return self.max_level;
        }
        self.max_resolution.map(|resolution| copc_level_for_resolution(base_spacing, resolution))
    }
}

/// Metadata exposed from a COPC header without loading point data.
#[derive(Clone, Debug, PartialEq)]
pub struct CopcFileInfo {
    /// Root octree bounds.
    pub root_bounds: CopcBounds,
    /// Base point spacing at octree level 0.
    pub spacing: f64,
    /// Declared number of points in the file header.
    pub point_count: u64,
}

/// Computes the shallowest octree level whose spacing is at most `resolution`.
#[must_use]
pub fn copc_level_for_resolution(base_spacing: f64, resolution: f64) -> i32 {
    if resolution <= 0.0 || base_spacing <= 0.0 {
        return 0;
    }
    (base_spacing / resolution).log2().ceil().max(0.0) as i32
}

#[cfg(test)]
mod tests {
    use super::{copc_level_for_resolution, CopcBounds, CopcQuery};

    #[test]
    fn validates_bounds() {
        let bounds = CopcBounds::from_ranges((0.0, 1.0), (0.0, 1.0), (0.0, 1.0));
        assert!(bounds.validate().is_ok());
        let invalid = CopcBounds::from_ranges((1.0, 0.0), (0.0, 1.0), (0.0, 1.0));
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn level_for_resolution_matches_copc_formula() {
        assert_eq!(copc_level_for_resolution(10.0, 0.5), 5);
        assert_eq!(copc_level_for_resolution(10.0, 10.0), 0);
    }

    #[test]
    fn explicit_level_overrides_resolution() {
        let query = CopcQuery {
            bounds: CopcBounds::new([0.0; 3], [1.0; 3]),
            max_resolution: Some(0.5),
            max_level: Some(2),
        };
        assert_eq!(query.max_level_for_spacing(10.0), Some(2));
    }
}

//! Bounding-volume types.

use spatialrust_math::{Mat3, Vec3};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Lower corner `(x, y, z)`.
    pub min: Vec3<f32>,
    /// Upper corner `(x, y, z)`.
    pub max: Vec3<f32>,
}

impl Aabb {
    /// Creates a box from its corners.
    #[must_use]
    pub const fn new(min: Vec3<f32>, max: Vec3<f32>) -> Self {
        Self { min, max }
    }

    /// Center of the box.
    #[must_use]
    pub fn center(&self) -> Vec3<f32> {
        Vec3::new(
            0.5 * (self.min.x + self.max.x),
            0.5 * (self.min.y + self.max.y),
            0.5 * (self.min.z + self.max.z),
        )
    }

    /// Side lengths `(dx, dy, dz)`.
    #[must_use]
    pub fn extent(&self) -> Vec3<f32> {
        Vec3::new(self.max.x - self.min.x, self.max.y - self.min.y, self.max.z - self.min.z)
    }
}

/// Oriented bounding box recovered from the principal axes of a cloud.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// Box center.
    pub center: Vec3<f32>,
    /// Column vectors are the three orthonormal box axes.
    pub axes: Mat3<f32>,
    /// Half side lengths along each axis.
    pub half_extents: Vec3<f32>,
}

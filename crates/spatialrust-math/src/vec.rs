use crate::{Real, Scalar};

/// 2D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec2<T: Scalar> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
}

/// 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec3<T: Scalar> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
    /// Z component.
    pub z: T,
}

/// 4D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec4<T: Scalar> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
    /// Z component.
    pub z: T,
    /// W component.
    pub w: T,
}

impl<T: Scalar> Vec3<T> {
    /// Creates a new 3D vector.
    #[must_use]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Scalar> core::ops::Add for Vec3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T: Scalar> core::ops::Sub for Vec3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T: Real> Vec3<T> {
    /// Dot product.
    #[must_use]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    /// Returns a unit vector when length is non-zero.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == T::default() {
            return self;
        }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }
}

#[cfg(test)]
mod tests {
    use super::Vec3;

    #[test]
    fn vec3_dot_and_cross() {
        let a = Vec3::new(1.0_f32, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0_f32, 0.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }
}

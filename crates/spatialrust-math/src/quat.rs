use crate::{Mat3, Real, Vec3};

/// Unit quaternion representing a 3D rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quat<T: Real> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
    /// Z component.
    pub z: T,
    /// W (scalar) component.
    pub w: T,
}

impl<T: Real> Quat<T> {
    /// Creates a quaternion from components.
    #[must_use]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }
}

impl Quat<f32> {
    /// Identity rotation for `f32`.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Normalizes the quaternion.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len == 0.0 {
            return Self::identity();
        }
        Self { x: self.x / len, y: self.y / len, z: self.z / len, w: self.w / len }
    }

    /// Creates a quaternion from an axis-angle representation.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3<f32>, angle: f32) -> Self {
        let axis = axis.normalize();
        let half = angle * 0.5;
        let s = half.sin();
        Self::new(axis.x * s, axis.y * s, axis.z * s, half.cos()).normalize()
    }

    /// Converts the quaternion to a rotation matrix.
    #[must_use]
    pub fn to_mat3(self) -> Mat3<f32> {
        let q = self.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        Mat3::from_rows(
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)],
            [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)],
            [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)],
        )
        .transpose()
    }

    /// Hamilton product.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
        .normalize()
    }
}

impl Quat<f64> {
    /// Identity rotation for `f64`.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Normalizes the quaternion.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len == 0.0 {
            return Self::identity();
        }
        Self { x: self.x / len, y: self.y / len, z: self.z / len, w: self.w / len }
    }

    /// Converts the quaternion to a rotation matrix.
    #[must_use]
    pub fn to_mat3(self) -> Mat3<f64> {
        let q = self.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        Mat3::from_rows(
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)],
            [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)],
            [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)],
        )
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::{Quat, Vec3};
    use crate::tolerance::{approx_eq, f32_eps};

    #[test]
    fn quat_identity_to_mat3() {
        let m = Quat::<f32>::identity().to_mat3();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let out = m.mul_vec3(v);
        assert!(approx_eq(out.x, v.x, f32_eps()));
        assert!(approx_eq(out.y, v.y, f32_eps()));
        assert!(approx_eq(out.z, v.z, f32_eps()));
    }

    #[test]
    fn quat_axis_angle_90deg_z() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let m = q.to_mat3();
        let out = m.mul_vec3(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(out.x, 0.0, 1e-5));
        assert!(approx_eq(out.y, 1.0, 1e-5));
    }
}

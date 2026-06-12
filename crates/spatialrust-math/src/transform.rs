use crate::{Mat4, Quat, Real, Vec3};

/// Rigid transform represented as a 4x4 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform3<T: Real> {
    matrix: Mat4<T>,
}

/// Proper rigid transform: rotation + translation without scale/shear.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Isometry3<T: Real> {
    rotation: Quat<T>,
    translation: Vec3<T>,
}

/// Trait for types that can transform 3D points.
pub trait TransformPoint<T: Real> {
    /// Transforms a point.
    fn transform_point(&self, point: Vec3<T>) -> Vec3<T>;

    /// Transforms a direction vector without translation.
    fn transform_vector(&self, vector: Vec3<T>) -> Vec3<T>;
}

impl Transform3<f32> {
    /// Creates a transform from a 4x4 matrix.
    #[must_use]
    pub const fn from_matrix(matrix: Mat4<f32>) -> Self {
        Self { matrix }
    }

    /// Identity transform.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_matrix(Mat4::<f32>::identity())
    }

    /// Returns the underlying matrix.
    #[must_use]
    pub const fn matrix(&self) -> Mat4<f32> {
        self.matrix
    }
}

impl Transform3<f64> {
    /// Creates a transform from a 4x4 matrix.
    #[must_use]
    pub const fn from_matrix(matrix: Mat4<f64>) -> Self {
        Self { matrix }
    }

    /// Identity transform.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_matrix(Mat4::<f64>::identity())
    }
}

impl TransformPoint<f32> for Transform3<f32> {
    fn transform_point(&self, point: Vec3<f32>) -> Vec3<f32> {
        self.matrix.transform_point(point)
    }

    fn transform_vector(&self, vector: Vec3<f32>) -> Vec3<f32> {
        self.matrix.transform_vector(vector)
    }
}

impl Isometry3<f32> {
    /// Identity isometry.
    #[must_use]
    pub fn identity() -> Self {
        Self::new(Quat::<f32>::identity(), Vec3::new(0.0, 0.0, 0.0))
    }

    /// Creates an isometry from rotation and translation.
    #[must_use]
    pub const fn new(rotation: Quat<f32>, translation: Vec3<f32>) -> Self {
        Self { rotation, translation }
    }

    /// Returns the rotation component.
    #[must_use]
    pub const fn rotation(&self) -> Quat<f32> {
        self.rotation
    }

    /// Returns the translation component.
    #[must_use]
    pub const fn translation(&self) -> Vec3<f32> {
        self.translation
    }

    /// Converts the isometry to a 4x4 matrix.
    #[must_use]
    pub fn to_mat4(self) -> Mat4<f32> {
        Mat4::<f32>::from_rotation_translation(self.rotation.to_mat3(), self.translation)
    }

    /// Composes two isometries.
    #[must_use]
    pub fn compose(self, other: Self) -> Self {
        let rotation = self.rotation.mul(other.rotation);
        let translation = self.rotation.to_mat3().mul_vec3(other.translation) + self.translation;
        Self { rotation, translation }
    }

    /// Returns the inverse isometry.
    #[must_use]
    pub fn inverse(self) -> Self {
        let inv_rotation =
            Quat::new(-self.rotation.x, -self.rotation.y, -self.rotation.z, self.rotation.w)
                .normalize();
        let inv_translation = inv_rotation.to_mat3().mul_vec3(Vec3::new(
            -self.translation.x,
            -self.translation.y,
            -self.translation.z,
        ));
        Self { rotation: inv_rotation, translation: inv_translation }
    }
}

impl TransformPoint<f32> for Isometry3<f32> {
    fn transform_point(&self, point: Vec3<f32>) -> Vec3<f32> {
        self.rotation.to_mat3().mul_vec3(point) + self.translation
    }

    fn transform_vector(&self, vector: Vec3<f32>) -> Vec3<f32> {
        self.rotation.to_mat3().mul_vec3(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::{Isometry3, TransformPoint};
    use crate::tolerance::{approx_eq, f32_eps};
    use crate::{Quat, Vec3};

    #[test]
    fn isometry_compose_and_inverse() {
        let a = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.5),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let b = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.0, 2.0, 0.0));
        let composed = a.compose(b);
        let point = Vec3::new(1.0, 1.0, 0.0);
        let restored = composed.compose(composed.inverse()).transform_point(point);
        assert!(approx_eq(restored.x, point.x, f32_eps()));
        assert!(approx_eq(restored.y, point.y, f32_eps()));
        assert!(approx_eq(restored.z, point.z, f32_eps()));
    }
}

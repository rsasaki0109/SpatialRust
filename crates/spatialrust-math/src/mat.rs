use crate::{Scalar, Vec3};

/// 3x3 matrix stored in row-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat3<T: Scalar> {
    /// Row-major matrix elements.
    pub m: [[T; 3]; 3],
}

/// 4x4 matrix stored in row-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat4<T: Scalar> {
    /// Row-major matrix elements.
    pub m: [[T; 4]; 4],
}

impl<T: Scalar> Mat3<T> {
    /// Creates a matrix from row vectors.
    #[must_use]
    pub const fn from_rows(row0: [T; 3], row1: [T; 3], row2: [T; 3]) -> Self {
        Self { m: [row0, row1, row2] }
    }
}

impl Mat3<f32> {
    /// Identity matrix for `f32`.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    }

    /// Transposed matrix.
    #[must_use]
    pub fn transpose(self) -> Self {
        Self::from_rows(
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        )
    }

    /// Matrix-vector multiplication.
    #[must_use]
    pub fn mul_vec3(self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    /// Matrix multiplication.
    #[must_use]
    pub fn mul_mat3(self, other: Self) -> Self {
        Self::from_rows(
            [
                self.m[0][0] * other.m[0][0]
                    + self.m[0][1] * other.m[1][0]
                    + self.m[0][2] * other.m[2][0],
                self.m[0][0] * other.m[0][1]
                    + self.m[0][1] * other.m[1][1]
                    + self.m[0][2] * other.m[2][1],
                self.m[0][0] * other.m[0][2]
                    + self.m[0][1] * other.m[1][2]
                    + self.m[0][2] * other.m[2][2],
            ],
            [
                self.m[1][0] * other.m[0][0]
                    + self.m[1][1] * other.m[1][0]
                    + self.m[1][2] * other.m[2][0],
                self.m[1][0] * other.m[0][1]
                    + self.m[1][1] * other.m[1][1]
                    + self.m[1][2] * other.m[2][1],
                self.m[1][0] * other.m[0][2]
                    + self.m[1][1] * other.m[1][2]
                    + self.m[1][2] * other.m[2][2],
            ],
            [
                self.m[2][0] * other.m[0][0]
                    + self.m[2][1] * other.m[1][0]
                    + self.m[2][2] * other.m[2][0],
                self.m[2][0] * other.m[0][1]
                    + self.m[2][1] * other.m[1][1]
                    + self.m[2][2] * other.m[2][1],
                self.m[2][0] * other.m[0][2]
                    + self.m[2][1] * other.m[1][2]
                    + self.m[2][2] * other.m[2][2],
            ],
        )
    }
}

impl Mat3<f64> {
    /// Identity matrix for `f64`.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    }

    /// Transposed matrix.
    #[must_use]
    pub fn transpose(self) -> Self {
        Self::from_rows(
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        )
    }

    /// Matrix-vector multiplication.
    #[must_use]
    pub fn mul_vec3(self, v: Vec3<f64>) -> Vec3<f64> {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }
}

impl<T: Scalar> Mat4<T> {
    /// Creates a matrix from row vectors.
    #[must_use]
    pub const fn from_rows(row0: [T; 4], row1: [T; 4], row2: [T; 4], row3: [T; 4]) -> Self {
        Self { m: [row0, row1, row2, row3] }
    }
}

impl Mat4<f32> {
    /// Identity matrix for `f32`.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_rows(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Homogeneous point transform.
    #[must_use]
    pub fn transform_point(self, point: Vec3<f32>) -> Vec3<f32> {
        let x =
            self.m[0][0] * point.x + self.m[0][1] * point.y + self.m[0][2] * point.z + self.m[0][3];
        let y =
            self.m[1][0] * point.x + self.m[1][1] * point.y + self.m[1][2] * point.z + self.m[1][3];
        let z =
            self.m[2][0] * point.x + self.m[2][1] * point.y + self.m[2][2] * point.z + self.m[2][3];
        let w =
            self.m[3][0] * point.x + self.m[3][1] * point.y + self.m[3][2] * point.z + self.m[3][3];
        if w == 0.0 {
            return Vec3::new(x, y, z);
        }
        Vec3::new(x / w, y / w, z / w)
    }

    /// Homogeneous vector transform (ignores translation).
    #[must_use]
    pub fn transform_vector(self, vector: Vec3<f32>) -> Vec3<f32> {
        Vec3::new(
            self.m[0][0] * vector.x + self.m[0][1] * vector.y + self.m[0][2] * vector.z,
            self.m[1][0] * vector.x + self.m[1][1] * vector.y + self.m[1][2] * vector.z,
            self.m[2][0] * vector.x + self.m[2][1] * vector.y + self.m[2][2] * vector.z,
        )
    }

    /// Builds a rigid transform matrix from rotation and translation.
    #[must_use]
    pub fn from_rotation_translation(rotation: Mat3<f32>, translation: Vec3<f32>) -> Self {
        Self::from_rows(
            [rotation.m[0][0], rotation.m[0][1], rotation.m[0][2], translation.x],
            [rotation.m[1][0], rotation.m[1][1], rotation.m[1][2], translation.y],
            [rotation.m[2][0], rotation.m[2][1], rotation.m[2][2], translation.z],
            [0.0, 0.0, 0.0, 1.0],
        )
    }
}

impl Mat4<f64> {
    /// Identity matrix for `f64`.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_rows(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Homogeneous point transform.
    #[must_use]
    pub fn transform_point(self, point: Vec3<f64>) -> Vec3<f64> {
        let x =
            self.m[0][0] * point.x + self.m[0][1] * point.y + self.m[0][2] * point.z + self.m[0][3];
        let y =
            self.m[1][0] * point.x + self.m[1][1] * point.y + self.m[1][2] * point.z + self.m[1][3];
        let z =
            self.m[2][0] * point.x + self.m[2][1] * point.y + self.m[2][2] * point.z + self.m[2][3];
        let w =
            self.m[3][0] * point.x + self.m[3][1] * point.y + self.m[3][2] * point.z + self.m[3][3];
        if w == 0.0 {
            return Vec3::new(x, y, z);
        }
        Vec3::new(x / w, y / w, z / w)
    }

    /// Builds a rigid transform matrix from rotation and translation.
    #[must_use]
    pub fn from_rotation_translation(rotation: Mat3<f64>, translation: Vec3<f64>) -> Self {
        Self::from_rows(
            [rotation.m[0][0], rotation.m[0][1], rotation.m[0][2], translation.x],
            [rotation.m[1][0], rotation.m[1][1], rotation.m[1][2], translation.y],
            [rotation.m[2][0], rotation.m[2][1], rotation.m[2][2], translation.z],
            [0.0, 0.0, 0.0, 1.0],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Mat3, Mat4, Vec3};

    #[test]
    fn mat3_mul_vec3() {
        let rot_y: Mat3<f32> = Mat3::from_rows([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
        let v = Vec3::new(1.0_f32, 0.0, 0.0);
        let out = rot_y.mul_vec3(v);
        assert!((out.x - 0.0).abs() < 1e-6);
        assert!((out.z - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn mat4_transform_point() {
        let transform = Mat4::<f32>::from_rotation_translation(
            Mat3::<f32>::identity(),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let p = Vec3::new(0.0_f32, 0.0, 0.0);
        let out = transform.transform_point(p);
        assert!((out.x - 1.0).abs() < 1e-6);
        assert!((out.y - 2.0).abs() < 1e-6);
        assert!((out.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mat4_f64_roundtrip() {
        let m = Mat4::<f64>::identity();
        let p = Vec3::new(1.0_f64, 2.0, 3.0);
        let out = m.transform_point(p);
        assert!((out.x - 1.0).abs() < 1e-12);
    }
}

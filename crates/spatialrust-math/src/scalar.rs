/// Numeric types supported by SpatialRust math primitives.
pub trait Scalar:
    Copy
    + Clone
    + Default
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}

/// Floating-point scalar used by spatial algorithms.
pub trait Real: Scalar {
    /// Absolute value.
    fn abs(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;
}

impl Real for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Real for f64 {
    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// Scalar component supported by generic CPU image kernels.
pub trait PixelComponent: Copy + Send + Sync + 'static {
    /// Converts a scalar into the kernel accumulator representation.
    fn to_f64(self) -> f64;
    /// Converts a finite accumulator value back into the scalar dtype.
    fn from_f64(value: f64) -> Self;
}

impl PixelComponent for u8 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value.round().clamp(0.0, 255.0) as Self
    }
}

impl PixelComponent for u16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value.round().clamp(0.0, f64::from(u16::MAX)) as Self
    }
}

impl PixelComponent for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as Self
    }
}

impl PixelComponent for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}

use crate::{VizError, VizResult};

/// Linear RGBA color with components in the inclusive range `0.0..=1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearRgba {
    /// Red component.
    pub red: f32,
    /// Green component.
    pub green: f32,
    /// Blue component.
    pub blue: f32,
    /// Alpha component.
    pub alpha: f32,
}

impl LinearRgba {
    /// Opaque white.
    pub const WHITE: Self = Self { red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0 };
    /// Opaque black.
    pub const BLACK: Self = Self { red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0 };

    /// Creates a validated linear color.
    pub fn try_new(red: f32, green: f32, blue: f32, alpha: f32) -> VizResult<Self> {
        let components = [red, green, blue, alpha];
        if components.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return Err(VizError::InvalidStyle(
                "RGBA components must be finite and in 0.0..=1.0".into(),
            ));
        }
        Ok(Self { red, green, blue, alpha })
    }
}

impl Default for LinearRgba {
    fn default() -> Self {
        Self::WHITE
    }
}

#[cfg(test)]
mod tests {
    use super::LinearRgba;

    #[test]
    fn rejects_non_finite_and_out_of_range_components() {
        assert!(LinearRgba::try_new(f32::NAN, 0.0, 0.0, 1.0).is_err());
        assert!(LinearRgba::try_new(1.1, 0.0, 0.0, 1.0).is_err());
        assert_eq!(LinearRgba::try_new(0.1, 0.2, 0.3, 0.4).unwrap().alpha, 0.4);
    }
}

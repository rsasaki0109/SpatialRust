use crate::{LinearRgba, VizError, VizResult};

/// Built-in scalar color maps with stable names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorMap {
    /// Perceptually uniform purple-to-yellow map.
    #[default]
    Viridis,
    /// Dark-blue through red to yellow map.
    Turbo,
    /// Monochrome black-to-white map.
    Gray,
}

/// Color source for point primitives.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointColor {
    /// One color for every point.
    Uniform(LinearRgba),
    /// Use the primitive's borrowed RGB columns.
    Rgb,
    /// Map the primitive's borrowed scalar column through a color map.
    Scalar {
        /// Inclusive lower display bound.
        min: f32,
        /// Inclusive upper display bound.
        max: f32,
        /// Color map.
        map: ColorMap,
    },
}

/// Point rendering style.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointStyle {
    /// Point diameter in logical pixels.
    pub size: f32,
    /// Point color source.
    pub color: PointColor,
}

impl PointStyle {
    /// Creates a validated point style.
    pub fn try_new(size: f32, color: PointColor) -> VizResult<Self> {
        if !size.is_finite() || size <= 0.0 {
            return Err(VizError::InvalidStyle("point size must be finite and positive".into()));
        }
        if let PointColor::Scalar { min, max, .. } = &color {
            if !min.is_finite() || !max.is_finite() || max <= min {
                return Err(VizError::InvalidStyle(
                    "scalar display range must be finite with min < max".into(),
                ));
            }
        }
        Ok(Self { size, color })
    }
}

impl Default for PointStyle {
    fn default() -> Self {
        Self { size: 1.0, color: PointColor::Uniform(LinearRgba::WHITE) }
    }
}

/// Style applied to a visual primitive.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VisualStyle {
    /// Point-specific style.
    Points(PointStyle),
    /// Uniform color for lines or triangle wireframes.
    Uniform(LinearRgba),
}

#[cfg(test)]
mod tests {
    use super::{ColorMap, PointColor, PointStyle};

    #[test]
    fn validates_point_size_and_scalar_range() {
        assert!(PointStyle::try_new(0.0, PointColor::Rgb).is_err());
        assert!(PointStyle::try_new(
            2.0,
            PointColor::Scalar { min: 1.0, max: 1.0, map: ColorMap::Viridis }
        )
        .is_err());
    }
}

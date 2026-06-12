use crate::{SpatialError, SpatialResult};

/// Supported scalar dtypes for point fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 32-bit signed integer.
    I32,
    /// 16-bit float.
    F16,
}

impl DType {
    /// Returns the size of one scalar component in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 | Self::I32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Semantic meaning of a point field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FieldSemantic {
    /// Position X coordinate.
    PositionX,
    /// Position Y coordinate.
    PositionY,
    /// Position Z coordinate.
    PositionZ,
    /// Surface normal X component.
    NormalX,
    /// Surface normal Y component.
    NormalY,
    /// Surface normal Z component.
    NormalZ,
    /// LiDAR intensity.
    Intensity,
    /// Red color channel.
    ColorR,
    /// Green color channel.
    ColorG,
    /// Blue color channel.
    ColorB,
    /// Estimated curvature.
    Curvature,
    /// LiDAR ring index.
    Ring,
    /// Per-point time offset.
    TimeOffset,
    /// Segmentation or semantic label.
    Label,
    /// Learned embedding component.
    Embedding,
    /// Unknown or vendor-specific field.
    Unknown,
}

/// One column in a point cloud schema.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointField {
    /// Field name.
    pub name: String,
    /// Semantic meaning.
    pub semantic: FieldSemantic,
    /// Scalar dtype.
    pub dtype: DType,
    /// Number of scalar components.
    pub components: usize,
}

impl PointField {
    /// Creates a scalar field.
    #[must_use]
    pub fn scalar(name: impl Into<String>, semantic: FieldSemantic, dtype: DType) -> Self {
        Self { name: name.into(), semantic, dtype, components: 1 }
    }

    /// Total byte size of one point value for this field.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.dtype.size_bytes() * self.components
    }
}

/// Schema describing the columns of a point cloud.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointSchema {
    fields: Vec<PointField>,
}

impl PointSchema {
    /// Creates an empty schema.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a field to the schema.
    pub fn with_field(mut self, field: PointField) -> Self {
        self.fields.push(field);
        self
    }

    /// Returns the schema fields.
    #[must_use]
    pub fn fields(&self) -> &[PointField] {
        &self.fields
    }

    /// Returns the number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns whether the schema has no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Finds a field by semantic.
    #[must_use]
    pub fn find_semantic(&self, semantic: FieldSemantic) -> Option<&PointField> {
        self.fields.iter().find(|field| field.semantic == semantic)
    }

    /// Validates that required position fields exist exactly once.
    pub fn validate_positions(&self) -> SpatialResult<()> {
        for semantic in
            [FieldSemantic::PositionX, FieldSemantic::PositionY, FieldSemantic::PositionZ]
        {
            let count = self.fields.iter().filter(|field| field.semantic == semantic).count();
            if count != 1 {
                return Err(SpatialError::SchemaValidation(format!(
                    "expected exactly one {semantic:?} field, found {count}"
                )));
            }
        }
        Ok(())
    }
}

/// Standard schemas used by typed views and IO adapters.
pub struct StandardSchemas;

impl StandardSchemas {
    /// `PointXYZ` schema.
    #[must_use]
    pub fn point_xyz() -> PointSchema {
        PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
    }

    /// `PointXYZI` schema.
    #[must_use]
    pub fn point_xyzi() -> PointSchema {
        Self::point_xyz().with_field(PointField::scalar(
            "intensity",
            FieldSemantic::Intensity,
            DType::F32,
        ))
    }

    /// `PointXYZRGB` schema.
    #[must_use]
    pub fn point_xyzrgb() -> PointSchema {
        Self::point_xyz()
            .with_field(PointField::scalar("r", FieldSemantic::ColorR, DType::U8))
            .with_field(PointField::scalar("g", FieldSemantic::ColorG, DType::U8))
            .with_field(PointField::scalar("b", FieldSemantic::ColorB, DType::U8))
    }

    /// `PointXYZINormal` schema.
    #[must_use]
    pub fn point_xyzinormal() -> PointSchema {
        Self::point_xyzi()
            .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
            .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
            .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldSemantic, StandardSchemas};

    #[test]
    fn standard_xyz_schema_validates() {
        let schema = StandardSchemas::point_xyz();
        schema.validate_positions().unwrap();
        assert_eq!(schema.len(), 3);
        assert!(schema.find_semantic(FieldSemantic::PositionX).is_some());
    }
}

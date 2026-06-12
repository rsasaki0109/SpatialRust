use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema};

use crate::error::{pcd_format, IoError};

/// PCD scalar type token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcdType {
    /// Signed integer.
    I,
    /// Unsigned integer.
    U,
    /// Floating point.
    F,
}

/// One PCD field specification from the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcdFieldSpec {
    /// Field name.
    pub name: String,
    /// Size in bytes of one scalar component.
    pub size: usize,
    /// Scalar type token.
    pub kind: PcdType,
    /// Number of scalar components.
    pub count: usize,
}

impl PcdFieldSpec {
    /// Returns the total byte size of this field for one point.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.size * self.count
    }

    /// Maps this field to a SpatialRust dtype.
    pub fn dtype(&self) -> Result<DType, IoError> {
        if self.count != 1 {
            return Err(pcd_format(format!(
                "field `{}` has unsupported COUNT {}",
                self.name, self.count
            )));
        }
        match (self.kind, self.size) {
            (PcdType::F, 4) => Ok(DType::F32),
            (PcdType::F, 8) => Ok(DType::F64),
            (PcdType::I, 4) => Ok(DType::I32),
            (PcdType::U, 1) => Ok(DType::U8),
            (PcdType::U, 2) => Ok(DType::U16),
            (PcdType::U, 4) => Ok(DType::U32),
            _ => Err(pcd_format(format!(
                "unsupported PCD field `{}` with TYPE {:?} and SIZE {}",
                self.name, self.kind, self.size
            ))),
        }
    }
}

/// Maps a PCD field name to a SpatialRust semantic.
#[must_use]
pub fn infer_field_semantic(name: &str) -> FieldSemantic {
    match name.to_ascii_lowercase().as_str() {
        "x" => FieldSemantic::PositionX,
        "y" => FieldSemantic::PositionY,
        "z" => FieldSemantic::PositionZ,
        "normal_x" | "nx" => FieldSemantic::NormalX,
        "normal_y" | "ny" => FieldSemantic::NormalY,
        "normal_z" | "nz" => FieldSemantic::NormalZ,
        "intensity" | "i" => FieldSemantic::Intensity,
        "curvature" => FieldSemantic::Curvature,
        "ring" => FieldSemantic::Ring,
        "timestamp" | "t" | "time_offset" => FieldSemantic::TimeOffset,
        "label" => FieldSemantic::Label,
        "r" | "red" => FieldSemantic::ColorR,
        "g" | "green" => FieldSemantic::ColorG,
        "b" | "blue" => FieldSemantic::ColorB,
        _ => FieldSemantic::Unknown,
    }
}

/// Builds a SpatialRust schema from PCD field specs.
pub fn schema_from_pcd_fields(fields: &[PcdFieldSpec]) -> Result<PointSchema, IoError> {
    let mut schema = PointSchema::new();
    for field in fields {
        if field.name.eq_ignore_ascii_case("rgb") {
            schema = schema
                .with_field(PointField::scalar("r", FieldSemantic::ColorR, DType::U8))
                .with_field(PointField::scalar("g", FieldSemantic::ColorG, DType::U8))
                .with_field(PointField::scalar("b", FieldSemantic::ColorB, DType::U8));
            continue;
        }
        let dtype = field.dtype()?;
        let semantic = infer_field_semantic(&field.name);
        schema = schema.with_field(PointField::scalar(field.name.clone(), semantic, dtype));
    }
    Ok(schema)
}

#[cfg(test)]
mod tests {
    use super::{infer_field_semantic, schema_from_pcd_fields, PcdFieldSpec, PcdType};
    use spatialrust_core::FieldSemantic;

    #[test]
    fn maps_xyz_semantics() {
        assert_eq!(infer_field_semantic("x"), FieldSemantic::PositionX);
        assert_eq!(infer_field_semantic("intensity"), FieldSemantic::Intensity);
    }

    #[test]
    fn expands_rgb_field() {
        let fields = vec![PcdFieldSpec { name: "rgb".into(), size: 4, kind: PcdType::F, count: 1 }];
        let schema = schema_from_pcd_fields(&fields).unwrap();
        assert_eq!(schema.len(), 3);
    }
}

use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema};

use crate::error::IoError;
use crate::ply::header::{PlyProperty, PlyPropertyKind};

/// Maps a PLY property name to a SpatialRust semantic.
#[must_use]
pub fn infer_property_semantic(name: &str) -> FieldSemantic {
    match name.to_ascii_lowercase().as_str() {
        "x" => FieldSemantic::PositionX,
        "y" => FieldSemantic::PositionY,
        "z" => FieldSemantic::PositionZ,
        "nx" | "normal_x" => FieldSemantic::NormalX,
        "ny" | "normal_y" => FieldSemantic::NormalY,
        "nz" | "normal_z" => FieldSemantic::NormalZ,
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

impl PlyProperty {
    /// Maps this property to a SpatialRust dtype.
    pub fn dtype(&self) -> Result<DType, IoError> {
        match self.kind {
            PlyPropertyKind::Float => Ok(DType::F32),
            PlyPropertyKind::Double => Ok(DType::F64),
            PlyPropertyKind::Int => Ok(DType::I32),
            PlyPropertyKind::UInt => Ok(DType::U32),
            PlyPropertyKind::UChar => Ok(DType::U8),
            PlyPropertyKind::UShort => Ok(DType::U16),
            PlyPropertyKind::Char | PlyPropertyKind::Short => Ok(DType::I32),
        }
    }
}

/// Builds a SpatialRust schema from PLY vertex properties.
pub fn schema_from_ply_properties(properties: &[PlyProperty]) -> Result<PointSchema, IoError> {
    let mut schema = PointSchema::new();
    for property in properties {
        let dtype = property.dtype()?;
        let semantic = infer_property_semantic(&property.name);
        schema = schema.with_field(PointField::scalar(property.name.clone(), semantic, dtype));
    }
    Ok(schema)
}

/// Builds a PLY property description from a SpatialRust field.
pub fn ply_property_from_field(field: &PointField) -> Result<PlyProperty, IoError> {
    let name = match field.semantic {
        FieldSemantic::PositionX => "x",
        FieldSemantic::PositionY => "y",
        FieldSemantic::PositionZ => "z",
        FieldSemantic::NormalX => "nx",
        FieldSemantic::NormalY => "ny",
        FieldSemantic::NormalZ => "nz",
        FieldSemantic::Intensity => "intensity",
        FieldSemantic::Curvature => "curvature",
        FieldSemantic::Ring => "ring",
        FieldSemantic::TimeOffset => "timestamp",
        FieldSemantic::Label => "label",
        FieldSemantic::ColorR => "red",
        FieldSemantic::ColorG => "green",
        FieldSemantic::ColorB => "blue",
        _ => field.name.as_str(),
    }
    .to_owned();

    let kind = match field.dtype {
        DType::F32 | DType::F16 => PlyPropertyKind::Float,
        DType::F64 => PlyPropertyKind::Double,
        DType::I32 => PlyPropertyKind::Int,
        DType::U32 => PlyPropertyKind::UInt,
        DType::U8 => PlyPropertyKind::UChar,
        DType::U16 => PlyPropertyKind::UShort,
    };

    Ok(PlyProperty { name, kind })
}

#[cfg(test)]
mod tests {
    use super::{infer_property_semantic, schema_from_ply_properties};
    use crate::ply::header::{PlyProperty, PlyPropertyKind};
    use spatialrust_core::FieldSemantic;

    #[test]
    fn maps_xyz_semantics() {
        assert_eq!(infer_property_semantic("x"), FieldSemantic::PositionX);
        assert_eq!(infer_property_semantic("red"), FieldSemantic::ColorR);
    }

    #[test]
    fn builds_xyz_schema() {
        let properties = vec![
            PlyProperty { name: "x".into(), kind: PlyPropertyKind::Float },
            PlyProperty { name: "y".into(), kind: PlyPropertyKind::Float },
            PlyProperty { name: "z".into(), kind: PlyPropertyKind::Float },
        ];
        let schema = schema_from_ply_properties(&properties).unwrap();
        assert_eq!(schema.len(), 3);
    }
}

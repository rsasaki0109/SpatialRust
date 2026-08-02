//! Schema identity, versioning, and compatibility.

use spatialrust_core::{PointField, PointSchema};

use crate::{RecordsError, RecordsResult};

/// Stable schema family identifier independent of field order serialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SchemaId(pub String);

impl SchemaId {
    /// Creates a schema identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrows the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SchemaId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for SchemaId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Semantic schema version used for evolution checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion {
    /// Breaking-change counter.
    pub major: u32,
    /// Compatible additive counter.
    pub minor: u32,
}

impl SchemaVersion {
    /// Creates a schema version.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Named, versioned [`PointSchema`] contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaDescriptor {
    /// Family id shared across compatible revisions.
    pub id: SchemaId,
    /// Revision selected for this descriptor.
    pub version: SchemaVersion,
    /// Concrete columns.
    pub schema: PointSchema,
}

impl SchemaDescriptor {
    /// Creates a validated schema descriptor.
    pub fn try_new(
        id: impl Into<SchemaId>,
        version: SchemaVersion,
        schema: PointSchema,
    ) -> RecordsResult<Self> {
        schema.validate_positions().map_err(RecordsError::from)?;
        if schema.fields().is_empty() {
            return Err(RecordsError::InvalidConfiguration("schema has no fields".into()));
        }
        Ok(Self { id: id.into(), version, schema })
    }

    /// Returns the point schema.
    #[must_use]
    pub fn point_schema(&self) -> &PointSchema {
        &self.schema
    }
}

/// High-level comparison result between two schemas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompatVerdict {
    /// Exact field set, order, dtypes, and semantics.
    Identical,
    /// Reader can consume a richer producer by dropping additive fields.
    BackwardCompatible,
    /// Reader can accept a poorer producer by filling defaults.
    ForwardCompatible,
    /// Major version or field contracts conflict.
    Incompatible,
}

/// Detailed schema comparison report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaCompatReport {
    /// Overall verdict.
    pub verdict: CompatVerdict,
    /// Fields present only in the expected/target schema.
    pub missing_in_actual: Vec<String>,
    /// Fields present only in the actual/source schema.
    pub extra_in_actual: Vec<String>,
    /// Same name but mismatched dtype/semantic/components.
    pub conflicting: Vec<String>,
}

/// Compares `actual` against `expected` using id/version plus field contracts.
pub fn compare_schemas(
    expected: &SchemaDescriptor,
    actual: &SchemaDescriptor,
) -> SchemaCompatReport {
    if expected.id != actual.id {
        return SchemaCompatReport {
            verdict: CompatVerdict::Incompatible,
            missing_in_actual: Vec::new(),
            extra_in_actual: Vec::new(),
            conflicting: vec![format!(
                "schema id `{}` vs `{}`",
                expected.id.as_str(),
                actual.id.as_str()
            )],
        };
    }
    if expected.version.major != actual.version.major {
        return SchemaCompatReport {
            verdict: CompatVerdict::Incompatible,
            missing_in_actual: Vec::new(),
            extra_in_actual: Vec::new(),
            conflicting: vec![format!(
                "major version {} vs {}",
                expected.version.major, actual.version.major
            )],
        };
    }

    let mut missing_in_actual = Vec::new();
    let mut conflicting = Vec::new();
    for field in expected.schema.fields() {
        match actual.schema.fields().iter().find(|candidate| candidate.name == field.name) {
            None => missing_in_actual.push(field.name.clone()),
            Some(actual_field) if !fields_compatible(field, actual_field) => {
                conflicting.push(field.name.clone());
            }
            Some(_) => {}
        }
    }
    let mut extra_in_actual = Vec::new();
    for field in actual.schema.fields() {
        if !expected.schema.fields().iter().any(|candidate| candidate.name == field.name) {
            extra_in_actual.push(field.name.clone());
        }
    }

    let verdict = if missing_in_actual.is_empty()
        && extra_in_actual.is_empty()
        && conflicting.is_empty()
        && expected.schema.fields() == actual.schema.fields()
    {
        CompatVerdict::Identical
    } else if !conflicting.is_empty() {
        CompatVerdict::Incompatible
    } else if missing_in_actual.is_empty() && !extra_in_actual.is_empty() {
        CompatVerdict::BackwardCompatible
    } else if !missing_in_actual.is_empty() && extra_in_actual.is_empty() {
        CompatVerdict::ForwardCompatible
    } else if missing_in_actual.is_empty() && extra_in_actual.is_empty() {
        // Same fields, possibly different order.
        CompatVerdict::Identical
    } else {
        CompatVerdict::Incompatible
    };

    SchemaCompatReport { verdict, missing_in_actual, extra_in_actual, conflicting }
}

fn fields_compatible(expected: &PointField, actual: &PointField) -> bool {
    expected.dtype == actual.dtype
        && expected.semantic == actual.semantic
        && expected.components == actual.components
}

#[cfg(test)]
mod tests {
    use super::{compare_schemas, CompatVerdict, SchemaDescriptor, SchemaVersion};
    use spatialrust_core::{DType, FieldSemantic, PointField, StandardSchemas};

    #[test]
    fn additive_field_is_backward_compatible() {
        let base = SchemaDescriptor::try_new(
            "point",
            SchemaVersion::new(1, 0),
            StandardSchemas::point_xyz(),
        )
        .unwrap();
        let richer = SchemaDescriptor::try_new(
            "point",
            SchemaVersion::new(1, 1),
            StandardSchemas::point_xyzi(),
        )
        .unwrap();
        let report = compare_schemas(&base, &richer);
        assert_eq!(report.verdict, CompatVerdict::BackwardCompatible);
        assert_eq!(report.extra_in_actual, vec!["intensity".to_owned()]);
    }

    #[test]
    fn dtype_conflict_is_incompatible() {
        let expected = SchemaDescriptor::try_new(
            "point",
            SchemaVersion::new(1, 0),
            StandardSchemas::point_xyz(),
        )
        .unwrap();
        let schema = StandardSchemas::point_xyz();
        // Replace x with f64 for conflict.
        let fields = schema.fields().to_vec();
        let mut rebuilt = spatialrust_core::PointSchema::new();
        for field in fields {
            if field.name == "x" {
                rebuilt = rebuilt.with_field(PointField::scalar(
                    "x",
                    FieldSemantic::PositionX,
                    DType::F64,
                ));
            } else {
                rebuilt = rebuilt.with_field(field);
            }
        }
        let actual = SchemaDescriptor::try_new("point", SchemaVersion::new(1, 0), rebuilt).unwrap();
        assert_eq!(compare_schemas(&expected, &actual).verdict, CompatVerdict::Incompatible);
    }
}

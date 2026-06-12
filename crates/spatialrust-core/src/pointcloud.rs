use crate::{
    CpuDevice, DType, Device, PointBuffer, PointBufferSet, PointField, PointSchema, SpatialError,
    SpatialMetadata, SpatialResult, StandardSchemas,
};

/// Schema-aware columnar point cloud stored on a device.
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloud {
    schema: PointSchema,
    buffers: PointBufferSet,
    len: usize,
    metadata: SpatialMetadata,
    device: CpuDevice,
}

/// Builds point clouds field-by-field.
#[derive(Clone, Debug, Default)]
pub struct PointCloudBuilder {
    schema: PointSchema,
    buffers: PointBufferSet,
    metadata: SpatialMetadata,
}

impl PointCloud {
    /// Creates an empty point cloud with the given schema.
    #[must_use]
    pub fn with_schema(schema: PointSchema) -> Self {
        Self {
            schema,
            buffers: PointBufferSet::new(),
            len: 0,
            metadata: SpatialMetadata::default(),
            device: CpuDevice,
        }
    }

    /// Creates an empty `PointXYZ` cloud.
    #[must_use]
    pub fn xyz() -> Self {
        Self::with_schema(StandardSchemas::point_xyz())
    }

    /// Returns the point cloud schema.
    #[must_use]
    pub fn schema(&self) -> &PointSchema {
        &self.schema
    }

    /// Returns the number of points.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the cloud contains no points.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns spatial metadata.
    #[must_use]
    pub fn metadata(&self) -> &SpatialMetadata {
        &self.metadata
    }

    /// Returns the storage device.
    #[must_use]
    pub fn device(&self) -> &dyn Device {
        &self.device
    }

    /// Returns read-only access to a field buffer by name.
    pub fn field(&self, name: &str) -> SpatialResult<&PointBuffer> {
        self.buffers.get(name).ok_or_else(|| SpatialError::MissingField(name.to_owned()))
    }

    /// Validates schema and buffer consistency.
    pub fn validate(&self) -> SpatialResult<()> {
        self.schema.validate_positions()?;
        for field in self.schema.fields() {
            let buffer = self.field(&field.name)?;
            if buffer.len() != self.len {
                return Err(SpatialError::BufferLengthMismatch {
                    expected: self.len,
                    found: buffer.len(),
                });
            }
            if buffer.dtype() != field.dtype && field.dtype != DType::F16 {
                return Err(SpatialError::SchemaValidation(format!(
                    "field `{}` dtype mismatch",
                    field.name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn from_builder(builder: PointCloudBuilder, len: usize) -> SpatialResult<Self> {
        let cloud = Self {
            schema: builder.schema,
            buffers: builder.buffers,
            len,
            metadata: builder.metadata,
            device: CpuDevice,
        };
        cloud.validate()?;
        Ok(cloud)
    }

    /// Constructs a point cloud from schema, column buffers, and metadata.
    pub fn try_from_parts(
        schema: PointSchema,
        buffers: PointBufferSet,
        metadata: SpatialMetadata,
    ) -> SpatialResult<Self> {
        if schema.is_empty() {
            return Ok(Self { schema, buffers, len: 0, metadata, device: CpuDevice });
        }

        let len = schema
            .fields()
            .first()
            .and_then(|field| buffers.get(&field.name))
            .map(|buffer| buffer.len())
            .ok_or_else(|| SpatialError::MissingField(schema.fields()[0].name.clone()))?;

        for field in schema.fields() {
            let buffer = buffers
                .get(&field.name)
                .ok_or_else(|| SpatialError::MissingField(field.name.clone()))?;
            if buffer.len() != len {
                return Err(SpatialError::BufferLengthMismatch {
                    expected: len,
                    found: buffer.len(),
                });
            }
        }

        let cloud = Self { schema, buffers, len, metadata, device: CpuDevice };
        cloud.validate()?;
        Ok(cloud)
    }
}

impl PointCloudBuilder {
    /// Creates a builder from a schema.
    #[must_use]
    pub fn new(schema: PointSchema) -> Self {
        Self { schema, ..Self::default() }
    }

    /// Creates a builder for the standard `PointXYZ` schema.
    #[must_use]
    pub fn xyz() -> Self {
        Self::new(StandardSchemas::point_xyz())
    }

    /// Sets spatial metadata.
    #[must_use]
    pub fn metadata(mut self, metadata: SpatialMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Appends one point by field values in schema order.
    ///
    /// Each field value must be a scalar convertible to the field dtype.
    pub fn push_point<I>(&mut self, values: I) -> SpatialResult<()>
    where
        I: IntoIterator<Item = f32>,
    {
        let values: Vec<f32> = values.into_iter().collect();
        if values.len() != self.schema.len() {
            return Err(SpatialError::InvalidArgument(format!(
                "expected {} field values, got {}",
                self.schema.len(),
                values.len()
            )));
        }

        let fields: Vec<PointField> = self.schema.fields().to_vec();
        for (field, value) in fields.iter().zip(values) {
            self.push_scalar(field, value)?;
        }
        Ok(())
    }

    /// Builds the final point cloud.
    pub fn build(self) -> SpatialResult<PointCloud> {
        let len = self
            .schema
            .fields()
            .first()
            .and_then(|field| self.buffers.get(&field.name))
            .map(|buffer| buffer.len())
            .unwrap_or(0);
        PointCloud::from_builder(self, len)
    }

    fn push_scalar(&mut self, field: &PointField, value: f32) -> SpatialResult<()> {
        let buffer = match self.buffers.get_mut(&field.name) {
            Some(buffer) => buffer,
            None => {
                let buffer = PointBuffer::with_capacity(field.dtype, 0);
                self.buffers.insert(field.name.clone(), buffer);
                self.buffers.get_mut(&field.name).expect("buffer inserted")
            }
        };

        match field.dtype {
            DType::F32 | DType::F16 => buffer.as_f32_mut()?.push(value),
            DType::F64 => match buffer {
                PointBuffer::F64(values) => values.push(f64::from(value)),
                _ => return Err(SpatialError::UnsupportedDType(field.dtype)),
            },
            DType::U8 => match buffer {
                PointBuffer::U8(values) => values.push(value.round() as u8),
                _ => return Err(SpatialError::UnsupportedDType(field.dtype)),
            },
            DType::U16 => match buffer {
                PointBuffer::U16(values) => values.push(value.round() as u16),
                _ => return Err(SpatialError::UnsupportedDType(field.dtype)),
            },
            DType::U32 => match buffer {
                PointBuffer::U32(values) => values.push(value.round() as u32),
                _ => return Err(SpatialError::UnsupportedDType(field.dtype)),
            },
            DType::I32 => match buffer {
                PointBuffer::I32(values) => values.push(value.round() as i32),
                _ => return Err(SpatialError::UnsupportedDType(field.dtype)),
            },
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PointCloudBuilder;
    use crate::{FieldSemantic, StandardSchemas};

    #[test]
    fn build_xyz_cloud() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        assert_eq!(cloud.len(), 2);
        assert!(cloud.validate().is_ok());
        let x = cloud.field("x").unwrap().as_f32().unwrap();
        assert_eq!(x, &[0.0, 1.0]);
    }

    #[test]
    fn standard_xyzi_has_intensity() {
        let schema = StandardSchemas::point_xyzi();
        assert!(schema.find_semantic(FieldSemantic::Intensity).is_some());
    }
}

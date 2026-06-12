use crate::{DType, PointField, SpatialError, SpatialResult};

/// Typed column buffer for one point field.
#[derive(Clone, Debug, PartialEq)]
pub enum PointBuffer {
    /// `f32` column.
    F32(Vec<f32>),
    /// `f64` column.
    F64(Vec<f64>),
    /// `u8` column.
    U8(Vec<u8>),
    /// `u16` column.
    U16(Vec<u16>),
    /// `u32` column.
    U32(Vec<u32>),
    /// `i32` column.
    I32(Vec<i32>),
}

impl PointBuffer {
    /// Creates an empty buffer for the given dtype.
    #[must_use]
    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        match dtype {
            DType::F32 | DType::F16 => Self::F32(Vec::with_capacity(capacity)),
            DType::F64 => Self::F64(Vec::with_capacity(capacity)),
            DType::U8 => Self::U8(Vec::with_capacity(capacity)),
            DType::U16 => Self::U16(Vec::with_capacity(capacity)),
            DType::U32 => Self::U32(Vec::with_capacity(capacity)),
            DType::I32 => Self::I32(Vec::with_capacity(capacity)),
        }
    }

    /// Creates a buffer from an existing `f32` vector.
    #[must_use]
    pub fn from_f32(values: Vec<f32>) -> Self {
        Self::F32(values)
    }

    /// Returns the number of elements in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }

    /// Returns whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the buffer dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
        }
    }

    /// Resizes the buffer to the requested number of elements.
    pub fn resize(&mut self, len: usize, field: &PointField) -> SpatialResult<()> {
        if field.dtype != self.dtype() && !(field.dtype == DType::F16 && self.dtype() == DType::F32)
        {
            return Err(SpatialError::UnsupportedDType(field.dtype));
        }
        match self {
            Self::F32(values) => values.resize(len, 0.0),
            Self::F64(values) => values.resize(len, 0.0),
            Self::U8(values) => values.resize(len, 0),
            Self::U16(values) => values.resize(len, 0),
            Self::U32(values) => values.resize(len, 0),
            Self::I32(values) => values.resize(len, 0),
        }
        Ok(())
    }

    /// Returns immutable access to an `f32` column.
    pub fn as_f32(&self) -> SpatialResult<&[f32]> {
        match self {
            Self::F32(values) => Ok(values),
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }

    /// Returns mutable access to an `f32` column.
    pub fn as_f32_mut(&mut self) -> SpatialResult<&mut Vec<f32>> {
        match self {
            Self::F32(values) => Ok(values),
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }

    /// Appends one `f32` value.
    pub fn push_f32(&mut self, value: f32) -> SpatialResult<()> {
        self.as_f32_mut()?.push(value);
        Ok(())
    }

    /// Appends one `f64` value.
    pub fn push_f64(&mut self, value: f64) -> SpatialResult<()> {
        match self {
            Self::F64(values) => {
                values.push(value);
                Ok(())
            }
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }

    /// Appends one `u8` value.
    pub fn push_u8(&mut self, value: u8) -> SpatialResult<()> {
        match self {
            Self::U8(values) => {
                values.push(value);
                Ok(())
            }
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }

    /// Appends one `u16` value.
    pub fn push_u16(&mut self, value: u16) -> SpatialResult<()> {
        match self {
            Self::U16(values) => {
                values.push(value);
                Ok(())
            }
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }

    /// Appends one `i32` value.
    pub fn push_i32(&mut self, value: i32) -> SpatialResult<()> {
        match self {
            Self::I32(values) => {
                values.push(value);
                Ok(())
            }
            _ => Err(SpatialError::UnsupportedDType(self.dtype())),
        }
    }
}

/// Collection of column buffers keyed by field name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointBufferSet {
    buffers: Vec<(String, PointBuffer)>,
}

impl PointBufferSet {
    /// Creates an empty buffer set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a named buffer.
    pub fn insert(&mut self, name: impl Into<String>, buffer: PointBuffer) {
        let name = name.into();
        if let Some((_, existing)) = self.buffers.iter_mut().find(|(key, _)| key == &name) {
            *existing = buffer;
        } else {
            self.buffers.push((name, buffer));
        }
    }

    /// Returns a buffer by field name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PointBuffer> {
        self.buffers.iter().find(|(key, _)| key == name).map(|(_, buffer)| buffer)
    }

    /// Returns a mutable buffer by field name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut PointBuffer> {
        self.buffers.iter_mut().find(|(key, _)| key == name).map(|(_, buffer)| buffer)
    }

    /// Returns all named buffers.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PointBuffer)> {
        self.buffers.iter().map(|(name, buffer)| (name.as_str(), buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::{PointBuffer, PointBufferSet};
    use crate::{DType, PointField};

    #[test]
    fn resize_f32_buffer() {
        let field = PointField::scalar("x", crate::FieldSemantic::PositionX, DType::F32);
        let mut buffer = PointBuffer::with_capacity(DType::F32, 4);
        buffer.resize(3, &field).unwrap();
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn buffer_set_lookup() {
        let mut set = PointBufferSet::new();
        set.insert("x", PointBuffer::from_f32(vec![1.0, 2.0]));
        assert_eq!(set.get("x").unwrap().len(), 2);
    }
}

//! Chunked spatial-record sources and sinks.

use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, SpatialTensor};

use crate::{RecordsError, RecordsResult, SchemaDescriptor, SpatialRecord};

/// Pull-based source of versioned spatial records.
pub trait SpatialRecordSource {
    /// Returns the schema contract for every emitted record.
    fn schema(&self) -> &SchemaDescriptor;

    /// Returns the next record, or `None` when exhausted.
    fn next_record(&mut self) -> Option<RecordsResult<SpatialRecord>>;
}

/// Push-based sink for versioned spatial records.
pub trait SpatialRecordSink {
    /// Accepts one record.
    fn write_record(&mut self, record: &SpatialRecord) -> RecordsResult<()>;

    /// Finalizes the sink. Default is a no-op.
    fn finish(&mut self) -> RecordsResult<()> {
        Ok(())
    }
}

/// Splits one in-memory cloud into fixed-size record chunks.
pub struct MemoryChunkSource {
    schema: SchemaDescriptor,
    metadata: SpatialMetadata,
    cloud: PointCloud,
    chunk_size: usize,
    offset: usize,
}

impl MemoryChunkSource {
    /// Creates a chunked source over `cloud` using `chunk_size` points per record.
    pub fn try_new(
        schema: SchemaDescriptor,
        cloud: PointCloud,
        chunk_size: usize,
    ) -> RecordsResult<Self> {
        if chunk_size == 0 {
            return Err(RecordsError::InvalidConfiguration("chunk_size must be positive".into()));
        }
        if cloud.schema() != schema.point_schema() {
            return Err(RecordsError::SchemaMismatch(
                "chunk source cloud schema must match descriptor".into(),
            ));
        }
        cloud.validate()?;
        Ok(Self { metadata: cloud.metadata().clone(), schema, cloud, chunk_size, offset: 0 })
    }

    /// Creates a source using [`SpatialTensor`]’s default chunk size as a hint.
    pub fn try_with_default_chunk(
        schema: SchemaDescriptor,
        cloud: PointCloud,
    ) -> RecordsResult<Self> {
        let _ = SpatialTensor::new(&cloud, spatialrust_core::DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE)?;
        Self::try_new(schema, cloud, spatialrust_core::DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE)
    }
}

impl SpatialRecordSource for MemoryChunkSource {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn next_record(&mut self) -> Option<RecordsResult<SpatialRecord>> {
        if self.offset >= self.cloud.len() {
            return None;
        }
        let end = (self.offset + self.chunk_size).min(self.cloud.len());
        let range = self.offset..end;
        self.offset = end;
        Some(slice_cloud(&self.schema, &self.cloud, &self.metadata, range))
    }
}

/// Collects records into an owned point cloud.
#[derive(Clone, Debug, Default)]
pub struct MemoryChunkSink {
    schema: Option<SchemaDescriptor>,
    buffers: PointBufferSet,
    metadata: SpatialMetadata,
    len: usize,
}

impl MemoryChunkSink {
    /// Creates an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes the sink into one assembled record.
    pub fn into_record(self) -> RecordsResult<Option<SpatialRecord>> {
        let Some(schema) = self.schema else {
            return Ok(None);
        };
        if self.len == 0 {
            let cloud = PointCloud::try_from_parts(
                schema.point_schema().clone(),
                PointBufferSet::new(),
                self.metadata,
            )?;
            return Ok(Some(SpatialRecord::try_new(schema, cloud)?));
        }
        let cloud =
            PointCloud::try_from_parts(schema.point_schema().clone(), self.buffers, self.metadata)?;
        Ok(Some(SpatialRecord::try_new(schema, cloud)?))
    }
}

impl SpatialRecordSink for MemoryChunkSink {
    fn write_record(&mut self, record: &SpatialRecord) -> RecordsResult<()> {
        match &self.schema {
            None => {
                self.schema = Some(record.schema().clone());
                self.metadata = record.metadata().clone();
            }
            Some(schema) if schema != record.schema() => {
                return Err(RecordsError::SchemaMismatch(
                    "chunk sink requires a homogeneous schema across records".into(),
                ));
            }
            Some(_) => {}
        }
        for field in record.schema().point_schema().fields() {
            let source = record.cloud().field(&field.name)?;
            match self.buffers.get_mut(&field.name) {
                Some(dst) => append_buffer(dst, source)?,
                None => {
                    self.buffers.insert(field.name.clone(), clone_buffer(source)?);
                }
            }
        }
        self.len += record.cloud().len();
        Ok(())
    }
}

fn slice_cloud(
    schema: &SchemaDescriptor,
    cloud: &PointCloud,
    metadata: &SpatialMetadata,
    range: std::ops::Range<usize>,
) -> RecordsResult<SpatialRecord> {
    let mut buffers = PointBufferSet::new();
    for field in schema.point_schema().fields() {
        let source = cloud.field(&field.name)?;
        buffers.insert(field.name.clone(), slice_buffer(source, &range)?);
    }
    let chunk =
        PointCloud::try_from_parts(schema.point_schema().clone(), buffers, metadata.clone())?;
    SpatialRecord::try_new(schema.clone(), chunk)
}

fn slice_buffer(
    buffer: &PointBuffer,
    range: &std::ops::Range<usize>,
) -> RecordsResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::F32(values[range.clone()].to_vec()),
        PointBuffer::F64(values) => PointBuffer::F64(values[range.clone()].to_vec()),
        PointBuffer::U8(values) => PointBuffer::U8(values[range.clone()].to_vec()),
        PointBuffer::U16(values) => PointBuffer::U16(values[range.clone()].to_vec()),
        PointBuffer::U32(values) => PointBuffer::U32(values[range.clone()].to_vec()),
        PointBuffer::I32(values) => PointBuffer::I32(values[range.clone()].to_vec()),
    })
}

fn clone_buffer(buffer: &PointBuffer) -> RecordsResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::F32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    })
}

fn append_buffer(dst: &mut PointBuffer, src: &PointBuffer) -> RecordsResult<()> {
    match (dst, src) {
        (PointBuffer::F32(dst), PointBuffer::F32(src)) => dst.extend_from_slice(src),
        (PointBuffer::F64(dst), PointBuffer::F64(src)) => dst.extend_from_slice(src),
        (PointBuffer::U8(dst), PointBuffer::U8(src)) => dst.extend_from_slice(src),
        (PointBuffer::U16(dst), PointBuffer::U16(src)) => dst.extend_from_slice(src),
        (PointBuffer::U32(dst), PointBuffer::U32(src)) => dst.extend_from_slice(src),
        (PointBuffer::I32(dst), PointBuffer::I32(src)) => dst.extend_from_slice(src),
        (dst, src) => {
            return Err(RecordsError::SchemaMismatch(format!(
                "cannot append {:?} into {:?}",
                src.dtype(),
                dst.dtype()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MemoryChunkSink, MemoryChunkSource, SpatialRecordSink, SpatialRecordSource};
    use crate::{SchemaDescriptor, SchemaVersion, SpatialRecord};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };

    #[test]
    fn memory_chunk_roundtrip() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0, 1.0, 2.0, 3.0, 4.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0; 5]));
        buffers.insert("z", PointBuffer::from_f32(vec![1.0; 5]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let schema =
            SchemaDescriptor::try_new("point", SchemaVersion::new(1, 0), cloud.schema().clone())
                .unwrap();
        let mut source = MemoryChunkSource::try_new(schema.clone(), cloud, 2).unwrap();
        let mut sink = MemoryChunkSink::new();
        let mut chunks = 0;
        while let Some(record) = source.next_record() {
            sink.write_record(&record.unwrap()).unwrap();
            chunks += 1;
        }
        assert_eq!(chunks, 3);
        let assembled = sink.into_record().unwrap().unwrap();
        assert_eq!(assembled.cloud().len(), 5);
        assert_eq!(assembled.schema(), &schema);
        let _ = SpatialRecord::try_new(schema, assembled.into_cloud());
    }
}

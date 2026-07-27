//! Shared state for feature-gated bounded format adapters.

use spatialrust_core::{DType, PointCloud, PointSchema};
use spatialrust_records::{
    CancellationToken, ChunkIdentity, MemoryReservation, MemoryTracker, RecordsError,
    RecordsResult, SchemaDescriptor, SchemaVersion, SpatialRecord, SpatialRecordChunk,
    StreamOptions,
};

use crate::IoError;

#[cfg(any(feature = "io-pcd", feature = "io-ply"))]
pub(crate) const MAX_ASCII_RECORD_BYTES: usize = 16 * 1024;

pub(crate) struct FormatStreamState {
    pub(crate) schema: SchemaDescriptor,
    pub(crate) options: StreamOptions,
    pub(crate) tracker: MemoryTracker,
    pub(crate) cancellation: CancellationToken,
    pub(crate) max_chunk_bytes: u64,
    next_sequence: u64,
    next_point_offset: u64,
}

impl FormatStreamState {
    pub(crate) fn new(
        format_id: &str,
        schema: PointSchema,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> Result<Self, IoError> {
        let schema =
            SchemaDescriptor::try_new(format_id, SchemaVersion::new(1, 0), schema.clone())?;
        let max_chunk_bytes = schema_capacity_bytes(&schema, options.chunk_points())?;
        if max_chunk_bytes > options.memory_budget().limit_bytes() {
            return Err(spatialrust_records::RecordsError::MemoryBudgetExceeded {
                requested: max_chunk_bytes,
                current: 0,
                limit: options.memory_budget().limit_bytes(),
            }
            .into());
        }
        let tracker = MemoryTracker::new(options.memory_budget());
        Ok(Self {
            schema,
            options,
            tracker,
            cancellation,
            max_chunk_bytes,
            next_sequence: 0,
            next_point_offset: 0,
        })
    }

    #[cfg(any(feature = "io-las", feature = "io-copc"))]
    pub(crate) fn reserve_points(&self, point_count: usize) -> RecordsResult<MemoryReservation> {
        self.reserve_points_with_scratch(point_count, 0)
    }

    pub(crate) fn reserve_points_with_scratch(
        &self,
        point_count: usize,
        scratch_bytes: u64,
    ) -> RecordsResult<MemoryReservation> {
        self.cancellation.check()?;
        let bytes = schema_capacity_bytes(&self.schema, point_count)
            .map_err(|error| RecordsError::InvalidChunk(error.to_string()))?
            .checked_add(scratch_bytes)
            .ok_or_else(|| RecordsError::InvalidChunk("streaming working set overflow".into()))?;
        self.tracker.try_reserve(bytes)
    }

    pub(crate) fn lease(
        &mut self,
        cloud: PointCloud,
        reservation: MemoryReservation,
    ) -> RecordsResult<SpatialRecordChunk> {
        if cloud.is_empty() {
            return Err(RecordsError::InvalidChunk(
                "format source must not emit empty chunks".into(),
            ));
        }
        if cloud.len() > self.options.chunk_points() {
            return Err(RecordsError::InvalidChunk(format!(
                "format source emitted {} points above declared limit {}",
                cloud.len(),
                self.options.chunk_points()
            )));
        }
        let point_count = u64::try_from(cloud.len())
            .map_err(|_| RecordsError::InvalidChunk("point count does not fit u64".into()))?;
        let record = SpatialRecord::try_new(self.schema.clone(), cloud)?;
        let identity =
            ChunkIdentity { sequence: self.next_sequence, point_offset: self.next_point_offset };
        let chunk = SpatialRecordChunk::try_from_reserved(identity, record, reservation)?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| RecordsError::ReceiptOverflow("format chunk sequence".into()))?;
        self.next_point_offset = self
            .next_point_offset
            .checked_add(point_count)
            .ok_or_else(|| RecordsError::ReceiptOverflow("format point offset".into()))?;
        Ok(chunk)
    }
}

pub(crate) fn schema_capacity_bytes(
    schema: &SchemaDescriptor,
    point_count: usize,
) -> Result<u64, IoError> {
    let point_count = u64::try_from(point_count)
        .map_err(|_| IoError::Streaming("point count does not fit u64".into()))?;
    let mut bytes_per_point = 0_u64;
    for field in schema.point_schema().fields() {
        let scalar_bytes = match field.dtype {
            DType::F32 | DType::F16 => 4,
            DType::F64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 | DType::I32 => 4,
        };
        bytes_per_point = bytes_per_point
            .checked_add(scalar_bytes)
            .ok_or_else(|| IoError::Streaming("schema byte size overflow".into()))?;
    }
    bytes_per_point
        .checked_mul(point_count)
        .ok_or_else(|| IoError::Streaming("chunk byte size overflow".into()))
}

pub(crate) fn records_io(error: IoError) -> RecordsError {
    RecordsError::InvalidChunk(error.to_string())
}

#[cfg(any(feature = "io-pcd", feature = "io-ply"))]
pub(crate) fn read_bounded_ascii_line<'a, R: std::io::BufRead>(
    reader: &mut R,
    buffer: &'a mut [u8],
    format: &str,
) -> Result<Option<&'a str>, IoError> {
    let mut written = 0;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if written == 0 {
                return Ok(None);
            }
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let copy_len = newline.unwrap_or(available.len());
        if written + copy_len > buffer.len() {
            return Err(IoError::Streaming(format!(
                "{format} ASCII record exceeds {} bytes",
                buffer.len()
            )));
        }
        buffer[written..written + copy_len].copy_from_slice(&available[..copy_len]);
        written += copy_len;
        let consumed = copy_len + usize::from(newline.is_some());
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if written > 0 && buffer[written - 1] == b'\r' {
        written -= 1;
    }
    std::str::from_utf8(&buffer[..written])
        .map(Some)
        .map_err(|_| IoError::Streaming(format!("{format} ASCII record is not UTF-8")))
}

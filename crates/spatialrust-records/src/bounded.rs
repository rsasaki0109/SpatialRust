//! Backward-compatible bounded record streams with leased chunk ownership.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc::{sync_channel, Receiver},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};

use spatialrust_core::{FieldSemantic, PointBuffer, PointBufferSet, PointCloud, SpatialMetadata};

use crate::{
    CancellationToken, MemoryReservation, MemoryTracker, RecordsError, RecordsResult,
    SchemaDescriptor, SpatialRecord, SpatialRecordSink, SpatialRecordSource, StreamOptions,
};

/// Stable location of one chunk in its source stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIdentity {
    /// Zero-based chunk sequence.
    pub sequence: u64,
    /// Global point offset of the first point in the chunk.
    pub point_offset: u64,
}

/// Finite axis-aligned bounds carried by a record chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordBounds3 {
    /// Inclusive minimum XYZ coordinates.
    pub min: [f64; 3],
    /// Inclusive maximum XYZ coordinates.
    pub max: [f64; 3],
}

impl RecordBounds3 {
    /// Creates validated finite, ordered bounds.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> RecordsResult<Self> {
        if min.into_iter().chain(max).any(|component| !component.is_finite())
            || (0..3).any(|axis| min[axis] > max[axis])
        {
            return Err(RecordsError::InvalidChunk(
                "chunk bounds must be finite and ordered".into(),
            ));
        }
        Ok(Self { min, max })
    }
}

type BufferPool = Arc<Mutex<Vec<PointBufferSet>>>;

/// One owned record plus the memory lease that keeps it inside the hard limit.
///
/// The record is borrowed from this wrapper and cannot outlive its reservation.
/// Source-specific buffers may be returned to a recycling pool when the chunk
/// is dropped.
#[derive(Debug)]
pub struct SpatialRecordChunk {
    identity: ChunkIdentity,
    bounds: Option<RecordBounds3>,
    tracked_bytes: u64,
    record: Option<SpatialRecord>,
    reservation: MemoryReservation,
    recycle_pool: Option<BufferPool>,
}

impl SpatialRecordChunk {
    fn new(
        identity: ChunkIdentity,
        bounds: Option<RecordBounds3>,
        tracked_bytes: u64,
        record: SpatialRecord,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            identity,
            bounds,
            tracked_bytes,
            record: Some(record),
            reservation,
            recycle_pool: None,
        }
    }

    fn with_recycle_pool(mut self, pool: BufferPool) -> Self {
        self.recycle_pool = Some(pool);
        self
    }

    /// Returns the deterministic source identity.
    #[must_use]
    pub const fn identity(&self) -> ChunkIdentity {
        self.identity
    }

    /// Returns optional finite bounds.
    #[must_use]
    pub const fn bounds(&self) -> Option<RecordBounds3> {
        self.bounds
    }

    /// Returns explicitly tracked column-capacity bytes for this chunk.
    #[must_use]
    pub const fn tracked_bytes(&self) -> u64 {
        self.tracked_bytes
    }

    /// Borrows the owned record while its memory lease remains alive.
    #[must_use]
    pub fn record(&self) -> &SpatialRecord {
        self.record.as_ref().expect("record is present until chunk drop")
    }

    /// Returns the exact held reservation.
    #[must_use]
    pub const fn reservation(&self) -> &MemoryReservation {
        &self.reservation
    }
}

impl Drop for SpatialRecordChunk {
    fn drop(&mut self) {
        let Some(pool) = &self.recycle_pool else {
            return;
        };
        let Some(record) = self.record.take() else {
            return;
        };
        let (_, mut buffers, _) = record.into_cloud().into_parts();
        clear_buffers(&mut buffers);
        lock_pool(pool).push(buffers);
    }
}

/// Pull-based bounded source. Existing [`SpatialRecordSource`] remains unchanged.
pub trait BoundedSpatialRecordSource {
    /// Schema shared by every emitted record.
    fn schema(&self) -> &SchemaDescriptor;
    /// Validated execution options.
    fn options(&self) -> &StreamOptions;
    /// Exact tracker shared by emitted chunk leases.
    fn memory_tracker(&self) -> &MemoryTracker;
    /// Cooperative cancellation state checked at chunk boundaries.
    fn cancellation_token(&self) -> CancellationToken;
    /// Conservative maximum resident bytes for one emitted chunk.
    fn max_chunk_bytes(&self) -> u64;
    /// Returns the next leased chunk, or `None` after an explicit end.
    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>>;
}

/// Push-based bounded sink that cannot retain a record beyond the chunk lease.
pub trait BoundedSpatialRecordSink {
    /// Writes one chunk synchronously.
    fn write_chunk(&mut self, chunk: &SpatialRecordChunk) -> RecordsResult<()>;

    /// Finalizes the sink.
    fn finish(&mut self) -> RecordsResult<()> {
        Ok(())
    }
}

/// Adapts an existing record source without changing its public trait.
pub struct LegacyBoundedSource<S> {
    source: S,
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    next_sequence: u64,
    next_point_offset: u64,
    max_chunk_bytes: u64,
}

impl<S: SpatialRecordSource> LegacyBoundedSource<S> {
    /// Wraps `source`, reserving its declared maximum before each pull.
    pub fn try_new(
        source: S,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> RecordsResult<Self> {
        let schema = source.schema().clone();
        let max_chunk_bytes = max_storage_bytes(&schema, options.chunk_points())?;
        if max_chunk_bytes > options.memory_budget().limit_bytes() {
            return Err(RecordsError::MemoryBudgetExceeded {
                requested: max_chunk_bytes,
                current: 0,
                limit: options.memory_budget().limit_bytes(),
            });
        }
        let tracker = MemoryTracker::new(options.memory_budget());
        Ok(Self {
            source,
            schema,
            options,
            tracker,
            cancellation,
            next_sequence: 0,
            next_point_offset: 0,
            max_chunk_bytes,
        })
    }

    /// Consumes the adapter and returns the legacy source.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.source
    }
}

impl<S: SpatialRecordSource> BoundedSpatialRecordSource for LegacyBoundedSource<S> {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        if let Err(error) = self.cancellation.check() {
            return Some(Err(error));
        }
        let mut reservation = match self.tracker.try_reserve(self.max_chunk_bytes) {
            Ok(reservation) => reservation,
            Err(error) => return Some(Err(error)),
        };
        let record = match self.source.next_record()? {
            Ok(record) => record,
            Err(error) => return Some(Err(error)),
        };
        if record.cloud().is_empty() {
            return Some(Err(RecordsError::InvalidChunk(
                "sources must not emit empty chunks".into(),
            )));
        }
        if record.cloud().len() > self.options.chunk_points() {
            return Some(Err(RecordsError::InvalidChunk(format!(
                "source emitted {} points above declared chunk limit {}",
                record.cloud().len(),
                self.options.chunk_points()
            ))));
        }
        let tracked_bytes = match record_storage_bytes(&record) {
            Ok(bytes) => bytes,
            Err(error) => return Some(Err(error)),
        };
        if tracked_bytes > self.max_chunk_bytes {
            return Some(Err(RecordsError::InvalidChunk(format!(
                "chunk storage {tracked_bytes} exceeds reserved maximum {}",
                self.max_chunk_bytes
            ))));
        }
        if let Err(error) = reservation.shrink_to(tracked_bytes) {
            return Some(Err(error));
        }
        let point_count = match u64::try_from(record.cloud().len()) {
            Ok(value) => value,
            Err(_) => {
                return Some(Err(RecordsError::InvalidChunk(
                    "chunk point count does not fit u64".into(),
                )))
            }
        };
        let identity =
            ChunkIdentity { sequence: self.next_sequence, point_offset: self.next_point_offset };
        self.next_sequence = match self.next_sequence.checked_add(1) {
            Some(value) => value,
            None => return Some(Err(RecordsError::ReceiptOverflow("chunk sequence".into()))),
        };
        self.next_point_offset = match self.next_point_offset.checked_add(point_count) {
            Some(value) => value,
            None => return Some(Err(RecordsError::ReceiptOverflow("point offset".into()))),
        };
        let bounds = record_bounds(&record);
        Some(Ok(SpatialRecordChunk::new(identity, bounds, tracked_bytes, record, reservation)))
    }
}

/// Adapts an existing synchronous sink to leased record chunks.
pub struct LegacyBoundedSink<S> {
    sink: S,
}

impl<S> LegacyBoundedSink<S> {
    /// Creates a synchronous bounded sink adapter.
    #[must_use]
    pub const fn new(sink: S) -> Self {
        Self { sink }
    }

    /// Consumes the adapter and returns its sink.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.sink
    }
}

impl<S: SpatialRecordSink> BoundedSpatialRecordSink for LegacyBoundedSink<S> {
    fn write_chunk(&mut self, chunk: &SpatialRecordChunk) -> RecordsResult<()> {
        self.sink.write_record(chunk.record())
    }

    fn finish(&mut self) -> RecordsResult<()> {
        self.sink.finish()
    }
}

enum PrefetchMessage {
    Item(Box<RecordsResult<SpatialRecordChunk>>),
    End,
}

/// Single-worker deterministic prefetch with count and memory backpressure.
pub struct PrefetchRecordSource {
    schema: SchemaDescriptor,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    max_chunk_bytes: u64,
    receiver: Option<Receiver<PrefetchMessage>>,
    worker: Option<JoinHandle<()>>,
    finished: bool,
    next_expected_sequence: u64,
    next_expected_point_offset: u64,
}

impl PrefetchRecordSource {
    /// Starts prefetch using `source.options().prefetch_chunks()` as queue capacity.
    pub fn try_new<S>(mut source: S) -> RecordsResult<Self>
    where
        S: BoundedSpatialRecordSource + Send + 'static,
    {
        let options = source.options().clone();
        let capacity = options.prefetch_chunks();
        if capacity == 0 {
            return Err(RecordsError::InvalidConfiguration(
                "prefetch source requires prefetch_chunks > 0".into(),
            ));
        }
        let max_chunk_bytes = source.max_chunk_bytes();
        let concurrent_chunks =
            u64::try_from(capacity).ok().and_then(|value| value.checked_add(1)).ok_or_else(
                || RecordsError::InvalidConfiguration("prefetch capacity overflow".into()),
            )?;
        let required = max_chunk_bytes.checked_mul(concurrent_chunks).ok_or_else(|| {
            RecordsError::InvalidConfiguration("prefetch memory requirement overflow".into())
        })?;
        if required > options.memory_budget().limit_bytes() {
            return Err(RecordsError::MemoryBudgetExceeded {
                requested: required,
                current: 0,
                limit: options.memory_budget().limit_bytes(),
            });
        }

        let schema = source.schema().clone();
        let tracker = source.memory_tracker().clone();
        let cancellation = source.cancellation_token();
        let worker_cancellation = cancellation.clone();
        let (sender, receiver) = sync_channel(capacity);
        let worker = thread::spawn(move || loop {
            if worker_cancellation.is_cancelled() {
                break;
            }
            match source.next_chunk() {
                Some(result) => {
                    let stop = result.is_err();
                    if sender.send(PrefetchMessage::Item(Box::new(result))).is_err() || stop {
                        break;
                    }
                }
                None => {
                    let _ = sender.send(PrefetchMessage::End);
                    break;
                }
            }
        });
        Ok(Self {
            schema,
            options,
            tracker,
            cancellation,
            max_chunk_bytes,
            receiver: Some(receiver),
            worker: Some(worker),
            finished: false,
            next_expected_sequence: 0,
            next_expected_point_offset: 0,
        })
    }
}

impl BoundedSpatialRecordSource for PrefetchRecordSource {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        if self.finished {
            return None;
        }
        let Some(receiver) = &self.receiver else {
            self.finished = true;
            return Some(Err(RecordsError::StreamClosed));
        };
        match receiver.recv() {
            Ok(PrefetchMessage::Item(result)) => match *result {
                Ok(chunk) => {
                    let identity = chunk.identity();
                    if identity.sequence != self.next_expected_sequence
                        || identity.point_offset != self.next_expected_point_offset
                    {
                        self.cancellation.cancel();
                        return Some(Err(RecordsError::InvalidChunk(format!(
                            "non-contiguous chunk identity {:?}, expected sequence {} offset {}",
                            identity, self.next_expected_sequence, self.next_expected_point_offset
                        ))));
                    }
                    let point_count = match u64::try_from(chunk.record().cloud().len()) {
                        Ok(value) => value,
                        Err(_) => {
                            self.cancellation.cancel();
                            return Some(Err(RecordsError::InvalidChunk(
                                "chunk point count does not fit u64".into(),
                            )));
                        }
                    };
                    self.next_expected_sequence = match self.next_expected_sequence.checked_add(1) {
                        Some(value) => value,
                        None => {
                            self.cancellation.cancel();
                            return Some(Err(RecordsError::ReceiptOverflow(
                                "prefetch chunk sequence".into(),
                            )));
                        }
                    };
                    self.next_expected_point_offset =
                        match self.next_expected_point_offset.checked_add(point_count) {
                            Some(value) => value,
                            None => {
                                self.cancellation.cancel();
                                return Some(Err(RecordsError::ReceiptOverflow(
                                    "prefetch point offset".into(),
                                )));
                            }
                        };
                    Some(Ok(chunk))
                }
                Err(error) => Some(Err(error)),
            },
            Ok(PrefetchMessage::End) => {
                self.finished = true;
                None
            }
            Err(_) => {
                self.finished = true;
                Some(Err(RecordsError::StreamClosed))
            }
        }
    }
}

impl Drop for PrefetchRecordSource {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// In-memory source used to verify real buffer recycling and lease accounting.
///
/// The immutable backing cloud represents external source storage. Emitted
/// chunk buffers are separately reserved and returned to the pool on drop.
pub struct RecyclingMemoryChunkSource {
    schema: SchemaDescriptor,
    cloud: PointCloud,
    options: StreamOptions,
    tracker: MemoryTracker,
    cancellation: CancellationToken,
    next_sequence: u64,
    next_point_offset: u64,
    offset: usize,
    max_chunk_bytes: u64,
    pool: BufferPool,
    buffer_set_allocations: Arc<AtomicU64>,
}

impl RecyclingMemoryChunkSource {
    /// Creates a recycling source over an immutable backing cloud.
    pub fn try_new(
        schema: SchemaDescriptor,
        cloud: PointCloud,
        options: StreamOptions,
        cancellation: CancellationToken,
    ) -> RecordsResult<Self> {
        if cloud.schema() != schema.point_schema() {
            return Err(RecordsError::SchemaMismatch(
                "recycling source cloud schema must match descriptor".into(),
            ));
        }
        cloud.validate()?;
        let max_chunk_bytes = max_storage_bytes(&schema, options.chunk_points())?;
        if max_chunk_bytes > options.memory_budget().limit_bytes() {
            return Err(RecordsError::MemoryBudgetExceeded {
                requested: max_chunk_bytes,
                current: 0,
                limit: options.memory_budget().limit_bytes(),
            });
        }
        let tracker = MemoryTracker::new(options.memory_budget());
        Ok(Self {
            schema,
            cloud,
            options,
            tracker,
            cancellation,
            next_sequence: 0,
            next_point_offset: 0,
            offset: 0,
            max_chunk_bytes,
            pool: Arc::new(Mutex::new(Vec::new())),
            buffer_set_allocations: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Returns buffer-set allocations performed because the pool was empty.
    #[must_use]
    pub fn buffer_set_allocations(&self) -> u64 {
        self.buffer_set_allocations.load(Ordering::Acquire)
    }

    /// Returns recycled buffer sets currently available to the source.
    #[must_use]
    pub fn pooled_buffer_sets(&self) -> usize {
        lock_pool(&self.pool).len()
    }

    fn take_buffers(&self) -> PointBufferSet {
        if let Some(buffers) = lock_pool(&self.pool).pop() {
            return buffers;
        }
        self.buffer_set_allocations.fetch_add(1, Ordering::AcqRel);
        let mut buffers = PointBufferSet::new();
        for field in self.schema.point_schema().fields() {
            buffers.insert(
                field.name.clone(),
                PointBuffer::with_capacity(field.dtype, self.options.chunk_points()),
            );
        }
        buffers
    }
}

impl BoundedSpatialRecordSource for RecyclingMemoryChunkSource {
    fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    fn options(&self) -> &StreamOptions {
        &self.options
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        &self.tracker
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.max_chunk_bytes
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        if let Err(error) = self.cancellation.check() {
            return Some(Err(error));
        }
        if self.offset >= self.cloud.len() {
            return None;
        }
        let end = (self.offset + self.options.chunk_points()).min(self.cloud.len());
        let range = self.offset..end;
        let point_count = end - self.offset;
        // Recycled vectors retain full chunk capacity even for the final short
        // chunk, so the lease keeps the full reusable allocation reserved.
        let tracked_bytes = self.max_chunk_bytes;
        let reservation = match self.tracker.try_reserve(tracked_bytes) {
            Ok(reservation) => reservation,
            Err(error) => return Some(Err(error)),
        };
        let mut buffers = self.take_buffers();
        for field in self.schema.point_schema().fields() {
            let source = match self.cloud.field(&field.name) {
                Ok(buffer) => buffer,
                Err(error) => return Some(Err(error.into())),
            };
            let Some(destination) = buffers.get_mut(&field.name) else {
                return Some(Err(RecordsError::SchemaMismatch(format!(
                    "recycled buffer set is missing `{}`",
                    field.name
                ))));
            };
            if let Err(error) = copy_buffer_range(destination, source, range.clone()) {
                return Some(Err(error));
            }
        }
        let metadata: SpatialMetadata = self.cloud.metadata().clone();
        let cloud =
            match PointCloud::try_from_parts(self.schema.point_schema().clone(), buffers, metadata)
            {
                Ok(cloud) => cloud,
                Err(error) => return Some(Err(error.into())),
            };
        let record = match SpatialRecord::try_new(self.schema.clone(), cloud) {
            Ok(record) => record,
            Err(error) => return Some(Err(error)),
        };
        let identity =
            ChunkIdentity { sequence: self.next_sequence, point_offset: self.next_point_offset };
        self.next_sequence = match self.next_sequence.checked_add(1) {
            Some(value) => value,
            None => return Some(Err(RecordsError::ReceiptOverflow("chunk sequence".into()))),
        };
        let point_count = match u64::try_from(point_count) {
            Ok(value) => value,
            Err(_) => {
                return Some(Err(RecordsError::InvalidChunk(
                    "chunk point count does not fit u64".into(),
                )))
            }
        };
        self.next_point_offset = match self.next_point_offset.checked_add(point_count) {
            Some(value) => value,
            None => return Some(Err(RecordsError::ReceiptOverflow("point offset".into()))),
        };
        self.offset = end;
        let bounds = record_bounds(&record);
        Some(Ok(SpatialRecordChunk::new(identity, bounds, tracked_bytes, record, reservation)
            .with_recycle_pool(self.pool.clone())))
    }
}

/// Returns the allocated scalar capacity bytes owned by a record.
pub fn record_storage_bytes(record: &SpatialRecord) -> RecordsResult<u64> {
    let mut total = 0_u64;
    for field in record.schema().point_schema().fields() {
        let buffer = record.cloud().field(&field.name)?;
        let bytes = buffer_capacity(buffer)
            .and_then(|len| len.checked_mul(buffer_scalar_bytes(buffer)))
            .ok_or_else(|| RecordsError::InvalidChunk("record storage byte overflow".into()))?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| RecordsError::InvalidChunk("record storage byte overflow".into()))?;
    }
    Ok(total)
}

fn max_storage_bytes(schema: &SchemaDescriptor, point_count: usize) -> RecordsResult<u64> {
    let point_count = u64::try_from(point_count)
        .map_err(|_| RecordsError::InvalidConfiguration("chunk size does not fit u64".into()))?;
    let mut bytes_per_point = 0_u64;
    for field in schema.point_schema().fields() {
        let scalar_bytes = match field.dtype {
            spatialrust_core::DType::F32 | spatialrust_core::DType::F16 => 4,
            spatialrust_core::DType::F64 => 8,
            spatialrust_core::DType::U8 => 1,
            spatialrust_core::DType::U16 => 2,
            spatialrust_core::DType::U32 | spatialrust_core::DType::I32 => 4,
        };
        bytes_per_point = bytes_per_point.checked_add(scalar_bytes).ok_or_else(|| {
            RecordsError::InvalidConfiguration("schema storage byte overflow".into())
        })?;
    }
    bytes_per_point
        .checked_mul(point_count)
        .ok_or_else(|| RecordsError::InvalidConfiguration("chunk storage byte overflow".into()))
}

fn buffer_scalar_bytes(buffer: &PointBuffer) -> u64 {
    match buffer {
        PointBuffer::F32(_) | PointBuffer::U32(_) | PointBuffer::I32(_) => 4,
        PointBuffer::F64(_) => 8,
        PointBuffer::U8(_) => 1,
        PointBuffer::U16(_) => 2,
    }
}

fn buffer_capacity(buffer: &PointBuffer) -> Option<u64> {
    let capacity = match buffer {
        PointBuffer::F32(values) => values.capacity(),
        PointBuffer::F64(values) => values.capacity(),
        PointBuffer::U8(values) => values.capacity(),
        PointBuffer::U16(values) => values.capacity(),
        PointBuffer::U32(values) => values.capacity(),
        PointBuffer::I32(values) => values.capacity(),
    };
    u64::try_from(capacity).ok()
}

fn record_bounds(record: &SpatialRecord) -> Option<RecordBounds3> {
    let schema = record.schema().point_schema();
    let x = record
        .cloud()
        .field(&schema.find_semantic(FieldSemantic::PositionX)?.name)
        .ok()?
        .as_f32()
        .ok()?;
    let y = record
        .cloud()
        .field(&schema.find_semantic(FieldSemantic::PositionY)?.name)
        .ok()?
        .as_f32()
        .ok()?;
    let z = record
        .cloud()
        .field(&schema.find_semantic(FieldSemantic::PositionZ)?.name)
        .ok()?
        .as_f32()
        .ok()?;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut found = false;
    for ((&x, &y), &z) in x.iter().zip(y).zip(z) {
        let point = [f64::from(x), f64::from(y), f64::from(z)];
        if point.iter().any(|value| !value.is_finite()) {
            continue;
        }
        found = true;
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    found.then_some(RecordBounds3 { min, max })
}

fn clear_buffers(buffers: &mut PointBufferSet) {
    for (_, buffer) in buffers.iter_mut() {
        clear_buffer(buffer);
    }
}

fn clear_buffer(buffer: &mut PointBuffer) {
    match buffer {
        PointBuffer::F32(values) => values.clear(),
        PointBuffer::F64(values) => values.clear(),
        PointBuffer::U8(values) => values.clear(),
        PointBuffer::U16(values) => values.clear(),
        PointBuffer::U32(values) => values.clear(),
        PointBuffer::I32(values) => values.clear(),
    }
}

fn copy_buffer_range(
    destination: &mut PointBuffer,
    source: &PointBuffer,
    range: std::ops::Range<usize>,
) -> RecordsResult<()> {
    clear_buffer(destination);
    match (destination, source) {
        (PointBuffer::F32(dst), PointBuffer::F32(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (PointBuffer::F64(dst), PointBuffer::F64(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (PointBuffer::U8(dst), PointBuffer::U8(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (PointBuffer::U16(dst), PointBuffer::U16(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (PointBuffer::U32(dst), PointBuffer::U32(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (PointBuffer::I32(dst), PointBuffer::I32(src)) => {
            dst.extend_from_slice(&src[range]);
        }
        (dst, src) => {
            return Err(RecordsError::SchemaMismatch(format!(
                "cannot recycle {:?} storage for {:?}",
                dst.dtype(),
                src.dtype()
            )));
        }
    }
    Ok(())
}

fn lock_pool(pool: &BufferPool) -> std::sync::MutexGuard<'_, Vec<PointBufferSet>> {
    pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::{
        record_storage_bytes, BoundedSpatialRecordSink, BoundedSpatialRecordSource,
        LegacyBoundedSink, LegacyBoundedSource, PrefetchRecordSource, RecyclingMemoryChunkSource,
    };
    use crate::{
        CancellationToken, MemoryBudget, MemoryChunkSink, MemoryChunkSource, SchemaDescriptor,
        SchemaVersion, StreamOptions,
    };
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, PointCloudBuilder, SpatialMetadata,
        StandardSchemas,
    };

    fn cloud(points: usize) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::xyz();
        for index in 0..points {
            builder.push_point([index as f32, 1.0, -1.0]).unwrap();
        }
        builder.build().unwrap()
    }

    fn schema() -> SchemaDescriptor {
        SchemaDescriptor::try_new("point", SchemaVersion::new(1, 0), StandardSchemas::point_xyz())
            .unwrap()
    }

    #[test]
    fn storage_accounting_uses_vector_capacity_not_only_length() {
        let mut buffers = PointBufferSet::new();
        for name in ["x", "y", "z"] {
            let mut values = Vec::with_capacity(10);
            values.push(1.0);
            buffers.insert(name, PointBuffer::from_f32(values));
        }
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let record =
            crate::SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap();
        assert_eq!(record_storage_bytes(&record).unwrap(), 120);
    }

    #[test]
    fn legacy_source_gets_identity_bounds_and_drop_scoped_memory() {
        let legacy = MemoryChunkSource::try_new(schema(), cloud(5), 2).unwrap();
        let options = StreamOptions::new(2, MemoryBudget::new(24).unwrap()).unwrap();
        let mut source =
            LegacyBoundedSource::try_new(legacy, options, CancellationToken::default()).unwrap();

        let first = source.next_chunk().unwrap().unwrap();
        assert_eq!(first.identity().sequence, 0);
        assert_eq!(first.identity().point_offset, 0);
        assert_eq!(first.bounds().unwrap().min, [0.0, 1.0, -1.0]);
        assert_eq!(source.memory_tracker().snapshot().current_bytes, 24);
        drop(first);
        assert_eq!(source.memory_tracker().snapshot().current_bytes, 0);

        let second = source.next_chunk().unwrap().unwrap();
        assert_eq!(second.identity().point_offset, 2);
    }

    #[test]
    fn cancellation_stops_before_pulling_another_chunk() {
        let token = CancellationToken::default();
        let legacy = MemoryChunkSource::try_new(schema(), cloud(2), 2).unwrap();
        let options = StreamOptions::new(2, MemoryBudget::new(24).unwrap()).unwrap();
        let mut source = LegacyBoundedSource::try_new(legacy, options, token.clone()).unwrap();
        token.cancel();
        assert!(source.next_chunk().unwrap().is_err());
    }

    #[test]
    fn recycling_source_reuses_one_buffer_set_in_steady_state() {
        let options = StreamOptions::new(2, MemoryBudget::new(24).unwrap()).unwrap();
        let mut source = RecyclingMemoryChunkSource::try_new(
            schema(),
            cloud(5),
            options,
            CancellationToken::default(),
        )
        .unwrap();
        for expected_sequence in 0..3 {
            let chunk = source.next_chunk().unwrap().unwrap();
            assert_eq!(chunk.identity().sequence, expected_sequence);
            drop(chunk);
            assert_eq!(source.pooled_buffer_sets(), 1);
        }
        assert_eq!(source.buffer_set_allocations(), 1);
        assert!(source.next_chunk().is_none());
    }

    #[test]
    fn prefetch_preserves_order_and_stays_within_budget() {
        let legacy = MemoryChunkSource::try_new(schema(), cloud(6), 2).unwrap();
        let options =
            StreamOptions::new(2, MemoryBudget::new(72).unwrap()).unwrap().with_prefetch_chunks(2);
        let bounded =
            LegacyBoundedSource::try_new(legacy, options, CancellationToken::default()).unwrap();
        let mut source = PrefetchRecordSource::try_new(bounded).unwrap();
        for expected in 0..3 {
            let chunk = source.next_chunk().unwrap().unwrap();
            assert_eq!(chunk.identity().sequence, expected);
        }
        assert!(source.next_chunk().is_none());
        assert!(source.memory_tracker().snapshot().peak_bytes <= 72);
    }

    #[test]
    fn prefetch_rejects_capacity_that_cannot_fit_budget() {
        let legacy = MemoryChunkSource::try_new(schema(), cloud(6), 2).unwrap();
        let options =
            StreamOptions::new(2, MemoryBudget::new(48).unwrap()).unwrap().with_prefetch_chunks(2);
        let bounded =
            LegacyBoundedSource::try_new(legacy, options, CancellationToken::default()).unwrap();
        assert!(PrefetchRecordSource::try_new(bounded).is_err());
    }

    #[test]
    fn bounded_sink_bridges_existing_sink_synchronously() {
        let legacy = MemoryChunkSource::try_new(schema(), cloud(4), 2).unwrap();
        let options = StreamOptions::new(2, MemoryBudget::new(24).unwrap()).unwrap();
        let mut source =
            LegacyBoundedSource::try_new(legacy, options, CancellationToken::default()).unwrap();
        let mut sink = LegacyBoundedSink::new(MemoryChunkSink::new());
        while let Some(chunk) = source.next_chunk() {
            sink.write_chunk(&chunk.unwrap()).unwrap();
        }
        BoundedSpatialRecordSink::finish(&mut sink).unwrap();
        let record = sink.into_inner().into_record().unwrap().unwrap();
        assert_eq!(record.cloud().len(), 4);
    }
}

//! Bounded streaming configuration, memory accounting, receipts, and workloads.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use crate::{RecordsError, RecordsResult};

/// Default number of points requested from a streaming source.
pub const DEFAULT_STREAM_CHUNK_POINTS: usize = 16_384;
/// Default hard limit for explicitly tracked streaming memory (256 MiB).
pub const DEFAULT_STREAM_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
/// Stable identifier for the versioned JSON receipt contract.
pub const STREAMING_RECEIPT_SCHEMA: &str = "spatialrust.streaming.receipt";
/// Current streaming receipt schema version.
pub const STREAMING_RECEIPT_VERSION: u32 = 1;

/// Hard limit for memory explicitly owned by a streaming execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    limit_bytes: u64,
}

impl MemoryBudget {
    /// Creates a non-zero hard memory limit.
    pub fn new(limit_bytes: u64) -> RecordsResult<Self> {
        if limit_bytes == 0 {
            return Err(RecordsError::InvalidConfiguration(
                "streaming memory budget must be positive".into(),
            ));
        }
        Ok(Self { limit_bytes })
    }

    /// Returns the hard limit in bytes.
    #[must_use]
    pub const fn limit_bytes(self) -> u64 {
        self.limit_bytes
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self { limit_bytes: DEFAULT_STREAM_MEMORY_BUDGET_BYTES }
    }
}

/// Ordering guarantee requested from a streaming source or pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamOrdering {
    /// Preserve the deterministic order defined by the source.
    #[default]
    Source,
    /// Permit reordering while requiring deterministic output for identical inputs.
    Deterministic,
}

/// Common bounded-stream execution options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamOptions {
    chunk_points: NonZeroUsize,
    memory_budget: MemoryBudget,
    prefetch_chunks: usize,
    ordering: StreamOrdering,
}

impl StreamOptions {
    /// Creates options with an explicit positive chunk size and hard memory budget.
    pub fn new(chunk_points: usize, memory_budget: MemoryBudget) -> RecordsResult<Self> {
        let chunk_points = NonZeroUsize::new(chunk_points).ok_or_else(|| {
            RecordsError::InvalidConfiguration("stream chunk_points must be positive".into())
        })?;
        Ok(Self {
            chunk_points,
            memory_budget,
            prefetch_chunks: 0,
            ordering: StreamOrdering::Source,
        })
    }

    /// Requests bounded source prefetch. Zero disables prefetch.
    #[must_use]
    pub const fn with_prefetch_chunks(mut self, prefetch_chunks: usize) -> Self {
        self.prefetch_chunks = prefetch_chunks;
        self
    }

    /// Selects the required ordering contract.
    #[must_use]
    pub const fn with_ordering(mut self, ordering: StreamOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Returns the requested maximum points per source chunk.
    #[must_use]
    pub const fn chunk_points(&self) -> usize {
        self.chunk_points.get()
    }

    /// Returns the hard memory budget.
    #[must_use]
    pub const fn memory_budget(&self) -> MemoryBudget {
        self.memory_budget
    }

    /// Returns the maximum number of prefetched chunks.
    #[must_use]
    pub const fn prefetch_chunks(&self) -> usize {
        self.prefetch_chunks
    }

    /// Returns the required ordering contract.
    #[must_use]
    pub const fn ordering(&self) -> StreamOrdering {
        self.ordering
    }
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            chunk_points: NonZeroUsize::new(DEFAULT_STREAM_CHUNK_POINTS)
                .expect("default stream chunk size is non-zero"),
            memory_budget: MemoryBudget::default(),
            prefetch_chunks: 0,
            ordering: StreamOrdering::Source,
        }
    }
}

#[derive(Debug)]
struct MemoryTrackerInner {
    limit_bytes: u64,
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

/// Concurrent, exact accounting for memory explicitly owned by a stream.
///
/// Reservations fail before the configured hard limit is exceeded. Dropping a
/// [`MemoryReservation`] releases its bytes, including during error unwinding.
#[derive(Clone, Debug)]
pub struct MemoryTracker {
    inner: Arc<MemoryTrackerInner>,
}

impl MemoryTracker {
    /// Creates a tracker for `budget`.
    #[must_use]
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            inner: Arc::new(MemoryTrackerInner {
                limit_bytes: budget.limit_bytes,
                current_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Atomically reserves bytes or fails without changing the current count.
    pub fn try_reserve(&self, bytes: u64) -> RecordsResult<MemoryReservation> {
        let limit = self.inner.limit_bytes;
        let result =
            self.inner.current_bytes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(bytes).filter(|next| *next <= limit)
            });
        match result {
            Ok(previous) => {
                let current = previous + bytes;
                self.inner.peak_bytes.fetch_max(current, Ordering::AcqRel);
                Ok(MemoryReservation { tracker: self.clone(), bytes })
            }
            Err(current) => {
                Err(RecordsError::MemoryBudgetExceeded { requested: bytes, current, limit })
            }
        }
    }

    /// Returns an atomic snapshot of current, peak, and limit bytes.
    #[must_use]
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            current_bytes: self.inner.current_bytes.load(Ordering::Acquire),
            peak_bytes: self.inner.peak_bytes.load(Ordering::Acquire),
            limit_bytes: self.inner.limit_bytes,
        }
    }
}

/// Point-in-time memory accounting values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Bytes currently reserved.
    pub current_bytes: u64,
    /// Maximum simultaneously reserved bytes.
    pub peak_bytes: u64,
    /// Configured hard limit.
    pub limit_bytes: u64,
}

/// RAII token that releases an exact tracked byte count on drop.
#[derive(Debug)]
pub struct MemoryReservation {
    tracker: MemoryTracker,
    bytes: u64,
}

impl MemoryReservation {
    /// Returns the reserved byte count.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Releases the unused tail of a conservative reservation.
    pub fn shrink_to(&mut self, bytes: u64) -> RecordsResult<()> {
        if bytes > self.bytes {
            return Err(RecordsError::InvalidConfiguration(format!(
                "cannot grow a memory reservation from {} to {bytes} bytes",
                self.bytes
            )));
        }
        let released = self.bytes - bytes;
        if released > 0 {
            let previous = self.tracker.inner.current_bytes.fetch_sub(released, Ordering::AcqRel);
            debug_assert!(previous >= released, "memory reservation accounting underflow");
            self.bytes = bytes;
        }
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let previous = self.tracker.inner.current_bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes, "memory reservation accounting underflow");
    }
}

/// Cloneable cooperative cancellation state checked at chunk boundaries.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Requests cancellation. Repeated calls are harmless.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns [`RecordsError::Cancelled`] after cancellation is observed.
    pub fn check(&self) -> RecordsResult<()> {
        if self.is_cancelled() {
            Err(RecordsError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Direction of an explicit host/device transfer.
#[cfg_attr(feature = "receipt-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "receipt-json", serde(rename_all = "snake_case"))]
#[cfg_attr(feature = "receipt-json", serde(deny_unknown_fields))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingTransferDirection {
    /// Host memory to a device.
    HostToDevice,
    /// Device memory to the host.
    DeviceToHost,
    /// Explicit device-to-device movement.
    DeviceToDevice,
}

/// One named, explicit transfer included in a streaming receipt.
#[cfg_attr(feature = "receipt-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "receipt-json", serde(deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingTransferReceipt {
    /// Stable operation name.
    pub name: String,
    /// Transfer direction.
    pub direction: StreamingTransferDirection,
    /// Bytes moved.
    pub bytes: u64,
}

/// Timing and allocation counters for one named execution phase.
#[cfg_attr(feature = "receipt-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "receipt-json", serde(deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingPhaseReceipt {
    /// Elapsed wall-clock nanoseconds.
    pub elapsed_ns: u64,
    /// Explicitly allocated bytes attributed to the phase.
    pub allocated_bytes: u64,
}

/// Versioned, deterministic accounting receipt for one streaming execution.
#[cfg_attr(feature = "receipt-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "receipt-json", serde(deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingReceipt {
    schema: String,
    version: u32,
    source_id: String,
    input_points: u64,
    output_points: u64,
    chunks_read: u64,
    chunks_written: u64,
    bytes_read: u64,
    bytes_written: u64,
    peak_tracked_bytes: u64,
    spilled_bytes: u64,
    phases: BTreeMap<String, StreamingPhaseReceipt>,
    transfers: Vec<StreamingTransferReceipt>,
}

impl StreamingReceipt {
    /// Creates an empty v1 receipt for a non-empty source identifier.
    pub fn new(source_id: impl Into<String>) -> RecordsResult<Self> {
        let source_id = source_id.into();
        if source_id.trim().is_empty() {
            return Err(RecordsError::InvalidReceipt("source_id must not be empty".into()));
        }
        Ok(Self {
            schema: STREAMING_RECEIPT_SCHEMA.into(),
            version: STREAMING_RECEIPT_VERSION,
            source_id,
            input_points: 0,
            output_points: 0,
            chunks_read: 0,
            chunks_written: 0,
            bytes_read: 0,
            bytes_written: 0,
            peak_tracked_bytes: 0,
            spilled_bytes: 0,
            phases: BTreeMap::new(),
            transfers: Vec::new(),
        })
    }

    /// Records one input chunk using checked counters.
    pub fn record_input_chunk(&mut self, points: u64, bytes: u64) -> RecordsResult<()> {
        checked_add(&mut self.input_points, points, "input_points")?;
        checked_add(&mut self.bytes_read, bytes, "bytes_read")?;
        checked_add(&mut self.chunks_read, 1, "chunks_read")
    }

    /// Records one output chunk using checked counters.
    pub fn record_output_chunk(&mut self, points: u64, bytes: u64) -> RecordsResult<()> {
        checked_add(&mut self.output_points, points, "output_points")?;
        checked_add(&mut self.bytes_written, bytes, "bytes_written")?;
        checked_add(&mut self.chunks_written, 1, "chunks_written")
    }

    /// Records bytes written to explicit temporary spill storage.
    pub fn record_spill(&mut self, bytes: u64) -> RecordsResult<()> {
        checked_add(&mut self.spilled_bytes, bytes, "spilled_bytes")
    }

    /// Captures the peak from an exact memory tracker.
    pub fn capture_memory(&mut self, tracker: &MemoryTracker) {
        self.peak_tracked_bytes = self.peak_tracked_bytes.max(tracker.snapshot().peak_bytes);
    }

    /// Inserts or replaces one named phase receipt.
    pub fn record_phase(
        &mut self,
        name: impl Into<String>,
        elapsed_ns: u64,
        allocated_bytes: u64,
    ) -> RecordsResult<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RecordsError::InvalidReceipt("phase name must not be empty".into()));
        }
        self.phases.insert(name, StreamingPhaseReceipt { elapsed_ns, allocated_bytes });
        Ok(())
    }

    /// Appends one named explicit transfer.
    pub fn record_transfer(
        &mut self,
        name: impl Into<String>,
        direction: StreamingTransferDirection,
        bytes: u64,
    ) -> RecordsResult<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RecordsError::InvalidReceipt("transfer name must not be empty".into()));
        }
        self.transfers.push(StreamingTransferReceipt { name, direction, bytes });
        Ok(())
    }

    /// Validates schema identity and all name constraints.
    pub fn validate(&self) -> RecordsResult<()> {
        if self.schema != STREAMING_RECEIPT_SCHEMA || self.version != STREAMING_RECEIPT_VERSION {
            return Err(RecordsError::InvalidReceipt(format!(
                "expected {STREAMING_RECEIPT_SCHEMA} v{STREAMING_RECEIPT_VERSION}, found {} v{}",
                self.schema, self.version
            )));
        }
        if self.source_id.trim().is_empty() {
            return Err(RecordsError::InvalidReceipt("source_id must not be empty".into()));
        }
        if self.phases.keys().any(|name| name.trim().is_empty()) {
            return Err(RecordsError::InvalidReceipt("phase name must not be empty".into()));
        }
        if self.transfers.iter().any(|transfer| transfer.name.trim().is_empty()) {
            return Err(RecordsError::InvalidReceipt("transfer name must not be empty".into()));
        }
        Ok(())
    }

    /// Serializes this receipt using the versioned JSON contract.
    #[cfg(feature = "receipt-json")]
    pub fn to_json(&self) -> RecordsResult<String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| RecordsError::InvalidReceipt(error.to_string()))
    }

    /// Parses and validates a versioned JSON receipt.
    #[cfg(feature = "receipt-json")]
    pub fn from_json(json: &str) -> RecordsResult<Self> {
        let receipt: Self = serde_json::from_str(json)
            .map_err(|error| RecordsError::InvalidReceipt(error.to_string()))?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Returns the receipt schema version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the stable source identifier.
    #[must_use]
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// Returns input points observed so far.
    #[must_use]
    pub const fn input_points(&self) -> u64 {
        self.input_points
    }

    /// Returns output points observed so far.
    #[must_use]
    pub const fn output_points(&self) -> u64 {
        self.output_points
    }

    /// Returns input chunks observed so far.
    #[must_use]
    pub const fn chunks_read(&self) -> u64 {
        self.chunks_read
    }

    /// Returns output chunks observed so far.
    #[must_use]
    pub const fn chunks_written(&self) -> u64 {
        self.chunks_written
    }

    /// Returns input bytes observed so far.
    #[must_use]
    pub const fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Returns output bytes observed so far.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns the maximum explicitly tracked live memory.
    #[must_use]
    pub const fn peak_tracked_bytes(&self) -> u64 {
        self.peak_tracked_bytes
    }

    /// Returns bytes written to temporary spill storage.
    #[must_use]
    pub const fn spilled_bytes(&self) -> u64 {
        self.spilled_bytes
    }

    /// Returns deterministic phase receipts ordered by name.
    #[must_use]
    pub const fn phases(&self) -> &BTreeMap<String, StreamingPhaseReceipt> {
        &self.phases
    }

    /// Returns explicit transfers in execution order.
    #[must_use]
    pub fn transfers(&self) -> &[StreamingTransferReceipt] {
        &self.transfers
    }
}

fn checked_add(target: &mut u64, value: u64, name: &str) -> RecordsResult<()> {
    *target =
        target.checked_add(value).ok_or_else(|| RecordsError::ReceiptOverflow(name.to_owned()))?;
    Ok(())
}

/// One canonical scale/chunk/budget combination for reproducible comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingWorkload {
    /// Stable workload identifier.
    pub id: &'static str,
    /// Number of generated or selected input points.
    pub point_count: u64,
    /// Requested maximum points per chunk.
    pub chunk_points: usize,
    /// Hard tracked-memory limit.
    pub memory_budget_bytes: u64,
}

const STREAMING_WORKLOADS: [StreamingWorkload; 9] = [
    workload("stream-1m-16k", 1_000_000, 16_384, 64),
    workload("stream-1m-64k", 1_000_000, 65_536, 64),
    workload("stream-1m-256k", 1_000_000, 262_144, 128),
    workload("stream-10m-16k", 10_000_000, 16_384, 64),
    workload("stream-10m-64k", 10_000_000, 65_536, 64),
    workload("stream-10m-256k", 10_000_000, 262_144, 128),
    workload("stream-100m-16k", 100_000_000, 16_384, 64),
    workload("stream-100m-64k", 100_000_000, 65_536, 64),
    workload("stream-100m-256k", 100_000_000, 262_144, 128),
];

const fn workload(
    id: &'static str,
    point_count: u64,
    chunk_points: usize,
    memory_mib: u64,
) -> StreamingWorkload {
    StreamingWorkload {
        id,
        point_count,
        chunk_points,
        memory_budget_bytes: memory_mib * 1024 * 1024,
    }
}

/// Returns the canonical 1M/10M/100M streaming workload matrix.
#[must_use]
pub const fn canonical_streaming_workloads() -> &'static [StreamingWorkload] {
    &STREAMING_WORKLOADS
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[cfg(feature = "receipt-json")]
    use super::STREAMING_RECEIPT_SCHEMA;
    use super::{
        canonical_streaming_workloads, CancellationToken, MemoryBudget, MemoryTracker,
        StreamOptions, StreamingReceipt, StreamingTransferDirection,
    };
    use crate::RecordsError;

    #[test]
    fn memory_budget_is_fail_closed_and_drop_releases() {
        let tracker = MemoryTracker::new(MemoryBudget::new(100).unwrap());
        let first = tracker.try_reserve(60).unwrap();
        assert_eq!(first.bytes(), 60);
        let error = tracker.try_reserve(41).unwrap_err();
        assert!(matches!(
            error,
            RecordsError::MemoryBudgetExceeded { requested: 41, current: 60, limit: 100 }
        ));
        assert_eq!(tracker.snapshot().current_bytes, 60);
        drop(first);
        assert_eq!(tracker.snapshot().current_bytes, 0);
        assert_eq!(tracker.snapshot().peak_bytes, 60);
    }

    #[test]
    fn reservation_can_shrink_but_never_grow() {
        let tracker = MemoryTracker::new(MemoryBudget::new(100).unwrap());
        let mut reservation = tracker.try_reserve(80).unwrap();
        reservation.shrink_to(30).unwrap();
        assert_eq!(reservation.bytes(), 30);
        assert_eq!(tracker.snapshot().current_bytes, 30);
        assert!(reservation.shrink_to(31).is_err());
        assert_eq!(tracker.snapshot().current_bytes, 30);
    }

    #[test]
    fn concurrent_reservations_never_exceed_limit() {
        let tracker = MemoryTracker::new(MemoryBudget::new(100).unwrap());
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let tracker = tracker.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let reservation = tracker.try_reserve(60);
                barrier.wait();
                reservation
            }));
        }
        barrier.wait();
        barrier.wait();
        let reservations =
            handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>();
        assert_eq!(reservations.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(tracker.snapshot().peak_bytes <= 100);
    }

    #[test]
    fn cancellation_is_clone_visible() {
        let token = CancellationToken::default();
        let worker = token.clone();
        token.cancel();
        assert!(worker.is_cancelled());
        assert!(matches!(worker.check(), Err(RecordsError::Cancelled)));
    }

    #[test]
    fn options_reject_zero_chunk_size() {
        let error = StreamOptions::new(0, MemoryBudget::default()).unwrap_err();
        assert!(matches!(error, RecordsError::InvalidConfiguration(_)));
    }

    #[test]
    fn receipt_accounts_named_work_without_hidden_transfers() {
        let tracker = MemoryTracker::new(MemoryBudget::new(1024).unwrap());
        let reservation = tracker.try_reserve(512).unwrap();
        let mut receipt = StreamingReceipt::new("synthetic://one-chunk").unwrap();
        receipt.record_input_chunk(10, 120).unwrap();
        receipt.record_output_chunk(4, 48).unwrap();
        receipt.record_phase("transform", 99, 512).unwrap();
        receipt
            .record_transfer("explicit-upload", StreamingTransferDirection::HostToDevice, 120)
            .unwrap();
        receipt.capture_memory(&tracker);
        drop(reservation);
        receipt.validate().unwrap();
        assert_eq!(receipt.input_points(), 10);
        assert_eq!(receipt.output_points(), 4);
        assert_eq!(receipt.peak_tracked_bytes(), 512);
        assert_eq!(receipt.transfers().len(), 1);
    }

    #[test]
    fn receipt_counter_overflow_is_denied_without_wrapping() {
        let mut receipt = StreamingReceipt::new("synthetic://overflow").unwrap();
        receipt.input_points = u64::MAX;
        assert!(matches!(
            receipt.record_input_chunk(1, 0),
            Err(RecordsError::ReceiptOverflow(name)) if name == "input_points"
        ));
        assert_eq!(receipt.input_points(), u64::MAX);
    }

    #[cfg(feature = "receipt-json")]
    #[test]
    fn json_receipt_roundtrip_is_versioned() {
        let mut receipt = StreamingReceipt::new("copc://autzen").unwrap();
        receipt.record_input_chunk(32, 384).unwrap();
        let json = receipt.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema"], STREAMING_RECEIPT_SCHEMA);
        assert_eq!(value["version"], 1);
        assert_eq!(StreamingReceipt::from_json(&json).unwrap(), receipt);
    }

    #[cfg(feature = "receipt-json")]
    #[test]
    fn json_receipt_rejects_unknown_fields_and_versions() {
        let receipt = StreamingReceipt::new("copc://autzen").unwrap();
        let mut value = serde_json::to_value(&receipt).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(StreamingReceipt::from_json(&value.to_string()).is_err());

        let mut value = serde_json::to_value(&receipt).unwrap();
        value["version"] = serde_json::json!(2);
        assert!(StreamingReceipt::from_json(&value.to_string()).is_err());
    }

    #[test]
    fn canonical_workloads_cover_scale_and_chunk_matrix() {
        let workloads = canonical_streaming_workloads();
        assert_eq!(workloads.len(), 9);
        for points in [1_000_000, 10_000_000, 100_000_000] {
            let chunks = workloads
                .iter()
                .filter(|workload| workload.point_count == points)
                .map(|workload| workload.chunk_points)
                .collect::<Vec<_>>();
            assert_eq!(chunks, [16_384, 65_536, 262_144]);
        }
    }
}

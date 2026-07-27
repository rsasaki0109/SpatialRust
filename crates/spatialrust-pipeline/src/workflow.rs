//! Type-erased, metered bounded-memory streaming workflows.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use spatialrust_math::Mat4;
use spatialrust_records::{
    BoundedSpatialRecordSink, BoundedSpatialRecordSource, CancellationToken, MemoryTracker,
    RecordsError, RecordsResult, SchemaDescriptor, SpatialRecordChunk, StreamOptions,
    StreamingReceipt,
};

use crate::{ChunkMapSource, StreamingVoxelConfig, StreamingVoxelSource};

type ReceiptState = Arc<Mutex<StreamingReceipt>>;

/// Composable bounded-memory point-cloud stream.
pub struct StreamingPipeline {
    source: Box<dyn BoundedSpatialRecordSource>,
    receipt: ReceiptState,
}

impl StreamingPipeline {
    /// Starts a workflow and meters chunks read from the original source.
    pub fn new(
        source: impl BoundedSpatialRecordSource + 'static,
        source_id: impl Into<String>,
    ) -> RecordsResult<Self> {
        let receipt = Arc::new(Mutex::new(StreamingReceipt::new(source_id)?));
        let source = MeteredInputSource { source, receipt: receipt.clone() };
        Ok(Self { source: Box::new(source), receipt })
    }

    /// Returns the output schema.
    #[must_use]
    pub fn schema(&self) -> &SchemaDescriptor {
        self.source.schema()
    }

    /// Returns the shared cooperative cancellation token.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.source.cancellation_token()
    }

    /// Adds an inclusive axis-aligned crop.
    pub fn crop(self, min: [f32; 3], max: [f32; 3], invert: bool) -> RecordsResult<Self> {
        let source = ChunkMapSource::crop(self.source, min, max, invert)?;
        Ok(Self { source: Box::new(source), receipt: self.receipt })
    }

    /// Adds an affine position/normal transform.
    pub fn transform(self, transform: Mat4<f32>) -> RecordsResult<Self> {
        let source = ChunkMapSource::transform(self.source, transform)?;
        Ok(Self { source: Box::new(source), receipt: self.receipt })
    }

    /// Adds deterministic global voxel aggregation backed by bounded spool storage.
    pub fn voxel(self, config: StreamingVoxelConfig) -> RecordsResult<Self> {
        let started = Instant::now();
        let source = StreamingVoxelSource::try_build(self.source, config)?;
        let spill_bytes = source.spool_bytes();
        {
            let mut receipt = lock_receipt(&self.receipt)?;
            receipt.record_spill(spill_bytes)?;
            receipt.record_phase("voxel", elapsed_ns(started), spill_bytes)?;
            receipt.capture_memory(source.memory_tracker());
        }
        Ok(Self { source: Box::new(source), receipt: self.receipt })
    }

    /// Drains the workflow into a synchronous bounded sink and returns its receipt.
    pub fn run_to_sink(
        self,
        sink: &mut dyn BoundedSpatialRecordSink,
    ) -> RecordsResult<StreamingReceipt> {
        let mut stream = self.into_iter();
        for chunk in stream.by_ref() {
            let chunk = chunk?;
            sink.write_chunk(&chunk)?;
        }
        sink.finish()?;
        stream.receipt()
    }
}

impl IntoIterator for StreamingPipeline {
    type Item = RecordsResult<SpatialRecordChunk>;
    type IntoIter = StreamingPipelineIter;

    fn into_iter(self) -> Self::IntoIter {
        let tracker = self.source.memory_tracker().clone();
        StreamingPipelineIter {
            source: self.source,
            receipt: self.receipt,
            tracker,
            completed: false,
        }
    }
}

/// Pull iterator that meters final output and exposes a live receipt snapshot.
pub struct StreamingPipelineIter {
    source: Box<dyn BoundedSpatialRecordSource>,
    receipt: ReceiptState,
    tracker: MemoryTracker,
    completed: bool,
}

impl StreamingPipelineIter {
    /// Returns the shared cooperative cancellation token.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.source.cancellation_token()
    }

    /// Clones the receipt as observed at the latest completed chunk boundary.
    pub fn receipt(&self) -> RecordsResult<StreamingReceipt> {
        let mut receipt = lock_receipt(&self.receipt)?;
        receipt.capture_memory(&self.tracker);
        Ok(receipt.clone())
    }
}

impl Iterator for StreamingPipelineIter {
    type Item = RecordsResult<SpatialRecordChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed {
            return None;
        }
        let next = self.source.next_chunk();
        match next {
            Some(Ok(chunk)) => {
                let points = match u64::try_from(chunk.record().cloud().len()) {
                    Ok(points) => points,
                    Err(_) => {
                        self.completed = true;
                        return Some(Err(RecordsError::ReceiptOverflow(
                            "pipeline output point count".into(),
                        )));
                    }
                };
                if let Err(error) = lock_receipt(&self.receipt).and_then(|mut receipt| {
                    receipt.record_output_chunk(points, chunk.tracked_bytes())
                }) {
                    self.completed = true;
                    return Some(Err(error));
                }
                Some(Ok(chunk))
            }
            Some(Err(error)) => {
                self.completed = true;
                Some(Err(error))
            }
            None => {
                self.completed = true;
                if let Ok(mut receipt) = lock_receipt(&self.receipt) {
                    receipt.capture_memory(&self.tracker);
                }
                None
            }
        }
    }
}

struct MeteredInputSource<S> {
    source: S,
    receipt: ReceiptState,
}

impl<S: BoundedSpatialRecordSource> BoundedSpatialRecordSource for MeteredInputSource<S> {
    fn schema(&self) -> &SchemaDescriptor {
        self.source.schema()
    }

    fn options(&self) -> &StreamOptions {
        self.source.options()
    }

    fn memory_tracker(&self) -> &MemoryTracker {
        self.source.memory_tracker()
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.source.cancellation_token()
    }

    fn max_chunk_bytes(&self) -> u64 {
        self.source.max_chunk_bytes()
    }

    fn next_chunk(&mut self) -> Option<RecordsResult<SpatialRecordChunk>> {
        match self.source.next_chunk()? {
            Ok(chunk) => {
                let points = match u64::try_from(chunk.record().cloud().len()) {
                    Ok(points) => points,
                    Err(_) => {
                        return Some(Err(RecordsError::ReceiptOverflow(
                            "pipeline input point count".into(),
                        )));
                    }
                };
                match lock_receipt(&self.receipt).and_then(|mut receipt| {
                    receipt.record_input_chunk(points, chunk.tracked_bytes())
                }) {
                    Ok(()) => Some(Ok(chunk)),
                    Err(error) => Some(Err(error)),
                }
            }
            Err(error) => Some(Err(error)),
        }
    }
}

fn lock_receipt(receipt: &ReceiptState) -> RecordsResult<MutexGuard<'_, StreamingReceipt>> {
    receipt
        .lock()
        .map_err(|_| RecordsError::InvalidReceipt("streaming receipt lock poisoned".into()))
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::StreamingPipeline;
    use spatialrust_core::{HasPositions3, PointCloudBuilder, StandardSchemas};
    use spatialrust_records::{
        CancellationToken, MemoryBudget, RecyclingMemoryChunkSource, SchemaDescriptor,
        SchemaVersion, StreamOptions,
    };

    fn source() -> RecyclingMemoryChunkSource {
        let mut builder = PointCloudBuilder::xyz();
        for point in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]] {
            builder.push_point(point).unwrap();
        }
        let schema = SchemaDescriptor::try_new(
            "workflow.xyz",
            SchemaVersion::new(1, 0),
            StandardSchemas::point_xyz(),
        )
        .unwrap();
        let options = StreamOptions::new(2, MemoryBudget::new(4096).unwrap()).unwrap();
        RecyclingMemoryChunkSource::try_new(
            schema,
            builder.build().unwrap(),
            options,
            CancellationToken::default(),
        )
        .unwrap()
    }

    #[test]
    fn meters_input_and_output_around_composable_crop() {
        let pipeline = StreamingPipeline::new(source(), "memory")
            .unwrap()
            .crop([0.5, -1.0, -1.0], [2.0, 1.0, 1.0], false)
            .unwrap();
        let mut stream = pipeline.into_iter();
        let mut x = Vec::new();
        for chunk in stream.by_ref() {
            let chunk = chunk.unwrap();
            x.extend_from_slice(chunk.record().cloud().positions3().unwrap().0);
        }
        assert_eq!(x, [1.0, 2.0]);
        let receipt = stream.receipt().unwrap();
        assert_eq!(receipt.input_points(), 3);
        assert_eq!(receipt.output_points(), 2);
        assert_eq!(receipt.chunks_read(), 2);
        assert_eq!(receipt.chunks_written(), 2);
    }

    #[test]
    fn iterator_observes_shared_cancellation() {
        let pipeline = StreamingPipeline::new(source(), "memory").unwrap();
        let token = pipeline.cancellation_token();
        token.cancel();
        let mut stream = pipeline.into_iter();
        assert!(stream.next().unwrap().is_err());
        assert!(stream.next().is_none());
    }
}

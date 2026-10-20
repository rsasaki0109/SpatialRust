//! Incremental scalar-capacity admission for the full-bag example.
//! Source leases, retained episode columns and assembly requests are distinct.
//! This accounts for scalar storage, not allocator overhead or process RSS.

use super::ClockCorrection;
use spatialrust_core::Timestamp;
use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud};
use spatialrust_records::{
    record_storage_bytes, RecordsError, RecordsResult, SpatialRecord, SpatialRecordChunk,
};
use spatialrust_sync::{
    ClockDomain, ClockId, MemoryEpisodeBuilder, StampedRecord, StampedTime, SyncError, SyncQuality,
};
use std::{collections::BTreeSet, error::Error};

type CollectionResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CollectionUsage {
    pub(super) peak_retained_and_pending_bytes: u64,
    pub(super) max_assembly_request_bytes: u64,
}

#[derive(Debug)]
pub(super) struct CollectedChunks {
    pub(super) record_count: u64,
    pub(super) chunk_count: u64,
    pub(super) frame_ids: BTreeSet<String>,
    pub(super) usage: CollectionUsage,
}

struct PendingScan {
    stamp: u64,
    frame: String,
    chunks: Vec<SpatialRecord>,
    points: u64,
    bytes: u64,
}

pub(super) fn collect_chunks(
    mut next: impl FnMut() -> Option<RecordsResult<SpatialRecordChunk>>,
    topic: &str,
    clock: &mut ClockCorrection,
    builder: &mut MemoryEpisodeBuilder,
) -> CollectionResult<CollectedChunks> {
    let initial_records = builder.len();
    let mut current: Option<PendingScan> = None;
    let mut stats = CollectedChunks {
        record_count: 0,
        chunk_count: 0,
        frame_ids: BTreeSet::new(),
        usage: CollectionUsage {
            peak_retained_and_pending_bytes: builder.bytes(),
            ..CollectionUsage::default()
        },
    };
    while let Some(chunk) = next() {
        let chunk = chunk?;
        stats.chunk_count =
            stats.chunk_count.checked_add(1).ok_or("full-bag chunk count overflow")?;
        let record = chunk.record();
        if record.cloud().is_empty() {
            return Err(
                RecordsError::InvalidChunk("collector chunks must not be empty".into()).into()
            );
        }
        let stamp = record.metadata().timestamp.as_nanos();
        let frame = &record.metadata().frame_id.0;
        if matches!(current.as_ref(), Some(pending) if pending.stamp != stamp || &pending.frame != frame)
        {
            finish_pending(&mut current, topic, clock, builder, &mut stats)?;
        }
        let pending = current.get_or_insert_with(|| PendingScan {
            stamp,
            frame: frame.clone(),
            chunks: Vec::new(),
            points: 0,
            bytes: 0,
        });
        if pending.chunks.first().is_some_and(|first| first.schema() != record.schema()) {
            return Err(RecordsError::SchemaMismatch(
                "scan chunks require a homogeneous schema".into(),
            )
            .into());
        }
        // The leased source record remains borrowed. Admission precedes the
        // explicit owned CPU copy; its source reservation is never detached.
        admit_pending(builder, pending.points, pending.bytes, record)?;
        let owned = record.clone();
        let (points, bytes) = admit_pending(builder, pending.points, pending.bytes, &owned)?;
        pending.chunks.push(owned);
        pending.points = points;
        pending.bytes = bytes;
        stats.usage.peak_retained_and_pending_bytes = stats
            .usage
            .peak_retained_and_pending_bytes
            .max(builder.bytes().checked_add(bytes).ok_or("retained scalar capacity overflow")?);
    }
    finish_pending(&mut current, topic, clock, builder, &mut stats)?;
    stats.record_count = u64::try_from(
        builder.len().checked_sub(initial_records).ok_or("collector record count decreased")?,
    )?;
    Ok(stats)
}

pub(super) fn checked_total(
    resource: &'static str,
    current: u64,
    requested: u64,
    limit: u64,
) -> CollectionResult<u64> {
    current.checked_add(requested).filter(|next| *next <= limit).ok_or_else(|| {
        SyncError::EpisodeLimitExceeded { resource, requested, current, limit }.into()
    })
}

fn admit_pending(
    builder: &MemoryEpisodeBuilder,
    points: u64,
    bytes: u64,
    record: &SpatialRecord,
) -> CollectionResult<(u64, u64)> {
    let limits = builder.limits();
    checked_total("records", u64::try_from(builder.len())?, 1, limits.max_records)?;
    let points =
        checked_total("points", points, u64::try_from(record.cloud().len())?, limits.max_points)?;
    checked_total("points", builder.points(), points, limits.max_points)?;
    let bytes = checked_total("bytes", bytes, record_storage_bytes(record)?, limits.max_bytes)?;
    checked_total("bytes", builder.bytes(), bytes, limits.max_bytes)?;
    Ok((points, bytes))
}

fn finish_pending(
    current: &mut Option<PendingScan>,
    topic: &str,
    clock: &mut ClockCorrection,
    builder: &mut MemoryEpisodeBuilder,
    stats: &mut CollectedChunks,
) -> CollectionResult<()> {
    let Some(pending) = current.take() else {
        return Ok(());
    };
    let (record, request_bytes) = assemble(pending.chunks, pending.bytes)?;
    stats.usage.max_assembly_request_bytes =
        stats.usage.max_assembly_request_bytes.max(request_bytes);
    let corrected_stamp = clock.correct(pending.stamp)?;
    let stamp = StampedTime {
        clock: ClockId::new(clock.clock_id.clone()),
        domain: ClockDomain::External,
        timestamp: Timestamp::from_nanos(corrected_stamp),
        quality: SyncQuality {
            offset_ns: clock.offset_i64()?,
            uncertainty_ns: clock.uncertainty_ns,
            estimated: true,
        },
    };
    // The final output's actual allocated column capacities, including any
    // allocator over-reservation, must pass the existing builder gate too.
    builder.push(StampedRecord::new(topic, stamp, record))?;
    stats.usage.peak_retained_and_pending_bytes =
        stats.usage.peak_retained_and_pending_bytes.max(builder.bytes());
    stats.frame_ids.insert(pending.frame);
    Ok(())
}

fn assemble(
    mut chunks: Vec<SpatialRecord>,
    admitted_bytes: u64,
) -> CollectionResult<(SpatialRecord, u64)> {
    if chunks.len() == 1 {
        let record = chunks.pop().ok_or("missing pending chunk")?;
        let provenance = record.provenance().clone().without_sequence();
        return Ok((record.with_provenance(provenance), 0));
    }
    let first = chunks.first().ok_or("missing pending chunks")?;
    let schema = first.schema().clone();
    let metadata = first.metadata().clone();
    let provenance = first.provenance().clone().without_sequence();
    let mut buffers = PointBufferSet::new();
    let mut requested_bytes = 0_u64;
    for field in schema.point_schema().fields() {
        let elements =
            chunks.iter().try_fold(0_usize, |len, record| -> CollectionResult<usize> {
                len.checked_add(record.cloud().field(&field.name)?.len())
                    .ok_or_else(|| "assembly element count overflow".into())
            })?;
        macro_rules! join_column {
            ($variant:ident, $scalar:ty) => {{
                let bytes = u64::try_from(elements)?
                    .checked_mul(std::mem::size_of::<$scalar>() as u64)
                    .ok_or("assembly byte count overflow")?;
                requested_bytes =
                    requested_bytes.checked_add(bytes).ok_or("assembly byte count overflow")?;
                if requested_bytes > admitted_bytes {
                    return Err("assembly request exceeds admitted scalar capacity".into());
                }
                let mut values = Vec::<$scalar>::new();
                values.try_reserve_exact(elements)?;
                for record in &chunks {
                    let PointBuffer::$variant(source) = record.cloud().field(&field.name)? else {
                        return Err(RecordsError::SchemaMismatch(
                            "assembly column dtype differs".into(),
                        )
                        .into());
                    };
                    values.extend_from_slice(source);
                }
                PointBuffer::$variant(values)
            }};
        }
        let buffer = match first.cloud().field(&field.name)? {
            PointBuffer::F32(_) => join_column!(F32, f32),
            PointBuffer::F64(_) => join_column!(F64, f64),
            PointBuffer::U8(_) => join_column!(U8, u8),
            PointBuffer::U16(_) => join_column!(U16, u16),
            PointBuffer::U32(_) => join_column!(U32, u32),
            PointBuffer::I32(_) => join_column!(I32, i32),
        };
        buffers.insert(field.name.clone(), buffer);
    }
    let cloud = PointCloud::try_from_parts(schema.point_schema().clone(), buffers, metadata)?;
    Ok((SpatialRecord::try_new_with_provenance(schema, cloud, provenance)?, requested_bytes))
}

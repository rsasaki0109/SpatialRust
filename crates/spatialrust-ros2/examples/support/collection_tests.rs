//! Synthetic collector admission/IO tests. No physical calibration evidence.

use super::*;
use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas};
use spatialrust_records::{
    record_storage_bytes, ChunkIdentity, MemoryTracker, RecordProvenance, RecordsError,
    SchemaVersion, SpatialRecord,
};
use spatialrust_sync::SyncError;
use std::{cell::Cell, rc::Rc};

fn record(frame: &str, stamp: u64, len: usize, capacity: usize, intensity: bool) -> SpatialRecord {
    let mut buffers = PointBufferSet::new();
    for (name, base) in [("x", 1.0), ("y", 2.0), ("z", 3.0)] {
        let mut values = Vec::with_capacity(capacity);
        values.extend((0..len).map(|i| base + i as f32));
        buffers.insert(name, PointBuffer::from_f32(values));
    }
    if intensity {
        buffers.insert("intensity", PointBuffer::from_f32(vec![7.0; len]));
    }
    let schema =
        if intensity { StandardSchemas::point_xyzi() } else { StandardSchemas::point_xyz() };
    let cloud = PointCloud::try_from_parts(
        schema,
        buffers,
        SpatialMetadata::new(frame, Timestamp::from_nanos(stamp)),
    )
    .unwrap();
    SpatialRecord::try_from_cloud_with_provenance(
        "fixture",
        SchemaVersion::new(1, 0),
        cloud,
        RecordProvenance::try_new("synthetic")
            .unwrap()
            .with_source_uri("synthetic://collector")
            .with_stream_id("fixture")
            .with_sequence(Some(stamp)),
    )
    .unwrap()
}

fn xyz(stamp: u64) -> SpatialRecord {
    record("front", stamp, 4, 4, false)
}

fn clock() -> ClockCorrection {
    ClockCorrection {
        clock_id: "synthetic-external".into(),
        uncertainty_ns: 2,
        offset_ns: 10.0,
        drift_ppm: 0.0,
        anchor_nanos: None,
    }
}

fn builder(records: u64, points: u64, bytes: u64) -> MemoryEpisodeBuilder {
    MemoryEpisodeBuilder::try_new(EpisodeLimits::new(records, points, bytes)).unwrap()
}

fn spy(
    items: Vec<RecordsResult<SpatialRecord>>,
) -> (impl FnMut() -> Option<RecordsResult<SpatialRecordChunk>>, Rc<Cell<usize>>, MemoryTracker) {
    let count = Rc::new(Cell::new(0));
    let observed = count.clone();
    let tracker = MemoryTracker::new(MemoryBudget::new(1_048_576).unwrap());
    let leases = tracker.clone();
    let mut items = items.into_iter();
    let next = move || {
        let item = items.next()?;
        let index = observed.get();
        observed.set(index + 1);
        Some(item.and_then(|record| {
            let bytes = record_storage_bytes(&record)?;
            let reservation = leases.try_reserve(bytes)?;
            SpatialRecordChunk::try_from_reserved(
                ChunkIdentity { sequence: index as u64, point_offset: 0 },
                record,
                reservation,
            )
        }))
    };
    (next, count, tracker)
}

fn assert_limit(error: &(dyn Error + 'static), expected: &str) {
    assert!(
        matches!(error.downcast_ref::<SyncError>(),
        Some(SyncError::EpisodeLimitExceeded { resource, .. }) if *resource == expected),
        "{error}"
    );
}

#[test]
fn record_limit_stops_before_next_scan() {
    let (next, count, tracker) = spy((1..=4).map(|n| Ok(xyz(n))).collect());
    let mut retained = builder(1, 100, 4096);
    let error = collect_chunks(next, "/front", &mut clock(), &mut retained).unwrap_err();
    assert_limit(error.as_ref(), "records");
    assert_eq!(count.get(), 2);
    assert_eq!(tracker.snapshot().current_bytes, 0);
}

#[test]
fn record_limit_is_shared_across_topics() {
    let mut retained = builder(2, 100, 4096);
    let (front, _, _) = spy(vec![Ok(xyz(1)), Ok(xyz(2))]);
    collect_chunks(front, "/front", &mut clock(), &mut retained).unwrap();
    let (rear, count, _) = spy(vec![Ok(xyz(3)), Ok(xyz(4)), Ok(xyz(5))]);
    let error = collect_chunks(rear, "/rear", &mut clock(), &mut retained).unwrap_err();
    assert_limit(error.as_ref(), "records");
    assert_eq!(count.get(), 1);
    assert_eq!(retained.len(), 2);
}

#[test]
fn pending_points_stop_before_next_chunk() {
    let (next, count, _) = spy(vec![Ok(xyz(1)), Ok(xyz(1)), Ok(xyz(1))]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 7, 4096)).unwrap_err();
    assert_limit(error.as_ref(), "points");
    assert_eq!(count.get(), 2);
}

#[test]
fn capacity_is_checked_before_retaining_input() {
    // Payload is only 48 bytes; source-owned capacity is 1,536 bytes.
    let (next, count, tracker) = spy(vec![Ok(record("front", 1, 4, 128, false)), Ok(xyz(2))]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 100, 96)).unwrap_err();
    assert_limit(error.as_ref(), "bytes");
    assert_eq!(count.get(), 1);
    assert_eq!(tracker.snapshot().current_bytes, 0);
}

#[test]
fn pending_capacity_stops_before_next_chunk() {
    let (next, count, _) = spy(vec![Ok(xyz(1)), Ok(xyz(1)), Ok(xyz(1))]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 100, 80)).unwrap_err();
    assert_limit(error.as_ref(), "bytes");
    assert_eq!(count.get(), 2);
}

#[test]
fn capacity_is_shared_across_topics() {
    let mut retained = builder(10, 100, 100);
    let (front, _, _) = spy(vec![Ok(xyz(1)), Ok(xyz(2))]);
    collect_chunks(front, "/front", &mut clock(), &mut retained).unwrap();
    let (rear, count, _) = spy(vec![Ok(xyz(3)), Ok(xyz(4))]);
    let error = collect_chunks(rear, "/rear", &mut clock(), &mut retained).unwrap_err();
    assert_limit(error.as_ref(), "bytes");
    assert_eq!(count.get(), 1);
    assert_eq!(retained.bytes(), 96);
}

#[test]
fn oversized_first_points_do_not_drain_tail() {
    let (next, count, _) = spy(vec![Ok(xyz(1)), Ok(xyz(2)), Ok(xyz(3))]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 3, 4096)).unwrap_err();
    assert_limit(error.as_ref(), "points");
    assert_eq!(count.get(), 1);
}

#[test]
fn exact_limits_reassemble_values_and_correct_only_stamped_time() {
    let first = xyz(10);
    let expected_metadata = first.metadata().clone();
    let expected_provenance = first.provenance().clone().without_sequence();
    let (next, count, tracker) = spy(vec![Ok(first), Ok(xyz(10)), Ok(xyz(20))]);
    let mut retained = builder(2, 12, 144);
    let stats = collect_chunks(next, "/front", &mut clock(), &mut retained).unwrap();
    assert_eq!((stats.record_count, stats.chunk_count, count.get()), (2, 3, 3));
    assert_eq!((retained.points(), retained.bytes()), (12, 144));
    assert_eq!(tracker.snapshot().current_bytes, 0);
    let episode = retained.finish();
    let records = episode.records();
    assert_eq!(records[0].record.metadata(), &expected_metadata);
    assert_eq!(records[0].record.provenance(), &expected_provenance);
    assert_eq!(records[0].stamp.as_nanos(), 20);
    assert_eq!(records[0].stamp.quality.uncertainty_ns, 2);
    assert_eq!(
        records[0].record.cloud().field("x").unwrap().as_f32().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn duplicate_headers_keep_existing_grouping_policy() {
    let (next, _, _) = spy(vec![Ok(xyz(10)), Ok(xyz(10))]);
    let mut retained = builder(1, 8, 96);
    let stats = collect_chunks(next, "/front", &mut clock(), &mut retained).unwrap();
    assert_eq!((stats.record_count, stats.chunk_count), (1, 2));
    assert_eq!(retained.finish().records()[0].record.cloud().len(), 8);
}

#[test]
fn frame_change_flushes_even_with_equal_timestamp() {
    let (next, _, _) = spy(vec![Ok(xyz(10)), Ok(record("other", 10, 4, 4, false))]);
    let mut retained = builder(2, 8, 96);
    let stats = collect_chunks(next, "/front", &mut clock(), &mut retained).unwrap();
    assert_eq!(stats.record_count, 2);
    assert_eq!(stats.frame_ids, BTreeSet::from(["front".into(), "other".into()]));
}

#[test]
fn empty_chunk_rejects_before_reading_tail() {
    let (next, count, _) = spy(vec![Ok(record("front", 1, 0, 0, false)), Ok(xyz(2))]);
    assert!(collect_chunks(next, "/front", &mut clock(), &mut builder(10, 100, 4096)).is_err());
    assert_eq!(count.get(), 1);
}

#[test]
fn schema_error_releases_leases_and_does_not_read_tail() {
    let (next, count, tracker) =
        spy(vec![Ok(xyz(1)), Ok(record("front", 1, 4, 4, true)), Ok(xyz(2))]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 100, 4096)).unwrap_err();
    assert!(error.to_string().to_lowercase().contains("schema"));
    assert_eq!(count.get(), 2);
    assert_eq!(tracker.snapshot().current_bytes, 0);
}

#[test]
fn source_error_propagates_without_reading_tail() {
    let (next, count, tracker) = spy(vec![
        Ok(xyz(1)),
        Err(RecordsError::InvalidChunk("injected failure".into())),
        Ok(xyz(2)),
    ]);
    let error =
        collect_chunks(next, "/front", &mut clock(), &mut builder(10, 100, 4096)).unwrap_err();
    assert!(error.to_string().contains("injected failure"));
    assert_eq!(count.get(), 2);
    assert_eq!(tracker.snapshot().current_bytes, 0);
}

#[test]
fn xyzi_preserves_intensity_metadata_and_first_lineage() {
    let first = record("front", 10, 4, 4, true);
    let metadata = first.metadata().clone();
    let provenance = first.provenance().clone().without_sequence();
    let (next, _, _) = spy(vec![Ok(first), Ok(record("front", 10, 4, 4, true))]);
    let mut retained = builder(1, 8, 128);
    collect_chunks(next, "/front", &mut clock(), &mut retained).unwrap();
    let episode = retained.finish();
    let record = &episode.records()[0].record;
    assert_eq!(record.metadata(), &metadata);
    assert_eq!(record.provenance(), &provenance);
    assert_eq!(record.cloud().field("intensity").unwrap().as_f32().unwrap(), &[7.0; 8]);
}

#[test]
fn shared_episode_retains_deterministic_timestamp_topic_order() {
    let mut retained = builder(4, 16, 192);
    let mut correction = clock();
    let (front, _, _) = spy(vec![Ok(xyz(20)), Ok(xyz(40))]);
    let (rear, _, _) =
        spy(vec![Ok(record("rear", 10, 4, 4, false)), Ok(record("rear", 20, 4, 4, false))]);
    collect_chunks(front, "/front", &mut correction, &mut retained).unwrap();
    collect_chunks(rear, "/rear", &mut correction, &mut retained).unwrap();
    let episode = retained.finish();
    let timeline: Vec<_> =
        episode.records().iter().map(|r| (r.stamp.as_nanos(), r.topic.as_str())).collect();
    assert_eq!(timeline, vec![(20, "/rear"), (30, "/front"), (30, "/rear"), (50, "/front")]);
}

#[test]
fn sqlite_limit_precedes_malformed_tail_decode() {
    use rusqlite::{params, Connection};
    use spatialrust_runtime::{encode_point_cloud2_xyz, PointCloud2Xyz};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic.db3");
    let db = Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);").unwrap();
    db.execute("INSERT INTO topics VALUES(1,'/front','sensor_msgs/msg/PointCloud2','cdr')", [])
        .unwrap();
    for n in 1..=2 {
        let message = PointCloud2Xyz::try_new("front", 0, n, vec![1.0; 12]).unwrap();
        db.execute(
            "INSERT INTO messages VALUES(?1,1,?1,?2)",
            params![n, encode_point_cloud2_xyz(&message).unwrap()],
        )
        .unwrap();
    }
    db.execute("INSERT INTO messages VALUES(3,1,3,?1)", params![vec![0_u8; 200]]).unwrap();
    drop(db);
    let options = StreamOptions::new(4, MemoryBudget::new(1_048_576).unwrap()).unwrap();
    let error = collect_topic(&path, "/front", options, &mut clock(), &mut builder(10, 3, 4096))
        .unwrap_err();
    assert_limit(error.as_ref(), "points");
}

#[test]
fn rejected_topic_preserves_previously_committed_records() {
    let mut retained = builder(1, 100, 4096);
    let (front, _, _) = spy(vec![Ok(xyz(10))]);
    collect_chunks(front, "/front", &mut clock(), &mut retained).unwrap();
    let before = (retained.len(), retained.points(), retained.bytes());
    let (rear, _, _) = spy(vec![Ok(xyz(20)), Ok(xyz(30))]);
    assert!(collect_chunks(rear, "/rear", &mut clock(), &mut retained).is_err());
    assert_eq!((retained.len(), retained.points(), retained.bytes()), before);
    assert_eq!(retained.finish().records()[0].record.metadata().timestamp.as_nanos(), 10);
}

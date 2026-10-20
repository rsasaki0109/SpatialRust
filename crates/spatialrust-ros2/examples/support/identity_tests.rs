//! Synthetic scan identity admission; these fixtures provide no metrology.

use super::*;
use rusqlite::{params, Connection};
use spatialrust_core::{
    PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_mapping::{StampedPose, Trajectory};
use spatialrust_math::Pose3;
use spatialrust_records::{SchemaVersion, SpatialRecord};
use spatialrust_runtime::{encode_point_cloud2_xyz, PointCloud2Xyz};
use spatialrust_sync::{ClockDomain, MemoryEpisode, StampedTime};

fn sample(topic: &str, frame: &str, raw: u64, corrected: u64) -> StampedRecord {
    let mut buffers = PointBufferSet::new();
    for name in ["x", "y", "z"] {
        buffers.insert(name, PointBuffer::from_f32(vec![1.0; 4]));
    }
    let cloud = PointCloud::try_from_parts(
        StandardSchemas::point_xyz(),
        buffers,
        SpatialMetadata::new(frame, Timestamp::from_nanos(raw)),
    )
    .unwrap();
    StampedRecord::new(
        topic,
        StampedTime::exact(
            "synthetic-external",
            ClockDomain::External,
            Timestamp::from_nanos(corrected),
        ),
        SpatialRecord::try_from_cloud("synthetic", SchemaVersion::new(1, 0), cloud).unwrap(),
    )
}

fn correction() -> ClockCorrection {
    ClockCorrection {
        clock_id: "synthetic-external".into(),
        uncertainty_ns: 2,
        offset_ns: 10.0,
        drift_ppm: 0.0,
        anchor_nanos: None,
    }
}

fn retained() -> MemoryEpisodeBuilder {
    MemoryEpisodeBuilder::try_new(EpisodeLimits::new(100, 1000, 16_384)).unwrap()
}

fn bag(messages: &[(u32, usize)], topic: &str) -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("identity.db3");
    let db = Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);").unwrap();
    db.execute("INSERT INTO topics VALUES(1,?1,'sensor_msgs/msg/PointCloud2','cdr')", [topic])
        .unwrap();
    for (index, &(stamp, points)) in messages.iter().enumerate() {
        let message = PointCloud2Xyz::try_new("front", 0, stamp, vec![1.0; points * 3]).unwrap();
        let id = i64::try_from(index + 1).unwrap();
        db.execute(
            "INSERT INTO messages VALUES(?1,1,?1,?2)",
            params![id, encode_point_cloud2_xyz(&message).unwrap()],
        )
        .unwrap();
    }
    drop(db);
    (directory, path)
}

fn collect(
    path: &Path,
    topic: &str,
    chunk_points: usize,
    builder: &mut MemoryEpisodeBuilder,
) -> Result<CollectedTopic, Box<dyn Error>> {
    collect_topic(
        path,
        topic,
        StreamOptions::new(chunk_points, MemoryBudget::new(1_048_576).unwrap()).unwrap(),
        &mut correction(),
        builder,
    )
}

fn trajectory(records: &[StampedRecord]) -> Trajectory {
    let mut trajectory = Trajectory::new();
    for (index, record) in records.iter().enumerate() {
        trajectory
            .push(StampedPose::new(
                record.stamp.clone(),
                Pose3::new(Isometry3::new(
                    Quat::identity(),
                    Vec3::new(index as f32 * 3.0, 0.0, 0.0),
                )),
            ))
            .unwrap();
    }
    trajectory
}

fn lookup(
    records: &[StampedRecord],
    trajectory: &Trajectory,
) -> Result<BTreeMap<u64, Isometry3<f32>>, Box<dyn Error>> {
    scan_identity::front_pose_lookup(&records.iter().collect::<Vec<_>>(), trajectory)
}

fn assert_rejected<T: std::fmt::Debug>(result: Result<T, Box<dyn Error>>, text: &str) {
    let error = result.unwrap_err();
    assert!(error.to_string().contains(text), "{error}");
}

#[test]
fn sqlite_distinct_messages_with_identical_headers_are_rejected() {
    for topic in ["/front", "/rear"] {
        let (_directory, path) = bag(&[(10, 5), (10, 5)], topic);
        assert_rejected(collect(&path, topic, 2, &mut retained()), "message count");
    }
}

#[test]
fn sqlite_one_message_many_chunks_is_one_scan() {
    let (_directory, path) = bag(&[(10, 5)], "/front");
    let mut builder = retained();
    let stats = collect(&path, "/front", 2, &mut builder).unwrap();
    assert_eq!((stats.bag_message_count, stats.record_count, stats.chunk_count), (1, 1, 3));
    assert_eq!((builder.len(), builder.points()), (1, 5));
    let episode = builder.finish();
    let scan = &episode.records()[0];
    assert_eq!(scan.stamp.as_nanos(), 20);
    assert_eq!(scan.record.metadata().timestamp.as_nanos(), 10);
}

#[test]
fn sqlite_distinct_multichunk_messages_remain_complete() {
    let (_directory, path) = bag(&[(10, 5), (20, 3)], "/front");
    let mut builder = retained();
    let stats = collect(&path, "/front", 2, &mut builder).unwrap();
    assert_eq!((stats.bag_message_count, stats.record_count, stats.chunk_count), (2, 2, 5));
    assert_eq!(builder.points(), 8);
}

#[test]
fn sqlite_skipped_empty_message_cannot_claim_completeness() {
    let (_directory, path) = bag(&[(10, 4), (20, 0)], "/front");
    assert_rejected(collect(&path, "/front", 2, &mut retained()), "message count");
}

#[test]
fn sqlite_nonconsecutive_repeat_is_rejected_after_sorting() {
    let (_directory, path) = bag(&[(10, 4), (20, 4), (10, 4)], "/front");
    let mut builder = retained();
    assert_eq!(collect(&path, "/front", 2, &mut builder).unwrap().record_count, 3);
    assert_rejected(
        scan_identity::topic_records(&builder.finish(), &TopicId::new("/front")),
        "strictly increasing",
    );
}

#[test]
fn duplicate_topic_timestamps_are_rejected() {
    for topic in ["/front", "/rear"] {
        let episode = MemoryEpisode::from_records(vec![
            sample(topic, "sensor", 10, 20),
            sample(topic, "sensor", 30, 20),
        ]);
        assert_rejected(
            scan_identity::topic_records(&episode, &TopicId::new(topic)),
            "strictly increasing",
        );
    }
}

#[test]
fn correction_rounding_collision_is_rejected_without_rewriting_raw_time() {
    let raw = 1_600_000_000_000_000_000;
    let mut clock = correction();
    let a = clock.correct(raw).unwrap();
    let b = clock.correct(raw + 1).unwrap();
    assert_eq!(a, b);
    let episode = MemoryEpisode::from_records(vec![
        sample("/front", "sensor", raw, a),
        sample("/front", "sensor", raw + 1, b),
    ]);
    assert_ne!(
        episode.records()[0].record.metadata().timestamp,
        episode.records()[1].record.metadata().timestamp
    );
    assert_rejected(
        scan_identity::topic_records(&episode, &TopicId::new("/front")),
        "strictly increasing",
    );
}

#[test]
fn equal_cross_topic_times_remain_valid() {
    let episode = MemoryEpisode::from_records(vec![
        sample("/front", "front", 10, 20),
        sample("/rear", "rear", 10, 20),
        sample("/front", "front", 30, 40),
    ]);
    for (topic, count) in [("/front", 2), ("/rear", 1)] {
        assert_eq!(
            scan_identity::topic_records(&episode, &TopicId::new(topic)).unwrap().len(),
            count
        );
    }
}

#[test]
fn trajectory_count_mismatch_is_rejected() {
    let records = vec![sample("/front", "front", 10, 20), sample("/front", "front", 30, 40)];
    assert_rejected(lookup(&records, &trajectory(&records[..1])), "sample count");
    assert_rejected(lookup(&records[..1], &trajectory(&records)), "sample count");
}

#[test]
fn duplicate_pose_keys_cannot_overwrite() {
    let records = vec![sample("/front", "front", 10, 20), sample("/front", "front", 30, 20)];
    assert_rejected(lookup(&records, &trajectory(&records)), "duplicate");
}

#[test]
fn trajectory_stamp_mismatch_is_rejected() {
    let records = vec![sample("/front", "front", 10, 20)];
    let mut wrong = records.clone();
    wrong[0].stamp.timestamp = Timestamp::from_nanos(21);
    assert_rejected(lookup(&records, &trajectory(&wrong)), "stamp");
}

#[test]
fn trajectory_clock_mismatch_is_rejected() {
    let records = vec![sample("/front", "front", 10, 20)];
    let mut wrong = records.clone();
    wrong[0].stamp.clock = "another-clock".into();
    assert_rejected(lookup(&records, &trajectory(&wrong)), "stamp");
}

#[test]
fn trajectory_domain_mismatch_is_rejected() {
    let records = vec![sample("/front", "front", 10, 20)];
    let mut wrong = records.clone();
    wrong[0].stamp.domain = ClockDomain::Sensor;
    assert_rejected(lookup(&records, &trajectory(&wrong)), "stamp");
}

#[test]
fn trajectory_quality_mismatch_is_rejected() {
    let records = vec![sample("/front", "front", 10, 20)];
    for quality in [
        spatialrust_sync::SyncQuality { offset_ns: 1, ..Default::default() },
        spatialrust_sync::SyncQuality { uncertainty_ns: 1, ..Default::default() },
        spatialrust_sync::SyncQuality { estimated: true, ..Default::default() },
    ] {
        let mut wrong = records.clone();
        wrong[0].stamp.quality = quality;
        assert_rejected(lookup(&records, &trajectory(&wrong)), "stamp");
    }
}

#[test]
fn mixed_topics_cannot_share_front_pose_lookup() {
    let records = vec![sample("/front", "front", 10, 20), sample("/rear", "front", 30, 40)];
    assert_rejected(lookup(&records, &trajectory(&records)), "topic");
}

#[test]
fn mixed_frames_cannot_share_front_pose_lookup() {
    let records = vec![sample("/front", "front", 10, 20), sample("/front", "another", 30, 40)];
    assert_rejected(lookup(&records, &trajectory(&records)), "frame");
}

#[test]
fn matched_front_keeps_its_exact_distinct_pose() {
    let front = vec![sample("/front", "front", 10, 20), sample("/front", "front", 30, 40)];
    let poses = lookup(&front, &trajectory(&front)).unwrap();
    let mut records = front;
    records.push(sample("/rear", "rear", 31, 41));
    let episode = MemoryEpisode::from_records(records);
    let front_id = TopicId::new("/front");
    let rear_id = TopicId::new("/rear");
    let bundle = DeterministicReplayer::new(&episode)
        .next_bundle(
            &front_id,
            &[rear_id.clone()],
            SyncWindow { max_delta_ns: 1, max_uncertainty_ns: 0 },
        )
        .unwrap()
        .unwrap();
    let stamp = bundle[&front_id].stamp.as_nanos();
    assert_eq!(stamp, 40);
    assert_eq!(bundle[&rear_id].stamp.as_nanos(), 41);
    assert_eq!(poses[&20].translation(), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(poses[&stamp].translation(), Vec3::new(3.0, 0.0, 0.0));
}

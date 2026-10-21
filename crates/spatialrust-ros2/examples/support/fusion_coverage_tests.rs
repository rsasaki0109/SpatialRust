//! Synthetic traceability controls; no measured calibration or accuracy claim.

#[path = "fusion_readback_tests.rs"]
mod readback_tests;

use super::super::coverage_characterization::{episode, sample};
use super::*;

fn context() -> Context {
    Context {
        identity: InputIdentity {
            source_sha256: "a".repeat(64),
            source_bytes: 100,
            calibration_sha256: "b".repeat(64),
            calibration_bytes: 200,
        },
        front_topic: "/front".into(),
        rear_topic: "/rear".into(),
        front_frame: "front".into(),
        rear_frame: "rear".into(),
        root_frame: "root".into(),
        clock_id: "fixture-time".into(),
        offset_ns: 0.0,
        drift_ppm: 0.0,
        uncertainty_ns: 0,
        anchor_nanos: 10,
        chunk_points: 2,
        max_delta_ns: 100,
    }
}

fn source(episode: &MemoryEpisode) -> MappingSourceSummary {
    let front = episode.records().iter().filter(|r| r.topic.as_str() == "/front").count() as u64;
    let rear = episode.records().len() as u64 - front;
    MappingSourceSummary {
        front_topic: "/front".into(),
        rear_topic: "/rear".into(),
        front_bag_message_count: front,
        rear_bag_message_count: rear,
        front_chunk_count: front * 2,
        rear_chunk_count: rear * 2,
        front_record_count: front,
        rear_record_count: rear,
        total_record_count: front + rear,
        total_point_count: (front + rear) * 4,
        retained_bytes: 1,
        peak_source_bytes: 1,
        start_nanos: episode.records().first().map(|r| r.stamp.as_nanos()),
        end_nanos: episode.records().last().map(|r| r.stamp.as_nanos()),
        full_bag_processed: true,
        truncated: false,
    }
}

fn visit_all(coverage: &mut Coverage) {
    for index in 0..coverage.records.len() {
        if coverage.records[index].sensor == Sensor::Front {
            coverage.note_front(index, 4).unwrap();
        }
    }
    for index in 0..coverage.associations.len() {
        coverage.note_rear(index, 4).unwrap();
    }
}

fn fixture() -> (MemoryEpisode, Context, MappingSourceSummary, CompletedCoverage) {
    let episode = episode(&[("/front", 10), ("/front", 20), ("/front", 30), ("/rear", 20)]);
    let context = context();
    let source = source(&episode);
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    visit_all(&mut coverage);
    let completed = coverage.finish(&episode, &context, &source, 6, 24).unwrap();
    (episode, context, source, completed)
}

fn validate(
    coverage: &Coverage,
    episode: &MemoryEpisode,
    context: &Context,
    source: &MappingSourceSummary,
) -> Result<Summary> {
    if coverage.summary.is_none() {
        return Err("completed coverage requires summary".into());
    }
    coverage.validated_summary(episode, context, source, 6, 24)
}

#[test]
fn reused_rear_distinguishes_six_calls_from_four_records() {
    let (_, _, _, completed) = fixture();
    let summary = completed.0.summary.unwrap();
    assert_eq!(
        (summary.total_visits, summary.unique_visited_records, summary.reused_rear_visits),
        (6, 4, 2)
    );
    assert_eq!((summary.total_point_visits, summary.matched_rear_records), (24, 1));
    assert_eq!(
        completed
            .0
            .associations
            .iter()
            .map(|a| (a.front_index, a.rear_index, a.delta_ns))
            .collect::<Vec<_>>(),
        vec![(0, 2, 10), (1, 2, 0), (3, 2, 10)]
    );
}

#[test]
fn unmatched_front_is_integrated_and_unmatched_rear_is_unvisited() {
    let episode = episode(&[("/front", 10), ("/front", 20), ("/rear", 11), ("/rear", 100)]);
    let mut context = context();
    context.max_delta_ns = 2;
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    visit_all(&mut coverage);
    let completed = coverage.finish(&episode, &context, &source(&episode), 3, 12).unwrap();
    let summary = completed.0.summary.unwrap();
    assert_eq!(
        (
            summary.unmatched_front_records,
            summary.unmatched_rear_records,
            summary.unique_visited_records,
            summary.unvisited_records
        ),
        (1, 1, 3, 1)
    );
    assert_eq!(completed.0.records[3].visits, 0);
}

#[test]
fn round_trip_is_validated_against_independent_expected_context() {
    let (episode, context, source, completed) = fixture();
    let decoded: Coverage =
        serde_json::from_slice(&serde_json::to_vec(&completed.0).unwrap()).unwrap();
    assert_eq!(
        validate(&decoded, &episode, &context, &source).unwrap(),
        completed.0.summary.unwrap()
    );
}

#[test]
fn forged_context_fields_are_rejected() {
    let (episode, context, source, completed) = fixture();
    for (field, value) in [
        ("front_topic", serde_json::json!("/other")),
        ("rear_topic", serde_json::json!("/other")),
        ("front_frame", serde_json::json!("other")),
        ("rear_frame", serde_json::json!("other")),
        ("root_frame", serde_json::json!("other")),
        ("clock_id", serde_json::json!("other")),
        ("offset_ns", serde_json::json!(1.0)),
        ("drift_ppm", serde_json::json!(1.0)),
        ("uncertainty_ns", serde_json::json!(1)),
        ("anchor_nanos", serde_json::json!(11)),
        ("chunk_points", serde_json::json!(3)),
        ("max_delta_ns", serde_json::json!(101)),
    ] {
        let mut json = serde_json::to_value(&completed.0).unwrap();
        json["context"][field] = value;
        let decoded: Coverage = serde_json::from_value(json).unwrap();
        assert!(validate(&decoded, &episode, &context, &source).is_err(), "{field}");
    }
}

#[test]
fn forged_input_and_calibration_identity_are_rejected() {
    let (episode, context, source, completed) = fixture();
    for (field, value) in [
        ("source_sha256", serde_json::json!("c".repeat(64))),
        ("calibration_sha256", serde_json::json!("d".repeat(64))),
        ("source_bytes", serde_json::json!(101)),
        ("calibration_bytes", serde_json::json!(201)),
    ] {
        let mut json = serde_json::to_value(&completed.0).unwrap();
        json["context"]["identity"][field] = value;
        assert!(
            validate(&serde_json::from_value(json).unwrap(), &episode, &context, &source).is_err(),
            "{field}"
        );
    }
}

#[test]
fn forged_record_descriptors_and_visit_counts_are_rejected() {
    let (episode, context, source, completed) = fixture();
    for (field, value) in [
        ("episode_index", serde_json::json!(1)),
        ("sensor", serde_json::json!("rear")),
        ("raw_nanos", serde_json::json!(11)),
        ("corrected_nanos", serde_json::json!(11)),
        ("source_points", serde_json::json!(5)),
        ("visits", serde_json::json!(2)),
        ("point_visits", serde_json::json!(8)),
    ] {
        let mut json = serde_json::to_value(&completed.0).unwrap();
        json["records"][0][field] = value;
        assert!(
            validate(&serde_json::from_value(json).unwrap(), &episode, &context, &source).is_err(),
            "{field}"
        );
    }
}

#[test]
fn forged_association_indices_deltas_and_calls_are_rejected() {
    let (episode, context, source, completed) = fixture();
    for (field, value) in [
        ("front_index", serde_json::json!(u64::MAX)),
        ("rear_index", serde_json::json!(u64::MAX)),
        ("delta_ns", serde_json::json!(11)),
        ("visited_points", serde_json::Value::Null),
        ("visited_points", serde_json::json!(5)),
    ] {
        let mut json = serde_json::to_value(&completed.0).unwrap();
        json["associations"][0][field] = value;
        assert!(
            validate(&serde_json::from_value(json).unwrap(), &episode, &context, &source).is_err(),
            "{field}"
        );
    }
}

#[test]
fn removed_extra_or_reordered_associations_are_rejected() {
    let (episode, context, source, completed) = fixture();
    for mode in 0..3 {
        let mut decoded: Coverage =
            serde_json::from_slice(&serde_json::to_vec(&completed.0).unwrap()).unwrap();
        match mode {
            0 => {
                decoded.associations.pop();
            }
            1 => decoded.associations.push(decoded.associations[0]),
            _ => decoded.associations.swap(0, 1),
        }
        assert!(validate(&decoded, &episode, &context, &source).is_err());
    }
}

#[test]
fn forged_summary_and_schema_are_rejected() {
    let (episode, context, source, completed) = fixture();
    let mut decoded: Coverage =
        serde_json::from_slice(&serde_json::to_vec(&completed.0).unwrap()).unwrap();
    decoded.summary.as_mut().unwrap().unique_visited_records = 6;
    assert!(validate(&decoded, &episode, &context, &source).is_err());
    decoded.schema_version = 2;
    assert!(validate(&decoded, &episode, &context, &source).is_err());
}

#[test]
fn unknown_json_fields_fail_closed() {
    let (_, _, _, completed) = fixture();
    let mut json = serde_json::to_value(&completed.0).unwrap();
    json["physical_coverage"] = serde_json::json!(true);
    assert!(serde_json::from_value::<Coverage>(json).is_err());
}

#[test]
fn omitted_front_call_blocks_completion() {
    let (episode, context, source, _) = fixture();
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    for index in 0..coverage.associations.len() {
        coverage.note_rear(index, 4).unwrap();
    }
    assert!(coverage.finish(&episode, &context, &source, 6, 24).is_err());
}

#[test]
fn omitted_rear_call_blocks_completion() {
    let (episode, context, source, _) = fixture();
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    for index in [0, 1, 3] {
        coverage.note_front(index, 4).unwrap();
    }
    assert!(coverage.finish(&episode, &context, &source, 6, 24).is_err());
}

#[test]
fn double_front_and_rear_calls_are_rejected_without_mutation() {
    let (_, _, _, completed) = fixture();
    let mut coverage = completed.0;
    let before = serde_json::to_vec(&coverage).unwrap();
    assert!(coverage.note_front(0, 4).is_err());
    assert!(coverage.note_rear(0, 4).is_err());
    assert_eq!(serde_json::to_vec(&coverage).unwrap(), before);
}

#[test]
fn wrong_points_roles_and_indices_do_not_record_calls() {
    let (episode, context, _, _) = fixture();
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    let before = serde_json::to_vec(&coverage).unwrap();
    assert!(coverage.note_front(0, 3).is_err());
    assert!(coverage.note_rear(0, 3).is_err());
    assert!(coverage.note_front(2, 4).is_err());
    assert!(coverage.note_front(usize::MAX, 4).is_err());
    assert!(coverage.note_rear(usize::MAX, 4).is_err());
    assert_eq!(serde_json::to_vec(&coverage).unwrap(), before);
}

#[test]
fn checked_counter_overflow_is_transactional() {
    let (episode, context, _, _) = fixture();
    let mut coverage = Coverage::prepare(&episode, &context).unwrap();
    for (visits, points) in [(u64::MAX, 0), (1, u64::MAX)] {
        coverage.records[2].visits = visits;
        coverage.records[2].point_visits = points;
        let before = serde_json::to_vec(&coverage).unwrap();
        assert!(coverage.note_rear(0, 4).is_err());
        assert_eq!(serde_json::to_vec(&coverage).unwrap(), before);
    }
}

#[test]
fn existing_tsdf_and_source_totals_must_reconcile() {
    let (episode, context, mut source, completed) = fixture();
    assert!(completed.0.validated_summary(&episode, &context, &source, 5, 24).is_err());
    assert!(completed.0.validated_summary(&episode, &context, &source, 6, 23).is_err());
    source.total_point_count += 1;
    assert!(validate(&completed.0, &episode, &context, &source).is_err());
    source.total_point_count -= 1;
    source.truncated = true;
    assert!(validate(&completed.0, &episode, &context, &source).is_err());
}

#[test]
fn stable_index_accepts_members_and_rejects_foreign_clones() {
    let episode = episode(&[("/rear", 20), ("/front", 20)]);
    for (index, record) in episode.records().iter().enumerate() {
        assert_eq!(record_index(&episode, record).unwrap(), index);
    }
    let foreign = sample("/front", 20);
    assert!(record_index(&episode, &foreign).is_err());
    assert!(record_index(&episode, &sample("/front", 21)).is_err());
}

#[test]
fn prepare_rejects_empty_missing_unmatched_and_duplicate_times() {
    for items in [
        vec![],
        vec![("/front", 10)],
        vec![("/rear", 10)],
        vec![("/front", 10), ("/rear", 1000)],
        vec![("/front", 10), ("/front", 10), ("/rear", 10)],
    ] {
        assert!(Coverage::prepare(&episode(&items), &context()).is_err());
    }
}

#[test]
fn record_clock_frame_quality_and_correction_are_bound() {
    for mode in 0..6 {
        let mut front = sample("/front", 10);
        match mode {
            0 => front.stamp.clock = "other".into(),
            1 => front.stamp.domain = ClockDomain::Sensor,
            2 => front.stamp.quality.estimated = false,
            3 => front.stamp.quality.uncertainty_ns = 1,
            4 => front.stamp.quality.offset_ns = 1,
            _ => front.stamp.timestamp = spatialrust_core::Timestamp::from_nanos(11),
        }
        assert!(Coverage::prepare(
            &MemoryEpisode::from_records(vec![front, sample("/rear", 10)]),
            &context()
        )
        .is_err());
    }
    let mut context = context();
    context.front_frame = "other".into();
    assert!(Coverage::prepare(&episode(&[("/front", 10), ("/rear", 10)]), &context).is_err());
}

#[test]
fn metadata_and_record_capacity_limits_are_inclusive() {
    let episode = episode(&[("/front", 10), ("/rear", 10)]);
    let mut limits = Limits {
        records: 2,
        metadata_bytes: metadata_bytes(2, 1, 2).unwrap(),
        ..Limits::default()
    };
    assert!(Coverage::with_limits(&episode, &context(), limits).is_ok());
    limits.metadata_bytes -= 1;
    assert!(Coverage::with_limits(&episode, &context(), limits).is_err());
    limits = Limits { records: 1, ..Limits::default() };
    assert!(Coverage::with_limits(&episode, &context(), limits).is_err());
    assert!(check_record_bound(MAX_RECORDS).is_ok());
    assert!(check_record_bound(MAX_RECORDS + 1).is_err());
    assert!(metadata_bytes(usize::MAX, 0, 0).is_err());
}

#[test]
fn limits_cannot_exceed_fixed_contract() {
    for limits in [
        Limits { records: MAX_RECORDS + 1, ..Limits::default() },
        Limits { metadata_bytes: MAX_METADATA_BYTES + 1, ..Limits::default() },
        Limits { json_bytes: MAX_JSON_BYTES + 1, ..Limits::default() },
        Limits { records: 0, ..Limits::default() },
    ] {
        assert!(limits.validate().is_err());
    }
}

#[test]
fn labels_and_header_capacities_are_bounded_before_use() {
    let mut context = context();
    context.front_topic = "x".repeat(MAX_LABEL_BYTES + 1);
    assert!(context.validate().is_err());
    context.front_topic = "x".repeat(MAX_LABEL_BYTES);
    assert!(context.validate().is_ok());
    context.front_topic.reserve(MAX_HEADER_BYTES * 2);
    assert!(context.validate().is_err());
}

#[test]
fn invalid_numeric_context_is_rejected() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut context = context();
        context.offset_ns = value;
        assert!(context.validate().is_err());
        context.offset_ns = 0.0;
        context.drift_ppm = value;
        assert!(context.validate().is_err());
    }
    let mut context = context();
    context.chunk_points = 0;
    assert!(context.validate().is_err());
}

#[test]
fn publish_streams_round_trip_and_leaves_no_temporary_file() {
    let (episode, context, source, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    completed.publish(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert!(validate(&serde_json::from_slice(&bytes).unwrap(), &episode, &context, &source).is_ok());
    assert!(!temp_path(&path).exists());
}

#[test]
fn publish_never_replaces_existing_destination() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    fs::write(&path, b"original").unwrap();
    assert!(completed.publish(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"original");
    assert!(!temp_path(&path).exists());
}

#[test]
fn preexisting_temporary_file_is_never_removed() {
    let (_, _, _, completed) = fixture();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    let temp = temp_path(&path);
    fs::write(&temp, b"owned by another attempt").unwrap();
    assert!(completed.publish(&path).is_err());
    assert_eq!(fs::read(&temp).unwrap(), b"owned by another attempt");
    assert!(!path.exists());
}

#[test]
fn newline_is_included_in_exact_output_byte_cap() {
    let (_, _, _, mut completed) = fixture();
    // The decimal limit is part of JSON, so settle its length first.
    let mut limit = 1_u64;
    loop {
        completed.0.limits.json_bytes = limit;
        let next = serde_json::to_vec(&completed.0).unwrap().len() as u64 + 1;
        if next == limit {
            break;
        }
        limit = next;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("coverage.json");
    completed.publish(&path).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), limit);
    completed.0.limits.json_bytes -= 1;
    let failed = directory.path().join("failed.json");
    assert!(completed.publish(&failed).is_err());
    assert!(!failed.exists());
    assert!(!temp_path(&failed).exists());
}

#[test]
fn streaming_limit_checks_arithmetic_before_writing() {
    let mut writer = LimitWriter { inner: Vec::new(), bytes: u64::MAX, limit: u64::MAX };
    assert!(writer.write_all(b"x").is_err());
    assert!(writer.inner.is_empty());
}

fn synthetic_run(
    front: &[u32],
    rear: &[u32],
    delta: u64,
) -> (tempfile::TempDir, super::super::Config) {
    use rusqlite::{params, Connection};
    use spatialrust_runtime::{encode_point_cloud2_xyz, PointCloud2Xyz};
    use spatialrust_viewer::{
        CalibrationArtifact, CalibrationEvidenceClock, CalibrationEvidenceFrame,
        CalibrationEvidenceState, ClockCalibration, FrameTransform, StudioSource,
    };
    use std::collections::BTreeMap;
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("synthetic.db3");
    let db = Connection::open(&input).unwrap();
    db.execute_batch("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB); INSERT INTO topics VALUES(1,'/front','sensor_msgs/msg/PointCloud2','cdr'); INSERT INTO topics VALUES(2,'/rear','sensor_msgs/msg/PointCloud2','cdr');").unwrap();
    let mut id = 0_i64;
    for (topic, frame, stamps) in [(1, "front", front), (2, "rear", rear)] {
        for &stamp in stamps {
            id += 1;
            let message = PointCloud2Xyz::try_new(
                frame,
                0,
                stamp,
                vec![0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 2.],
            )
            .unwrap();
            db.execute(
                "INSERT INTO messages VALUES(?1,?2,?3,?4)",
                params![id, topic, i64::from(stamp), encode_point_cloud2_xyz(&message).unwrap()],
            )
            .unwrap();
        }
    }
    drop(db);
    let receipt = FileReceipt::from_path(ReceiptRole::Input, &input).unwrap();
    let sha = receipt.sha256.unwrap();
    let artifact = |name: &str| {
        let path = directory.path().join(format!("{name}.json"));
        fs::write(&path, b"{\"synthetic_only\":true}\n").unwrap();
        let receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, &path).unwrap();
        CalibrationArtifact::try_new(
            name,
            "registered",
            Some(path.display().to_string()),
            receipt.sha256,
            true,
        )
        .unwrap()
    };
    let clock = CalibrationEvidenceClock::try_new(
        "synthetic-sensor",
        "fixture-time",
        "synthetic identity model; no measured evidence",
        ClockCalibration::try_new(
            "registered",
            "synthetic only",
            2,
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            true,
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let frame = CalibrationEvidenceFrame::try_new(
        "synthetic identity graph; no measured evidence",
        "root",
        BTreeMap::from([("front".into(), "front".into()), ("rear".into(), "rear".into())]),
        vec!["root".into(), "front".into(), "rear".into()],
        ["front", "rear"]
            .into_iter()
            .map(|sensor| {
                FrameTransform::try_new("root", sensor, [0.; 3], [0., 0., 0., 1.], None, true, true)
                    .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let state = CalibrationEvidenceState::try_new(
        "synthetic traceability fixture",
        StudioSource::try_new("synthetic", input.display().to_string(), &sha, &sha, true).unwrap(),
        artifact("clock_evidence"),
        artifact("frame_evidence"),
        clock,
        frame,
        Vec::new(),
    )
    .unwrap();
    assert!(state.registration_ready);
    let calibration = directory.path().join("calibration.json");
    fs::write(&calibration, serde_json::to_vec(&state).unwrap()).unwrap();
    let config = super::super::Config {
        input,
        calibration_evidence: Some(calibration),
        output_dir: directory.path().join("output"),
        expected_sha256: sha,
        front_topic: "/front".into(),
        rear_topic: "/rear".into(),
        root_frame: "root".into(),
        max_delta_ns: delta,
        chunk_points: 2,
        source_memory_bytes: 1_048_576,
        max_records: 100,
        max_points: 1000,
        max_bytes: 1_048_576,
        min_output_free_bytes: 1,
    };
    (directory, config)
}

#[test]
fn synthetic_sqlite_run_links_verified_coverage_and_preserves_state_schema() {
    use spatialrust_io::DatasetManifest;
    use spatialrust_viewer::FullBagMappingState;
    let (_directory, config) = synthetic_run(&[10, 20], &[11, 100], 2);
    let output = config.output_dir.clone();
    let source_receipt = FileReceipt::from_path(ReceiptRole::Input, &config.input).unwrap();
    let calibration_receipt = FileReceipt::from_path(
        ReceiptRole::Auxiliary,
        config.calibration_evidence.as_ref().unwrap(),
    )
    .unwrap();
    super::super::run_config(config).unwrap();
    let coverage_path = output.join(super::super::COVERAGE_FILE);
    let coverage: Coverage = serde_json::from_slice(&fs::read(&coverage_path).unwrap()).unwrap();
    assert_eq!(
        coverage.context.identity,
        InputIdentity::from_receipts(&source_receipt, &calibration_receipt).unwrap()
    );
    let summary = coverage.summary.unwrap();
    assert_eq!(
        (
            summary.total_visits,
            summary.unique_visited_records,
            summary.unvisited_records,
            summary.total_point_visits
        ),
        (3, 3, 1, 12)
    );
    let state: FullBagMappingState =
        serde_json::from_slice(&fs::read(output.join(super::super::STATE_FILE)).unwrap()).unwrap();
    state.validate().unwrap();
    assert!(state.summary.mapping_admitted);
    assert_eq!(state.version, 1);
    assert_eq!(state.tsdf.as_ref().unwrap().integrated_record_count, summary.total_visits);
    assert_eq!(state.tsdf.as_ref().unwrap().integrated_point_count, summary.total_point_visits);
    let artifact = state.artifacts.iter().find(|a| a.role == "fusion-coverage").unwrap();
    let receipt = FileReceipt::from_path(ReceiptRole::Output, &coverage_path).unwrap();
    assert_eq!(artifact.sha256, receipt.sha256.unwrap());
    assert_eq!(artifact.size_bytes, receipt.size_bytes.unwrap());
    let manifest: DatasetManifest =
        serde_json::from_slice(&fs::read(output.join(super::super::MANIFEST_FILE)).unwrap())
            .unwrap();
    assert!(manifest.entries.iter().any(|entry| entry.path == coverage_path));
    manifest.validate_local_files().unwrap();
}

#[test]
fn synthetic_sqlite_reuse_records_actual_successful_tsdf_calls() {
    let (_directory, config) = synthetic_run(&[10, 20, 30], &[20], 100);
    let output = config.output_dir.clone();
    super::super::run_config(config).unwrap();
    let coverage: Coverage =
        serde_json::from_slice(&fs::read(output.join(super::super::COVERAGE_FILE)).unwrap())
            .unwrap();
    let summary = coverage.summary.unwrap();
    assert_eq!(
        (
            summary.total_visits,
            summary.unique_visited_records,
            summary.reused_rear_visits,
            summary.total_point_visits
        ),
        (6, 4, 2, 24)
    );
}

#[test]
fn blocked_synthetic_input_hash_omits_coverage_and_keeps_diagnostics() {
    let (_directory, mut config) = synthetic_run(&[10, 20], &[11, 100], 2);
    config.expected_sha256 = "0".repeat(64);
    let output = config.output_dir.clone();
    assert!(super::super::run_config(config).is_err());
    assert!(!output.join(super::super::COVERAGE_FILE).exists());
    assert!(output.join(super::super::STATE_FILE).is_file());
    assert!(output.join(super::super::MANIFEST_FILE).is_file());
}

#[test]
fn missing_calibration_omits_coverage_and_keeps_diagnostics() {
    let (_directory, mut config) = synthetic_run(&[10, 20], &[11, 100], 2);
    config.calibration_evidence = None;
    let output = config.output_dir.clone();
    assert!(super::super::run_config(config).is_err());
    assert!(!output.join(super::super::COVERAGE_FILE).exists());
    assert!(output.join(super::super::STATE_FILE).is_file());
}

#[test]
fn identity_receipts_require_roles_sizes_and_lowercase_checksums() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input");
    fs::write(&path, b"synthetic").unwrap();
    let source = FileReceipt::from_path(ReceiptRole::Input, &path).unwrap();
    let calibration = FileReceipt::from_path(ReceiptRole::Auxiliary, &path).unwrap();
    assert!(InputIdentity::from_receipts(&source, &calibration).is_ok());
    let mut invalid = source.clone();
    invalid.role = ReceiptRole::Output;
    assert!(InputIdentity::from_receipts(&invalid, &calibration).is_err());
    invalid = source.clone();
    invalid.size_bytes = Some(0);
    assert!(InputIdentity::from_receipts(&invalid, &calibration).is_err());
    invalid = source.clone();
    invalid.sha256 = Some("A".repeat(64));
    assert!(InputIdentity::from_receipts(&invalid, &calibration).is_err());
    invalid = source;
    invalid.sha256 = None;
    assert!(InputIdentity::from_receipts(&invalid, &calibration).is_err());
}

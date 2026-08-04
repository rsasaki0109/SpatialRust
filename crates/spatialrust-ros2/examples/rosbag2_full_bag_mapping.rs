//! Run a source-bound, bounded full-bag odometry and TSDF mapping gate.
//!
//! The runner consumes every selected PointCloud2 record within explicit
//! episode limits.  It applies only a previously registered clock model and
//! root-to-sensor frame paths from `CalibrationEvidenceState`; it never solves
//! or invents calibration.  Missing or mismatched evidence still produces a
//! JSON/HTML/manifest receipt and exits with status 2.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{de::DeserializeOwned, Serialize};
use spatialrust_core::Timestamp;
use spatialrust_interchange::export_triangle_mesh_gltf_json;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_mapping::{IcpScanMatcher, ScanOdometry, ScanOdometryConfig};
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, MemoryChunkSink,
    SpatialRecordSink, StreamOptions,
};
use spatialrust_registration::IcpConfig;
use spatialrust_ros2::Rosbag2PointCloudSource;
use spatialrust_scene::TsdfVolume;
use spatialrust_sync::{
    ClockDomain, ClockId, DeterministicReplayer, EpisodeLimits, MemoryEpisodeBuilder,
    StampedRecord, StampedTime, SyncQuality, SyncWindow, TopicId,
};
use spatialrust_viewer::{
    CalibrationEvidenceFrame, CalibrationEvidenceState, FrameTransform, FullBagMappingState,
    MappingOdometrySummary, MappingSourceSummary, MappingTsdfSummary, ReplayArtifact, StudioSource,
};

const CHUNK_POINTS: usize = 1_000_000;
const SOURCE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const EPISODE_MAX_RECORDS: u64 = 1_000_000;
const EPISODE_MAX_POINTS: u64 = 16_777_216;
const EPISODE_MAX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DELTA_NS: u64 = 100_000_000;
const TSDF_ORIGIN: Vec3<f32> = Vec3::new(-24.0, -24.0, -12.0);
const TSDF_VOXEL_SIZE: f32 = 0.5;
const TSDF_DIMS: [usize; 3] = [96, 96, 48];
const TSDF_TRUNCATION: f32 = 1.0;
const STATE_FILE: &str = "full-bag-mapping.json";
const HTML_FILE: &str = "full-bag-mapping.html";
const MANIFEST_FILE: &str = "full-bag-mapping.manifest.json";
const MESH_FILE: &str = "full-bag-mapping.mesh.gltf";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    calibration_evidence: Option<PathBuf>,
    output_dir: PathBuf,
    expected_sha256: String,
    front_topic: String,
    rear_topic: String,
    root_frame: String,
    max_delta_ns: u64,
    chunk_points: usize,
    source_memory_bytes: u64,
    max_records: u64,
    max_points: u64,
    max_bytes: u64,
    min_output_free_bytes: u64,
}

#[derive(Debug)]
struct CollectedTopic {
    records: Vec<StampedRecord>,
    bag_message_count: u64,
    chunk_count: u64,
    peak_source_bytes: u64,
    frame_ids: BTreeSet<String>,
}

#[derive(Debug)]
struct ClockCorrection {
    clock_id: String,
    uncertainty_ns: u64,
    offset_ns: f64,
    drift_ppm: f64,
    anchor_nanos: Option<u64>,
}

#[derive(Debug)]
struct MappingExecution {
    source_summary: MappingSourceSummary,
    odometry: MappingOdometrySummary,
    tsdf: MappingTsdfSummary,
    mesh_json: String,
    clock_applied: bool,
    frame_graph_applied: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-full-bag-mapping: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    validate_config(&config)?;
    ensure_outputs_absent(&config)?;

    let output_parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    let preflight = StoragePreflight::check(output_parent, config.min_output_free_bytes)?;
    let input_receipt = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let input_size = input_receipt.size_bytes.ok_or("input size was not produced")?;
    let observed_sha256 = input_receipt.sha256.clone().ok_or("input checksum was not produced")?;
    let input_path = config.input.display().to_string();
    let source_identity_matches = observed_sha256 == config.expected_sha256;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        input_path.clone(),
        config.expected_sha256.clone(),
        observed_sha256.clone(),
        source_identity_matches,
    )?;

    let mut blockers = Vec::new();
    if !source_identity_matches {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            ),
        );
    }
    let (calibration, calibration_receipt) = load_calibration(
        config.calibration_evidence.as_deref(),
        &input_path,
        input_size,
        &observed_sha256,
        &mut blockers,
    )?;

    let calibration_ready = calibration.as_ref().is_some_and(|calibration| {
        calibration.registration_ready
            && calibration.source.identity_matches
            && calibration.source.path == input_path
            && calibration.source.observed_sha256 == observed_sha256
    });
    let mut execution = None;
    let started = Instant::now();
    if source_identity_matches && calibration_ready {
        match execute_mapping(&config, calibration.as_ref().expect("calibration is ready")) {
            Ok(result) => execution = Some(result),
            Err(error) => {
                push_blocker(&mut blockers, format!("mapping execution blocked: {error}"))
            }
        }
    }
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    fs::create_dir_all(&config.output_dir)?;
    let mut artifacts = vec![replay_artifact("canonical-input", &input_receipt)?];
    if let Some(receipt) = &calibration_receipt {
        artifacts.push(replay_artifact("calibration-evidence", receipt)?);
    }
    let mesh_path = config.output_dir.join(MESH_FILE);
    if let Some(result) = &execution {
        fs::write(&mesh_path, format!("{}\n", result.mesh_json))?;
        let mesh_receipt = FileReceipt::from_path(ReceiptRole::Output, &mesh_path)?;
        artifacts.push(replay_artifact("tsdf-mesh", &mesh_receipt)?);
    }

    let zero_summary = || MappingSourceSummary {
        front_topic: config.front_topic.clone(),
        rear_topic: config.rear_topic.clone(),
        front_bag_message_count: 0,
        rear_bag_message_count: 0,
        front_chunk_count: 0,
        rear_chunk_count: 0,
        front_record_count: 0,
        rear_record_count: 0,
        total_record_count: 0,
        total_point_count: 0,
        retained_bytes: 0,
        peak_source_bytes: 0,
        start_nanos: None,
        end_nanos: None,
        full_bag_processed: false,
        truncated: false,
    };
    let (source_summary, odometry, tsdf, clock_applied, frame_graph_applied) =
        if let Some(result) = execution {
            (
                result.source_summary,
                Some(result.odometry),
                Some(result.tsdf),
                result.clock_applied,
                result.frame_graph_applied,
            )
        } else {
            (zero_summary(), None, None, false, false)
        };

    let state = FullBagMappingState::try_new(
        format!("Full-Bag Mapping Gate — {}", file_label(&config.input)),
        source,
        calibration,
        source_summary,
        odometry,
        tsdf,
        artifacts,
        clock_applied,
        frame_graph_applied,
        blockers,
    )?;
    state.validate()?;

    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state, elapsed_ns)?)?;

    let mut manifest = DatasetManifest::new();
    manifest.entries.push(input_receipt);
    if let Some(path) = config.calibration_evidence.as_deref() {
        if path.is_file() {
            manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, path)?);
        }
    }
    if mesh_path.is_file() {
        manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &mesh_path)?);
    }
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Full-bag mapping receipt: {} (mapping_admitted={}, elapsed_ns={})",
        state_path.display(),
        state.summary.mapping_admitted,
        elapsed_ns
    );
    println!("Full-bag mapping dashboard: {}", html_path.display());
    println!(
        "Full-bag mapping manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.summary.mapping_admitted {
        return Err("full-bag mapping admission failed; see the external receipt blockers".into());
    }
    Ok(())
}

fn execute_mapping(
    config: &Config,
    calibration: &CalibrationEvidenceState,
) -> Result<MappingExecution, Box<dyn Error>> {
    let mut clock = ClockCorrection::try_new(calibration)?;
    let options =
        StreamOptions::new(config.chunk_points, MemoryBudget::new(config.source_memory_bytes)?)?;
    let front = collect_topic(&config.input, &config.front_topic, options.clone(), &mut clock)?;
    let rear = collect_topic(&config.input, &config.rear_topic, options, &mut clock)?;
    if front.records.len() < 2 {
        return Err("at least two complete front records are required for full-bag odometry".into());
    }
    let front_frame_ids = front.frame_ids.clone();
    let rear_frame_ids = rear.frame_ids.clone();
    if front_frame_ids.len() != 1 || rear_frame_ids.len() != 1 {
        return Err("each selected sensor must retain exactly one source frame".into());
    }
    let front_frame = front_frame_ids.iter().next().expect("front frame exists");
    let rear_frame = rear_frame_ids.iter().next().expect("rear frame exists");
    if calibration.frame.root_frame != config.root_frame {
        return Err(format!(
            "registered calibration root `{}` differs from requested root `{}`",
            calibration.frame.root_frame, config.root_frame
        )
        .into());
    }
    let expected_front = calibration
        .frame
        .required_frames
        .get("front")
        .ok_or("calibration evidence has no required front frame")?;
    let expected_rear = calibration
        .frame
        .required_frames
        .get("rear")
        .ok_or("calibration evidence has no required rear frame")?;
    if front_frame != expected_front || rear_frame != expected_rear {
        return Err(format!(
            "source frames differ from registered calibration: front `{front_frame}`/`{expected_front}`, rear `{rear_frame}`/`{expected_rear}"
        )
        .into());
    }

    let front_record_count = u64::try_from(front.records.len())?;
    let rear_record_count = u64::try_from(rear.records.len())?;
    let front_bag_message_count = front.bag_message_count;
    let rear_bag_message_count = rear.bag_message_count;
    let front_chunk_count = front.chunk_count;
    let rear_chunk_count = rear.chunk_count;
    let peak_source_bytes = front.peak_source_bytes.max(rear.peak_source_bytes);
    let front_records = front.records;
    let rear_records = rear.records;
    let mut builder = MemoryEpisodeBuilder::try_new(EpisodeLimits::new(
        config.max_records,
        config.max_points,
        config.max_bytes,
    ))?;
    for record in front_records {
        builder.push(record)?;
    }
    for record in rear_records {
        builder.push(record)?;
    }
    let total_record_count = u64::try_from(builder.len())?;
    let total_point_count = builder.points();
    let retained_bytes = builder.bytes();
    let episode = builder.finish();
    let start_nanos = episode.records().first().map(|record| record.stamp.as_nanos());
    let end_nanos = episode.records().last().map(|record| record.stamp.as_nanos());
    let source_summary = MappingSourceSummary {
        front_topic: config.front_topic.clone(),
        rear_topic: config.rear_topic.clone(),
        front_bag_message_count,
        rear_bag_message_count,
        front_chunk_count,
        rear_chunk_count,
        front_record_count,
        rear_record_count,
        total_record_count,
        total_point_count,
        retained_bytes,
        peak_source_bytes,
        start_nanos,
        end_nanos,
        full_bag_processed: true,
        truncated: false,
    };

    let front_id = TopicId::new(config.front_topic.clone());
    let front_records: Vec<&StampedRecord> =
        episode.records().iter().filter(|record| record.topic == front_id).collect();
    let rear_id = TopicId::new(config.rear_topic.clone());
    let odometry = ScanOdometry::try_new(ScanOdometryConfig::new(front_records.len(), 3))?;
    let matcher = IcpScanMatcher::new(IcpConfig {
        max_iterations: 10,
        max_correspondence_distance: 2.0,
        ..IcpConfig::default()
    });
    let odometry_result = odometry.estimate(&episode, &front_id, &matcher)?;
    if odometry_result.truncated {
        return Err("full-bag odometry was truncated by the scan bound".into());
    }
    let root_t_front =
        resolve_frame_transform(&calibration.frame, &calibration.frame.root_frame, front_frame)?;
    let root_t_rear =
        resolve_frame_transform(&calibration.frame, &calibration.frame.root_frame, rear_frame)?;

    let mut replayer = DeterministicReplayer::new(&episode);
    let mut matched_rear = Vec::new();
    let window =
        SyncWindow { max_delta_ns: config.max_delta_ns, max_uncertainty_ns: clock.uncertainty_ns };
    while let Some(bundle) =
        replayer.next_bundle(&front_id, std::slice::from_ref(&rear_id), window)?
    {
        let front_record = *bundle.get(&front_id).ok_or("front sync bundle member missing")?;
        let rear_record = *bundle.get(&rear_id).ok_or("rear sync bundle member missing")?;
        matched_rear.push((front_record.stamp.as_nanos(), rear_record));
    }
    if matched_rear.is_empty() {
        return Err("no front/rear records matched within the configured sync window".into());
    }

    let mut pose_by_stamp = BTreeMap::new();
    for (record, pose) in front_records.iter().zip(odometry_result.trajectory.samples()) {
        pose_by_stamp.insert(record.stamp.as_nanos(), pose.pose.isometry);
    }
    let mut volume = TsdfVolume::try_new(TSDF_ORIGIN, TSDF_VOXEL_SIZE, TSDF_DIMS, TSDF_TRUNCATION)?;
    let mut integrated_records = 0_u64;
    let mut integrated_points = 0_u64;
    for (record, pose) in front_records.iter().zip(odometry_result.trajectory.samples()) {
        let volume_t_sensor = root_t_front.compose(pose.pose.isometry);
        let points = volume.integrate_cloud_with_pose(
            record.record.cloud(),
            volume_t_sensor,
            Vec3::new(0.0, 0.0, 0.0),
        )?;
        integrated_records = integrated_records.checked_add(1).ok_or("TSDF record overflow")?;
        integrated_points =
            integrated_points.checked_add(u64::try_from(points)?).ok_or("TSDF point overflow")?;
    }
    for (stamp, record) in matched_rear {
        let pose =
            *pose_by_stamp.get(&stamp).ok_or("matched rear record has no front trajectory pose")?;
        let volume_t_sensor = root_t_rear.compose(pose);
        let points = volume.integrate_cloud_with_pose(
            record.record.cloud(),
            volume_t_sensor,
            Vec3::new(0.0, 0.0, 0.0),
        )?;
        integrated_records = integrated_records.checked_add(1).ok_or("TSDF record overflow")?;
        integrated_points =
            integrated_points.checked_add(u64::try_from(points)?).ok_or("TSDF point overflow")?;
    }
    let mesh = volume.extract_mesh(1.0);
    let mesh_json = export_triangle_mesh_gltf_json(&mesh)?;
    let odometry_summary = MappingOdometrySummary {
        topic: odometry_result.topic.as_str().to_owned(),
        source_frame: odometry_result.frame_id.0.clone(),
        root_frame: calibration.frame.root_frame.clone(),
        clock_id: calibration.clock.target_domain.clone(),
        clock_domain: "external-calibrated".into(),
        matcher: "point-to-point ICP; bounded ten-iteration full-bag run".into(),
        scan_count: u64::try_from(odometry_result.trajectory.samples().len())?,
        motion_count: u64::try_from(odometry_result.motions.len())?,
        pose_graph_node_count: u64::try_from(odometry_result.pose_graph.nodes().len())?,
        pose_graph_edge_count: u64::try_from(odometry_result.pose_graph.edges().len())?,
        complete: true,
        truncated: false,
    };
    let tsdf_summary = MappingTsdfSummary {
        frame_id: calibration.frame.root_frame.clone(),
        origin: [TSDF_ORIGIN.x, TSDF_ORIGIN.y, TSDF_ORIGIN.z],
        voxel_size: TSDF_VOXEL_SIZE,
        dims: TSDF_DIMS,
        truncation: TSDF_TRUNCATION,
        integrated_record_count: integrated_records,
        integrated_point_count: integrated_points,
        mesh_vertex_count: u64::try_from(mesh.vertex_count())?,
        mesh_triangle_count: u64::try_from(mesh.triangle_count())?,
        complete: true,
    };
    Ok(MappingExecution {
        source_summary,
        odometry: odometry_summary,
        tsdf: tsdf_summary,
        mesh_json,
        clock_applied: true,
        frame_graph_applied: true,
    })
}

fn collect_topic(
    input: &Path,
    topic: &str,
    options: StreamOptions,
    clock: &mut ClockCorrection,
) -> Result<CollectedTopic, Box<dyn Error>> {
    let mut source =
        Rosbag2PointCloudSource::open(input, topic, options, CancellationToken::default())?;
    let bag_message_count = source.topic().message_count;
    let peak_source_bytes = source.max_message_bytes();
    let mut current: Option<(u64, String, MemoryChunkSink)> = None;
    let mut records = Vec::new();
    let mut chunk_count = 0_u64;
    let mut frame_ids = BTreeSet::new();
    while let Some(chunk) = source.next_chunk() {
        let chunk = chunk?;
        chunk_count = chunk_count.checked_add(1).ok_or("full-bag chunk count overflow")?;
        let record = chunk.record().clone();
        let stamp = record.metadata().timestamp.as_nanos();
        let frame = record.metadata().frame_id.0.clone();
        match current.as_mut() {
            Some((current_stamp, current_frame, sink))
                if *current_stamp == stamp && current_frame == &frame =>
            {
                sink.write_record(&record)?;
            }
            Some(_) => {
                finish_pending(&mut current, topic, clock, &mut records, &mut frame_ids)?;
                let mut sink = MemoryChunkSink::new();
                sink.write_record(&record)?;
                current = Some((stamp, frame, sink));
            }
            None => {
                let mut sink = MemoryChunkSink::new();
                sink.write_record(&record)?;
                current = Some((stamp, frame, sink));
            }
        }
    }
    finish_pending(&mut current, topic, clock, &mut records, &mut frame_ids)?;
    Ok(CollectedTopic { records, bag_message_count, chunk_count, peak_source_bytes, frame_ids })
}

fn finish_pending(
    current: &mut Option<(u64, String, MemoryChunkSink)>,
    topic: &str,
    clock: &mut ClockCorrection,
    records: &mut Vec<StampedRecord>,
    frame_ids: &mut BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let Some((raw_stamp, frame, sink)) = current.take() else {
        return Ok(());
    };
    let record = sink.into_record()?.ok_or("empty PointCloud2 record")?;
    let corrected_stamp = clock.correct(raw_stamp)?;
    let quality = SyncQuality {
        offset_ns: clock.offset_i64()?,
        uncertainty_ns: clock.uncertainty_ns,
        estimated: true,
    };
    let stamp = StampedTime {
        clock: ClockId::new(clock.clock_id.clone()),
        domain: ClockDomain::External,
        timestamp: Timestamp::from_nanos(corrected_stamp),
        quality,
    };
    frame_ids.insert(frame);
    records.push(StampedRecord::new(topic, stamp, record));
    Ok(())
}

impl ClockCorrection {
    fn try_new(calibration: &CalibrationEvidenceState) -> Result<Self, Box<dyn Error>> {
        let clock = &calibration.clock.calibration;
        let uncertainty = clock.uncertainty_nanos.unwrap_or(0.0);
        let offset = clock.median_offset_nanos.unwrap_or(0.0);
        let drift = clock.drift_ppm.unwrap_or(0.0);
        if !uncertainty.is_finite()
            || uncertainty < 0.0
            || !offset.is_finite()
            || !drift.is_finite()
        {
            return Err("registered clock evidence has non-finite correction values".into());
        }
        let uncertainty_ns = u64::try_from(uncertainty.ceil() as i128)
            .map_err(|_| "clock uncertainty does not fit in u64")?;
        Ok(Self {
            clock_id: calibration.clock.target_domain.clone(),
            uncertainty_ns,
            offset_ns: offset,
            drift_ppm: drift,
            anchor_nanos: None,
        })
    }

    fn correct(&mut self, raw: u64) -> Result<u64, Box<dyn Error>> {
        let anchor = *self.anchor_nanos.get_or_insert(raw);
        let delta = raw as f64 - anchor as f64;
        let corrected = raw as f64 + self.offset_ns + delta * self.drift_ppm / 1_000_000.0;
        if !corrected.is_finite() || corrected < 0.0 || corrected > u64::MAX as f64 {
            return Err("clock correction produced an out-of-range timestamp".into());
        }
        Ok(corrected.round() as u64)
    }

    fn offset_i64(&self) -> Result<i64, Box<dyn Error>> {
        let rounded = self.offset_ns.round();
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err("clock offset does not fit in i64".into());
        }
        Ok(rounded as i64)
    }
}

fn resolve_frame_transform(
    evidence: &CalibrationEvidenceFrame,
    root: &str,
    target: &str,
) -> Result<Isometry3<f32>, Box<dyn Error>> {
    if root == target {
        return Ok(Isometry3::identity());
    }
    let mut queue = VecDeque::from([(root.to_owned(), Isometry3::identity())]);
    let mut visited = BTreeSet::from([root.to_owned()]);
    while let Some((frame, root_t_frame)) = queue.pop_front() {
        for edge in evidence
            .edges
            .iter()
            .filter(|edge| edge.accepted && edge.source_bound && edge.parent_frame == frame)
        {
            let next = edge.child_frame.clone();
            if !visited.insert(next.clone()) {
                continue;
            }
            let root_t_child = root_t_frame.compose(edge_isometry(edge)?);
            if next == target {
                return Ok(root_t_child);
            }
            queue.push_back((next, root_t_child));
        }
    }
    Err(format!("no accepted source-bound frame path from `{root}` to `{target}`").into())
}

fn edge_isometry(edge: &FrameTransform) -> Result<Isometry3<f32>, Box<dyn Error>> {
    let translation = edge.translation_m;
    let rotation = edge.rotation_xyzw;
    if translation.iter().any(|value| !value.is_finite())
        || rotation.iter().any(|value| !value.is_finite())
    {
        return Err("frame evidence contains non-finite transform values".into());
    }
    let quaternion =
        Quat::new(rotation[0] as f32, rotation[1] as f32, rotation[2] as f32, rotation[3] as f32)
            .normalize();
    Ok(Isometry3::new(
        quaternion,
        Vec3::new(translation[0] as f32, translation[1] as f32, translation[2] as f32),
    ))
}

fn load_calibration(
    path: Option<&Path>,
    input_path: &str,
    input_size: u64,
    observed_sha256: &str,
    blockers: &mut Vec<String>,
) -> Result<(Option<CalibrationEvidenceState>, Option<FileReceipt>), Box<dyn Error>> {
    let Some(path) = path else {
        push_blocker(blockers, "--calibration-evidence was not supplied");
        return Ok((None, None));
    };
    let receipt = if path.is_file() {
        Some(FileReceipt::from_path(ReceiptRole::Auxiliary, path)?)
    } else {
        push_blocker(
            blockers,
            format!("calibration evidence file '{}' is missing", path.display()),
        );
        None
    };
    let Some(receipt) = receipt else {
        return Ok((None, None));
    };
    let parsed: CalibrationEvidenceState = match read_json(path) {
        Ok(value) => value,
        Err(error) => {
            push_blocker(blockers, format!("calibration evidence JSON is invalid: {error}"));
            return Ok((None, Some(receipt)));
        }
    };
    if let Err(error) = parsed.validate() {
        push_blocker(blockers, format!("calibration evidence state is invalid: {error}"));
        return Ok((None, Some(receipt)));
    }
    if parsed.source.path != input_path
        || parsed.source.observed_sha256 != observed_sha256
        || parsed.source.expected_sha256 != observed_sha256
    {
        push_blocker(blockers, "calibration evidence source identity is not bound to this input");
    }
    if parsed.source.path != input_path || parsed.source.observed_sha256 != observed_sha256 {
        push_blocker(blockers, "calibration evidence path or observed SHA differs from this input");
    }
    if input_size == 0 {
        push_blocker(blockers, "input file has zero size");
    }
    Ok((Some(parsed), Some(receipt)))
}

fn replay_artifact(role: &str, receipt: &FileReceipt) -> Result<ReplayArtifact, Box<dyn Error>> {
    ReplayArtifact::try_new(
        role,
        receipt.path.display().to_string(),
        receipt.size_bytes.ok_or("artifact size is missing")?,
        receipt.sha256.clone().ok_or("artifact checksum is missing")?,
    )
    .map_err(Into::into)
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.input.is_absolute()
        || !config.output_dir.is_absolute()
        || config.calibration_evidence.as_ref().is_some_and(|path| !path.is_absolute())
    {
        return Err("input, calibration evidence, and output paths must be absolute".into());
    }
    if config.front_topic.trim().is_empty()
        || config.rear_topic.trim().is_empty()
        || config.front_topic == config.rear_topic
        || config.root_frame.trim().is_empty()
    {
        return Err("front/rear topics and root frame must be distinct and non-empty".into());
    }
    for (name, value) in [
        ("max delta", config.max_delta_ns),
        ("chunk points", u64::try_from(config.chunk_points).unwrap_or(0)),
        ("source memory", config.source_memory_bytes),
        ("max records", config.max_records),
        ("max points", config.max_points),
        ("max bytes", config.max_bytes),
        ("minimum output free bytes", config.min_output_free_bytes),
    ] {
        if value == 0 {
            return Err(format!("{name} must be greater than zero").into());
        }
    }
    validate_sha256(&config.expected_sha256)
}

fn ensure_outputs_absent(config: &Config) -> Result<(), Box<dyn Error>> {
    if config.output_dir.exists() {
        return Err(format!(
            "output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut calibration_evidence = None;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut front_topic = "/lidar_front/points_raw".to_owned();
    let mut rear_topic = "/lidar_rear/points_raw".to_owned();
    let mut root_frame = "base_link".to_owned();
    let mut max_delta_ns = MAX_DELTA_NS;
    let mut chunk_points = CHUNK_POINTS;
    let mut source_memory_bytes = SOURCE_MEMORY_BYTES;
    let mut max_records = EPISODE_MAX_RECORDS;
    let mut max_points = EPISODE_MAX_POINTS;
    let mut max_bytes = EPISODE_MAX_BYTES;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--calibration-evidence" => {
                calibration_evidence = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--front-topic" => front_topic = next_value(&mut args, &flag)?,
            "--rear-topic" => rear_topic = next_value(&mut args, &flag)?,
            "--root-frame" => root_frame = next_value(&mut args, &flag)?,
            "--max-delta-ns" => max_delta_ns = parse_u64(&mut args, &flag)?,
            "--chunk-points" => chunk_points = parse_usize(&mut args, &flag)?,
            "--source-memory-bytes" => source_memory_bytes = parse_u64(&mut args, &flag)?,
            "--max-records" => max_records = parse_u64(&mut args, &flag)?,
            "--max-points" => max_points = parse_u64(&mut args, &flag)?,
            "--max-bytes" => max_bytes = parse_u64(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        input,
        calibration_evidence,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        front_topic,
        rear_topic,
        root_frame,
        max_delta_ns,
        chunk_points,
        source_memory_bytes,
        max_records,
        max_points,
        max_bytes,
        min_output_free_bytes,
    })
}

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, Box<dyn Error>> {
    Ok(next_value(args, flag)?.parse()?)
}

fn parse_usize(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<usize, Box<dyn Error>> {
    Ok(next_value(args, flag)?.parse()?)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn validate_sha256(value: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err("--expected-input-sha256 must be 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    write_text_atomically(path, &format!("{}\n", serde_json::to_string_pretty(value)?))
}

fn write_text_atomically(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("output '{}' already exists", path.display()).into());
    }
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temp, text)?;
    fs::rename(&temp, path)?;
    Ok(())
}

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
}

fn file_label(path: &Path) -> String {
    path.file_name().and_then(|name| name.to_str()).unwrap_or("input").to_owned()
}

fn escape_html(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn render_dashboard(
    state: &FullBagMappingState,
    elapsed_ns: u64,
) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let blockers = state
        .blockers
        .iter()
        .map(|blocker| format!("<li>{}</li>", escape_html(blocker)))
        .collect::<String>();
    Ok(format!(
        r##"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root{{color-scheme:dark;--bg:#050b15;--panel:#0d2032;--line:#245170;--muted:#8ca9bd;--cyan:#5bdcff;--green:#63e7a5;--red:#ff7184;--amber:#ffd166}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#18506e 0,#07111e 38%,#050b15 100%);color:#eef9ff;font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:18px}}h1{{font-size:28px;margin:5px 0}}.eyebrow{{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}}.sub,.mono{{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;color:var(--muted)}}.badge{{border:1px solid var(--line);border-radius:999px;padding:8px 13px;font-weight:800;white-space:nowrap}}.ok{{color:var(--green);border-color:#237850}}.blocked{{color:var(--red);border-color:#873442}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.panel{{background:linear-gradient(145deg,rgba(16,40,61,.96),rgba(7,18,31,.96));border:1px solid var(--line);border-radius:15px;padding:15px;box-shadow:0 14px 34px #0005}}.wide{{grid-column:1/-1}}h2{{color:var(--muted);font-size:11px;letter-spacing:.13em;text-transform:uppercase;margin:0 0 8px}}.metric{{font-size:23px;font-weight:800;color:var(--cyan)}}.danger{{color:var(--red)}}.row{{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #183b55;padding:8px 0}}.row:last-child{{border-bottom:0}}ul{{margin:0;padding-left:20px}}li{{margin:6px 0;color:#ff9aa4}}pre{{max-height:440px;overflow:auto;white-space:pre-wrap}}@media(max-width:800px){{main{{padding:13px}}.grid{{grid-template-columns:1fr 1fr}}header{{display:block}}.badge{{display:inline-block;margin-top:12px}}}}
</style></head><body><main><header><div><div class="eyebrow">SpatialRust / 145K-A bounded full-bag mapping</div><h1>{title}</h1><div class="sub">{source}</div></div><div class="badge {badge_class}">{admission}</div></header><section class="grid"><article class="panel"><h2>Source</h2><div class="metric {source_class}">{source_status}</div><div class="sub">{source_detail}</div></article><article class="panel"><h2>Calibration</h2><div class="metric {calibration_class}">{calibration}</div><div class="sub">clock / frame registration</div></article><article class="panel"><h2>Full bag</h2><div class="metric {full_class}">{full_bag}</div><div class="sub">{records} records · {points} points</div></article><article class="panel"><h2>Mapping</h2><div class="metric {mapping_class}">{mapping}</div><div class="sub">{elapsed_ns} ns wall time</div></article><article class="panel wide"><h2>Stage gates</h2><div class="row"><span>clock applied</span><span>{clock_applied}</span></div><div class="row"><span>frame graph applied</span><span>{frame_applied}</span></div><div class="row"><span>odometry</span><span>{odometry}</span></div><div class="row"><span>TSDF</span><span>{tsdf}</span></div></article><article class="panel wide"><h2>Admission blockers</h2><ul>{blockers}</ul></article><article class="panel wide"><h2>Portable JSON state</h2><pre class="mono" id="state"></pre></article></section></main><script>const s={state_json};document.getElementById('state').textContent=JSON.stringify(s,null,2);</script></body></html>"##,
        title = title,
        source = escape_html(&state.source.path),
        badge_class = if state.summary.mapping_admitted { "ok" } else { "blocked" },
        admission =
            if state.summary.mapping_admitted { "MAPPING ADMITTED" } else { "MAPPING BLOCKED" },
        source_class = if state.source.identity_matches { "" } else { "danger" },
        source_status = if state.source.identity_matches { "MATCH" } else { "MISMATCH" },
        source_detail = state.source.observed_sha256,
        calibration_class = if state.summary.calibration_registered { "" } else { "danger" },
        calibration = if state.summary.calibration_registered { "REGISTERED" } else { "MISSING" },
        full_class = if state.summary.full_bag_processed { "" } else { "danger" },
        full_bag = if state.summary.full_bag_processed { "COMPLETE" } else { "BOUNDED" },
        records = state.source_summary.total_record_count,
        points = state.source_summary.total_point_count,
        mapping_class = if state.summary.mapping_admitted { "" } else { "danger" },
        mapping = if state.summary.mapping_admitted { "ADMITTED" } else { "BLOCKED" },
        elapsed_ns = elapsed_ns,
        clock_applied = state.summary.clock_applied,
        frame_applied = state.summary.frame_graph_applied,
        odometry = state.summary.odometry_complete,
        tsdf = state.summary.tsdf_complete,
        state_json = state_json,
    ))
}

fn usage() -> String {
    String::from(
        "rosbag2_full_bag_mapping INPUT_DB3 --calibration-evidence STATE_JSON --output-dir ABSOLUTE_DIR --expected-input-sha256 SHA256",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_math::TransformPoint;
    use spatialrust_viewer::{CalibrationArtifact, CalibrationEvidenceClock, ClockCalibration};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_full_bag_mapping_options() {
        let config = parse_args(
            [
                "/media/input.db3",
                "--calibration-evidence",
                "/media/calibration.json",
                "--output-dir",
                "/media/output",
                "--expected-input-sha256",
                SHA,
                "--max-points",
                "1024",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.max_points, 1024);
        assert_eq!(config.front_topic, "/lidar_front/points_raw");
        assert_eq!(config.calibration_evidence, Some(PathBuf::from("/media/calibration.json")));
    }

    #[test]
    fn resolves_parent_to_child_frame_path() {
        let frame = CalibrationEvidenceFrame::try_new(
            "fixture",
            "base_link",
            BTreeMap::from([
                ("front".into(), "front_frame".into()),
                ("rear".into(), "rear_frame".into()),
            ]),
            vec!["base_link".into(), "front_frame".into(), "rear_frame".into()],
            vec![
                FrameTransform::try_new(
                    "base_link",
                    "front_frame",
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    None,
                    true,
                    true,
                )
                .unwrap(),
                FrameTransform::try_new(
                    "base_link",
                    "rear_frame",
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    None,
                    true,
                    true,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let transform = resolve_frame_transform(&frame, "base_link", "front_frame").unwrap();
        let point = transform.transform_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((point.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clock_correction_is_deterministic_and_anchored() {
        let source = StudioSource::try_new("fixture", "/media/input.db3", SHA, SHA, true).unwrap();
        let calibration = CalibrationEvidenceState::try_new(
            "fixture",
            source,
            CalibrationArtifact::try_new(
                "clock_evidence",
                "registered",
                Some("/media/clock.json".into()),
                Some(SHA.into()),
                true,
            )
            .unwrap(),
            CalibrationArtifact::try_new(
                "frame_evidence",
                "registered",
                Some("/media/frame.json".into()),
                Some(SHA.into()),
                true,
            )
            .unwrap(),
            CalibrationEvidenceClock::try_new(
                "sensor",
                "external",
                "fixture",
                ClockCalibration::try_new(
                    "registered",
                    "anchored offset",
                    2,
                    Some(10.0),
                    Some(10.0),
                    Some(0.0),
                    Some(0.0),
                    true,
                    false,
                )
                .unwrap(),
            )
            .unwrap(),
            CalibrationEvidenceFrame::try_new(
                "fixture",
                "base_link",
                BTreeMap::from([("front".into(), "front".into()), ("rear".into(), "rear".into())]),
                vec!["base_link".into(), "front".into(), "rear".into()],
                vec![
                    FrameTransform::try_new(
                        "base_link",
                        "front",
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        None,
                        true,
                        true,
                    )
                    .unwrap(),
                    FrameTransform::try_new(
                        "base_link",
                        "rear",
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        None,
                        true,
                        true,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
            Vec::new(),
        )
        .unwrap();
        let mut correction = ClockCorrection::try_new(&calibration).unwrap();
        assert_eq!(correction.correct(100).unwrap(), 110);
        assert_eq!(correction.correct(200).unwrap(), 210);
    }
}

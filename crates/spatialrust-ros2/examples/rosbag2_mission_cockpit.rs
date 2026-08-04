//! Build a bounded, interactive mission cockpit from source-bound ROS 2 receipts.
//!
//! The example consumes the canonical rosbag2 input only for a small, indexed
//! point sample. Live-publish and edge-partition JSON remain the authority for
//! packet and transfer admission. The generated HTML is self-contained and
//! provides timeline playback, bounded 3D inspection, point selection, and
//! distance measurement without applying an implicit clock or TF transform.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
};
use spatialrust_ros2::Rosbag2PointCloudSource;
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, EpisodeLimits, MemoryEpisodeBuilder, StampedRecord,
    StampedTime,
};
use spatialrust_viewer::{
    CalibrationEvidenceState, EdgePartitionState, LivePublishPacket, LivePublishState,
    MissionCockpitFrame, MissionCockpitLayer, MissionCockpitLink, MissionCockpitNode,
    MissionCockpitPoint, MissionCockpitState, MissionCockpitSummary, MissionCockpitTimeline,
    ReplayArtifact, StudioSource, MISSION_COCKPIT_MAX_SAMPLED_POINTS,
};

const CHUNK_POINTS: usize = 65_536;
const SOURCE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const EPISODE_MAX_POINTS: u64 = 2_000_000;
const EPISODE_MAX_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_SAMPLE_POINTS: usize = 256;
const STATE_FILE: &str = "mission-cockpit.json";
const HTML_FILE: &str = "mission-cockpit.html";
const MANIFEST_FILE: &str = "mission-cockpit.manifest.json";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    live_publish_json: PathBuf,
    edge_partition_json: PathBuf,
    calibration_readiness: PathBuf,
    calibration_evidence: Option<PathBuf>,
    output_dir: PathBuf,
    expected_sha256: String,
    sample_points: usize,
    min_output_free_bytes: u64,
}

#[derive(Debug)]
struct ReadinessGate {
    source_bound: bool,
    registration_ready: bool,
    blockers: Vec<String>,
}

#[derive(Debug)]
struct FrameRun {
    frames: Vec<MissionCockpitFrame>,
    order_verified: bool,
}

#[derive(Debug, Default)]
struct LinkCounters {
    transfer_count: u64,
    completed_transfer_count: u64,
    payload_bytes: u64,
    counted_copy_bytes: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-mission-cockpit: {error}");
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
    let observed_sha256 = input_receipt.sha256.clone().ok_or("input checksum was not produced")?;
    let input_path = config.input.display().to_string();
    let source_identity_match = observed_sha256 == config.expected_sha256;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        input_path.clone(),
        config.expected_sha256.clone(),
        observed_sha256.clone(),
        source_identity_match,
    )?;

    let live_publish: LivePublishState = read_json(&config.live_publish_json)?;
    live_publish.validate()?;
    let edge_partition: EdgePartitionState = read_json(&config.edge_partition_json)?;
    edge_partition.validate()?;
    let readiness_value: Value = read_json(&config.calibration_readiness)?;
    let readiness =
        readiness_gate(&readiness_value, &input_path, &observed_sha256, input_receipt.size_bytes)?;
    let calibration_evidence = config
        .calibration_evidence
        .as_ref()
        .map(|path| read_json::<CalibrationEvidenceState>(path))
        .transpose()?;
    let calibration_evidence_receipt = config
        .calibration_evidence
        .as_ref()
        .map(|path| FileReceipt::from_path(ReceiptRole::Auxiliary, path))
        .transpose()?;

    let mut blockers = Vec::new();
    if !source_identity_match {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            ),
        );
    }
    if live_publish.source.path != input_path
        || live_publish.source.observed_sha256 != observed_sha256
    {
        push_blocker(&mut blockers, "live-publish receipt is bound to a different input");
    }
    if edge_partition.source.path != input_path
        || edge_partition.source.observed_sha256 != observed_sha256
    {
        push_blocker(&mut blockers, "edge-partition receipt is bound to a different input");
    }
    if edge_partition.upstream_live_publish_path != config.live_publish_json.display().to_string() {
        push_blocker(
            &mut blockers,
            "edge-partition receipt references a different live-publish input",
        );
    }
    if edge_partition.calibration_readiness_path
        != config.calibration_readiness.display().to_string()
    {
        push_blocker(
            &mut blockers,
            "edge-partition receipt references a different readiness input",
        );
    }
    if !readiness.source_bound {
        push_blocker(&mut blockers, "calibration readiness receipt is not bound to this input");
    }
    if !readiness.registration_ready {
        push_blocker(
            &mut blockers,
            "calibration readiness registration is incomplete; mapping remains blocked",
        );
    }
    for blocker in &readiness.blockers {
        push_blocker(&mut blockers, format!("readiness: {blocker}"));
    }
    for blocker in &live_publish.blockers {
        push_blocker(&mut blockers, format!("live-publish: {blocker}"));
    }
    for blocker in &edge_partition.blockers {
        push_blocker(&mut blockers, format!("edge-partition: {blocker}"));
    }
    let evidence_registration_ready = if let Some(evidence) = &calibration_evidence {
        evidence.validate()?;
        let source_bound = evidence.source.identity_matches
            && evidence.source.path == input_path
            && evidence.source.observed_sha256 == observed_sha256;
        if !source_bound {
            push_blocker(&mut blockers, "calibration evidence is bound to a different input");
        }
        if !evidence.registration_ready {
            push_blocker(&mut blockers, "calibration evidence registration is incomplete");
        }
        for blocker in &evidence.blockers {
            push_blocker(&mut blockers, format!("calibration-evidence: {blocker}"));
        }
        source_bound && evidence.registration_ready
    } else {
        true
    };
    if !live_publish.summary.calibration_applied || !edge_partition.summary.calibration_applied {
        push_blocker(
            &mut blockers,
            "clock/TF calibration was not applied; cockpit geometry remains in packet frames",
        );
    }

    let upstream_compatible = source_identity_match
        && live_publish.source.identity_matches
        && edge_partition.source.identity_matches
        && live_publish.source.path == input_path
        && edge_partition.source.path == input_path
        && edge_partition.upstream_live_publish_path
            == config.live_publish_json.display().to_string()
        && edge_partition.calibration_readiness_path
            == config.calibration_readiness.display().to_string();
    let frame_run =
        if upstream_compatible && live_publish.publish_ready && edge_partition.partition_ready {
            collect_frames(&config.input, &live_publish, config.sample_points)?
        } else {
            FrameRun { frames: Vec::new(), order_verified: false }
        };
    if upstream_compatible
        && live_publish.publish_ready
        && edge_partition.partition_ready
        && !frame_run.order_verified
    {
        push_blocker(
            &mut blockers,
            "bounded cockpit samples did not match the admitted packet sequence",
        );
    }
    let admitted_frames = upstream_compatible
        && live_publish.publish_ready
        && edge_partition.partition_ready
        && frame_run.order_verified;
    if !admitted_frames {
        push_blocker(&mut blockers, "cockpit packet frames withheld by an upstream admission gate");
    }

    let frames = if admitted_frames { frame_run.frames } else { Vec::new() };
    let nodes = build_nodes(&edge_partition)?;
    let links = if admitted_frames { build_links(&edge_partition)? } else { Vec::new() };
    let layers = build_layers(&live_publish)?;
    let timeline = build_timeline(&frames, &live_publish.time_basis)?;
    let frame_count = u64::try_from(frames.len())?;
    let total_point_count = frames.iter().try_fold(0_u64, |total, frame| {
        total.checked_add(frame.point_count).ok_or("cockpit point count overflow")
    })?;
    let sampled_point_count = frames.iter().try_fold(0_u64, |total, frame| {
        let sample_count = u64::try_from(frame.sampled_points.len())
            .map_err(|_| "cockpit sample count overflow")?;
        total.checked_add(sample_count).ok_or("cockpit sampled point count overflow")
    })?;
    let transfer_count = links.iter().try_fold(0_u64, |total, link| {
        total.checked_add(link.transfer_count).ok_or("cockpit transfer count overflow")
    })?;
    let completed_transfer_count = links.iter().try_fold(0_u64, |total, link| {
        total.checked_add(link.completed_transfer_count).ok_or("cockpit completion count overflow")
    })?;
    let payload_bytes = links.iter().try_fold(0_u64, |total, link| {
        total.checked_add(link.payload_bytes).ok_or("cockpit payload byte overflow")
    })?;
    let counted_copy_bytes = links.iter().try_fold(0_u64, |total, link| {
        total.checked_add(link.counted_copy_bytes).ok_or("cockpit copy byte overflow")
    })?;
    let calibration_registered = admitted_frames
        && readiness.source_bound
        && readiness.registration_ready
        && evidence_registration_ready
        && live_publish.summary.calibration_registered
        && edge_partition.summary.calibration_registered;
    let calibration_applied = calibration_registered
        && live_publish.summary.calibration_applied
        && edge_partition.summary.calibration_applied;
    let summary = MissionCockpitSummary::try_new(
        frame_count,
        frame_count,
        total_point_count,
        sampled_point_count,
        transfer_count,
        completed_transfer_count,
        payload_bytes,
        counted_copy_bytes,
        admitted_frames && live_publish.publish_ready,
        admitted_frames && edge_partition.partition_ready,
        calibration_registered,
        calibration_applied,
        live_publish.time_basis.clone(),
    )?;

    fs::create_dir_all(&config.output_dir)?;
    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    let live_receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, &config.live_publish_json)?;
    let edge_receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, &config.edge_partition_json)?;
    let readiness_receipt =
        FileReceipt::from_path(ReceiptRole::Auxiliary, &config.calibration_readiness)?;
    let mut artifacts = vec![
        replay_artifact("canonical-input", &input_receipt)?,
        replay_artifact("live-publish", &live_receipt)?,
        replay_artifact("edge-partition", &edge_receipt)?,
        replay_artifact("calibration-readiness", &readiness_receipt)?,
    ];
    if let Some(receipt) = &calibration_evidence_receipt {
        artifacts.push(replay_artifact("calibration-evidence", receipt)?);
    }
    let expected_frame_ids = live_publish.expected_frame_ids.clone();
    let state = MissionCockpitState::try_new(
        format!("Spatial Mission Cockpit — {}", file_label(&config.input)),
        source,
        expected_frame_ids,
        timeline,
        frames,
        layers,
        nodes,
        links,
        summary,
        artifacts,
        blockers,
    )?;
    state.validate()?;
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state)?)?;

    let mut manifest = DatasetManifest::new();
    manifest.entries.push(input_receipt);
    manifest.entries.push(live_receipt);
    manifest.entries.push(edge_receipt);
    manifest.entries.push(readiness_receipt);
    if let Some(receipt) = calibration_evidence_receipt {
        manifest.entries.push(receipt);
    }
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Mission Cockpit receipt: {} (publish_ready={}, partition_ready={}, mapping_admitted={})",
        state_path.display(),
        state.publish_ready,
        state.partition_ready,
        state.mapping_admitted
    );
    println!("Mission Cockpit dashboard: {}", html_path.display());
    println!(
        "Mission Cockpit manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.publish_ready {
        return Err(
            "mission cockpit failed its source, upstream, or bounded-sample admission checks"
                .into(),
        );
    }
    Ok(())
}

fn collect_frames(
    input: &Path,
    live_publish: &LivePublishState,
    sample_points: usize,
) -> Result<FrameRun, Box<dyn Error>> {
    let max_records = live_publish.summary.selected_record_count.max(1);
    let mut builder = MemoryEpisodeBuilder::try_new(EpisodeLimits::new(
        max_records,
        EPISODE_MAX_POINTS,
        EPISODE_MAX_BYTES,
    ))?;
    for topic in &live_publish.topics {
        if topic.retained_record_count == 0 {
            continue;
        }
        let options = StreamOptions::new(CHUNK_POINTS, MemoryBudget::new(SOURCE_MEMORY_BYTES)?)?;
        let mut source = Rosbag2PointCloudSource::open(
            input,
            &topic.source_topic,
            options,
            CancellationToken::default(),
        )?;
        let mut retained = 0_u64;
        while retained < topic.retained_record_count {
            let Some(chunk) = source.next_chunk() else {
                break;
            };
            let chunk = chunk?;
            let record = chunk.record().clone();
            let stamp =
                StampedTime::exact("ros2", ClockDomain::External, record.metadata().timestamp);
            builder.push(StampedRecord::new(topic.source_topic.as_str(), stamp, record))?;
            retained = retained.checked_add(1).ok_or("cockpit retained record overflow")?;
        }
        if retained != topic.retained_record_count {
            return Ok(FrameRun { frames: Vec::new(), order_verified: false });
        }
    }
    let episode = builder.finish();
    let mut replayer = DeterministicReplayer::new(&episode);
    let mut records = Vec::new();
    while let Some(record) = replayer.next_record() {
        records.push(record);
    }
    if records.len() != live_publish.packets.len() {
        return Ok(FrameRun { frames: Vec::new(), order_verified: false });
    }
    let mut frames = Vec::with_capacity(records.len());
    for (expected_sequence, (record, packet)) in
        records.iter().zip(&live_publish.packets).enumerate()
    {
        if !packet_matches(record, packet, expected_sequence)? {
            return Ok(FrameRun { frames: Vec::new(), order_verified: false });
        }
        frames.push(sample_frame(record, packet, sample_points)?);
    }
    Ok(FrameRun { frames, order_verified: true })
}

fn packet_matches(
    record: &StampedRecord,
    packet: &LivePublishPacket,
    expected_sequence: usize,
) -> Result<bool, Box<dyn Error>> {
    Ok(packet.sequence == u64::try_from(expected_sequence)?
        && packet.source_topic == record.topic.as_str()
        && packet.frame_id == record.record.metadata().frame_id.0
        && packet.stamp_nanos == record.stamp.as_nanos()
        && packet.point_count == u64::try_from(record.record.cloud().len())?)
}

fn sample_frame(
    record: &StampedRecord,
    packet: &LivePublishPacket,
    sample_points: usize,
) -> Result<MissionCockpitFrame, Box<dyn Error>> {
    let cloud = record.record.cloud();
    let x = cloud.field("x")?.as_f32()?;
    let y = cloud.field("y")?.as_f32()?;
    let z = cloud.field("z")?.as_f32()?;
    if x.len() != y.len() || x.len() != z.len() || x.is_empty() {
        return Err("source XYZ fields have inconsistent or empty lengths".into());
    }
    let limit = sample_points.min(MISSION_COCKPIT_MAX_SAMPLED_POINTS).min(x.len());
    if limit == 0 {
        return Err("--sample-points must retain at least one source point".into());
    }
    let mut indices = Vec::with_capacity(limit);
    let mut points = Vec::with_capacity(limit);
    for sample_index in 0..limit {
        let source_index =
            sample_index.checked_mul(x.len()).ok_or("cockpit sample index overflow")? / limit;
        indices.push(u64::try_from(source_index)?);
        points.push(MissionCockpitPoint::try_new(
            x[source_index],
            y[source_index],
            z[source_index],
        )?);
    }
    Ok(MissionCockpitFrame::try_new(
        packet.sequence,
        packet.source_topic.clone(),
        packet.publish_topic.clone(),
        packet.frame_id.clone(),
        packet.stamp_nanos,
        packet.point_count,
        indices,
        points,
    )?)
}

fn build_layers(
    live_publish: &LivePublishState,
) -> Result<Vec<MissionCockpitLayer>, Box<dyn Error>> {
    let colors = [[91, 220, 255], [255, 143, 188], [255, 209, 102], [166, 130, 255]];
    let mut layers = Vec::with_capacity(live_publish.topics.len() + 1);
    for (index, topic) in live_publish.topics.iter().enumerate() {
        layers.push(MissionCockpitLayer::try_new(
            format!("point-cloud-{index}"),
            format!("{} point cloud", topic.source_topic),
            "point-cloud",
            true,
            vec![topic.source_topic.clone()],
            colors[index % colors.len()],
        )?);
    }
    layers.push(MissionCockpitLayer::try_new(
        "transfer-graph",
        "Edge → host transfer graph",
        "transfer-graph",
        true,
        Vec::new(),
        [99, 231, 165],
    )?);
    Ok(layers)
}

fn build_nodes(
    edge_partition: &EdgePartitionState,
) -> Result<Vec<MissionCockpitNode>, Box<dyn Error>> {
    let partition_count = edge_partition.partitions.len();
    let mut nodes = Vec::new();
    for (partition_index, partition) in edge_partition.partitions.iter().enumerate() {
        let x = if partition_count <= 1 {
            0.5
        } else {
            0.18 + 0.64 * partition_index as f32 / (partition_count - 1) as f32
        };
        let denominator = partition.node_ids.len().max(1) as f32;
        for (node_index, node_id) in partition.node_ids.iter().enumerate() {
            let y = 0.22 + 0.56 * (node_index as f32 + 0.5) / denominator;
            nodes.push(MissionCockpitNode::try_new(
                node_id.clone(),
                partition.id.clone(),
                partition.placement.clone(),
                x,
                y,
            )?);
        }
    }
    Ok(nodes)
}

fn build_links(
    edge_partition: &EdgePartitionState,
) -> Result<Vec<MissionCockpitLink>, Box<dyn Error>> {
    let mut grouped = BTreeMap::<(String, String), LinkCounters>::new();
    for transfer in &edge_partition.transfers {
        let counter =
            grouped.entry((transfer.from_node.clone(), transfer.to_node.clone())).or_default();
        counter.transfer_count =
            counter.transfer_count.checked_add(1).ok_or("cockpit transfer count overflow")?;
        if transfer.completed {
            counter.completed_transfer_count = counter
                .completed_transfer_count
                .checked_add(1)
                .ok_or("cockpit completion count overflow")?;
        }
        counter.payload_bytes = counter
            .payload_bytes
            .checked_add(transfer.payload_bytes)
            .ok_or("cockpit payload byte overflow")?;
        counter.counted_copy_bytes = counter
            .counted_copy_bytes
            .checked_add(transfer.counted_copy_bytes)
            .ok_or("cockpit copy byte overflow")?;
    }
    grouped
        .into_iter()
        .map(|((from_node, to_node), counter)| {
            Ok(MissionCockpitLink::try_new(
                from_node,
                to_node,
                counter.transfer_count,
                counter.completed_transfer_count,
                counter.payload_bytes,
                counter.counted_copy_bytes,
            )?)
        })
        .collect()
}

fn build_timeline(
    frames: &[MissionCockpitFrame],
    time_basis: &str,
) -> Result<MissionCockpitTimeline, Box<dyn Error>> {
    let (start_nanos, end_nanos) = match (frames.first(), frames.last()) {
        (Some(first), Some(last)) => (first.stamp_nanos, last.stamp_nanos),
        _ => (0, 0),
    };
    MissionCockpitTimeline::try_new(
        time_basis,
        start_nanos,
        end_nanos,
        start_nanos,
        u64::try_from(frames.len())?,
    )
    .map_err(Into::into)
}

fn readiness_gate(
    readiness: &Value,
    input_path: &str,
    observed_sha256: &str,
    input_size: Option<u64>,
) -> Result<ReadinessGate, Box<dyn Error>> {
    let mut blockers = Vec::new();
    if readiness.get("schema").and_then(Value::as_str)
        != Some("spatialrust.rosbag2.calibration-readiness")
    {
        blockers.push("calibration readiness schema is unsupported".into());
    }
    if readiness.get("version").and_then(Value::as_u64) != Some(1) {
        blockers.push("calibration readiness version is unsupported".into());
    }
    let source_path = string_at(readiness, &["input", "path"]);
    let source_sha = string_at(readiness, &["input", "sha256"]);
    let source_size = readiness.get("input").and_then(|input| input.get("size_bytes"));
    let source_bound = source_path.as_deref() == Some(input_path)
        && source_sha.as_deref() == Some(observed_sha256)
        && source_size.and_then(Value::as_u64) == input_size;
    let registration_ready =
        readiness.get("registration_ready").and_then(Value::as_bool).unwrap_or(false);
    for blocker in string_array(readiness.get("blockers")) {
        push_blocker(&mut blockers, blocker);
    }
    Ok(ReadinessGate { source_bound, registration_ready, blockers })
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str().map(str::to_owned)
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flat_map(|values| values.iter())
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
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

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.input.is_absolute()
        || !config.live_publish_json.is_absolute()
        || !config.edge_partition_json.is_absolute()
        || !config.calibration_readiness.is_absolute()
        || config.calibration_evidence.as_ref().is_some_and(|path| !path.is_absolute())
        || !config.output_dir.is_absolute()
    {
        return Err(
            "input, receipt, --calibration-readiness, --calibration-evidence, and --output-dir paths must be absolute".into(),
        );
    }
    if config.sample_points == 0 || config.sample_points > MISSION_COCKPIT_MAX_SAMPLED_POINTS {
        return Err(format!(
            "--sample-points must be between 1 and {}",
            MISSION_COCKPIT_MAX_SAMPLED_POINTS
        )
        .into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    let mut live_publish_json = None;
    let mut edge_partition_json = None;
    let mut calibration_readiness = None;
    let mut calibration_evidence = None;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut sample_points = DEFAULT_SAMPLE_POINTS;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--live-publish-json" => {
                live_publish_json = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--edge-partition-json" => {
                edge_partition_json = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--calibration-readiness" => {
                calibration_readiness = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--calibration-evidence" => {
                calibration_evidence = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--sample-points" => sample_points = parse_usize(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        input: PathBuf::from(input),
        live_publish_json: live_publish_json.ok_or("--live-publish-json is required")?,
        edge_partition_json: edge_partition_json.ok_or("--edge-partition-json is required")?,
        calibration_readiness: calibration_readiness
            .ok_or("--calibration-readiness is required")?,
        calibration_evidence,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        sample_points,
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

fn usage() -> String {
    String::from("rosbag2_mission_cockpit <input.db3> --help")
}

fn render_dashboard(state: &MissionCockpitState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#050b15;--panel:#0b1828;--panel2:#10283d;--line:#245170;--muted:#8ca9bd;--cyan:#5bdcff;--green:#63e7a5;--pink:#ff8fbc;--red:#ff7184;--amber:#ffd166;--violet:#a682ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#18506e 0,#07111e 38%,#050b15 100%);color:#eef9ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1650px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:18px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}.title{font-size:30px;font-weight:800;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:800;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(16,40,61,.96),rgba(7,18,31,.96));border:1px solid var(--line);border-radius:15px;padding:15px;box-shadow:0 14px 34px #0005}.panel h2{color:var(--muted);font-size:11px;letter-spacing:.13em;text-transform:uppercase;margin:0 0 10px}.metric{font-size:24px;font-weight:800;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.wide{grid-column:span 2}.full{grid-column:1/-1}.scene-panel{grid-column:span 4;padding:12px}.side-panel{grid-column:span 2}.scene-wrap{position:relative;min-height:560px;border:1px solid #1d4561;border-radius:12px;overflow:hidden;background:#040b14}.scene-wrap canvas{display:block;width:100%;height:560px;cursor:grab}.scene-wrap canvas.dragging{cursor:grabbing}.hint{position:absolute;left:13px;bottom:10px;color:#8ea9bd;background:#06111cdd;border:1px solid #214964;border-radius:8px;padding:7px 9px;font-size:11px}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:11px}.controls button,.controls input[type=range]{accent-color:var(--cyan)}button{background:#12344e;border:1px solid #2d6d93;color:#effbff;border-radius:8px;padding:7px 11px;cursor:pointer}button:hover{background:#1c4e6e}.range{flex:1;min-width:180px}.readout{color:var(--cyan);font:12px ui-monospace,monospace;min-width:180px}.layer-list{display:grid;gap:8px}.layer{display:flex;align-items:center;gap:8px;border:1px solid #1d4561;border-radius:9px;padding:8px;background:#081725}.swatch{width:10px;height:10px;border-radius:50%;display:inline-block}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #183b55;padding:8px 0}.row:last-child{border-bottom:0}.danger{color:var(--red)}.warning{color:var(--amber)}.pass{color:var(--green)}.selection{min-height:126px;background:#071522;border:1px solid #1d4561;border-radius:10px;padding:10px}.measure{font-size:23px;color:var(--amber);font-weight:800}.graph-panel{grid-column:span 3}.graph-panel canvas{width:100%;height:240px;display:block}.timeline-panel{grid-column:span 3}.bar{height:9px;background:#06111c;border-radius:5px;overflow:hidden;margin:12px 0 4px}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}.chips{display:flex;flex-wrap:wrap;gap:7px}.chip{background:#10324d;border:1px solid #2d638a;border-radius:999px;padding:5px 9px;font-size:11px}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.artifact{border-bottom:1px solid #183b55;padding:8px 0;overflow-wrap:anywhere}.artifact:last-child{border-bottom:0}.raw{max-height:300px;overflow:auto;white-space:pre-wrap;margin:0}.empty{color:var(--muted);padding:20px;text-align:center}@media(max-width:1100px){.scene-panel,.side-panel,.graph-panel,.timeline-panel{grid-column:span 6}}@media(max-width:650px){main{padding:13px}.grid{grid-template-columns:1fr 1fr}.scene-panel,.side-panel,.graph-panel,.timeline-panel,.wide,.full{grid-column:1/-1}.scene-wrap,.scene-wrap canvas{min-height:400px;height:400px}.top{display:block}.badge{display:inline-block;margin-top:13px}}
</style></head><body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / 145J-A interactive mission cockpit</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Source</h2><div id="sourceStatus" class="metric"></div><div id="sourceDetail" class="small"></div></article>
<article class="panel"><h2>Publish</h2><div id="publish" class="metric"></div><div id="publishDetail" class="small"></div></article>
<article class="panel"><h2>Partition</h2><div id="partition" class="metric"></div><div id="partitionDetail" class="small"></div></article>
<article class="panel"><h2>Mapping</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Frames</h2><div id="frames" class="metric"></div><div id="framesDetail" class="small"></div></article>
<article class="panel"><h2>Samples</h2><div id="samples" class="metric"></div><div id="samplesDetail" class="small"></div></article>
<article class="panel scene-panel"><h2>Interactive packet scene</h2><div class="scene-wrap"><canvas id="scene"></canvas><div class="hint">drag: orbit · wheel: zoom · click: select · shift-click: measure</div></div><div class="controls"><button id="play">▶ Play</button><button id="reset">Reset view</button><input id="cursor" class="range" type="range" min="0" max="0" value="0"><span id="cursorReadout" class="readout">no admitted frame</span></div></article>
<article class="panel side-panel"><h2>Layers</h2><div id="layers" class="layer-list"></div><h2 style="margin-top:18px">Selection / measurement</h2><div id="selection" class="selection"><span class="small">Select a sampled point.</span></div><div class="row"><span>time basis</span><span id="timeBasis" class="mono"></span></div><div class="row"><span>payload / explicit copy</span><span id="bytes" class="mono"></span></div></article>
<article class="panel graph-panel"><h2>Edge → host execution graph</h2><canvas id="graph"></canvas><div id="graphDetail" class="small"></div></article>
<article class="panel timeline-panel"><h2>Timeline / packet topics</h2><div id="timeline" class="small"></div><div class="bar"><div id="timelineBar" class="fill"></div></div><div id="topics" class="chips" style="margin-top:12px"></div></article>
<article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel wide"><h2>Checksummed inputs</h2><div id="artifacts"></div></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="raw mono"></pre></article>
</section></main>
<script id="cockpit-state" type="application/json">__STATE_JSON__</script><script>
const state=JSON.parse(document.getElementById('cockpit-state').textContent),q=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),fmtBytes=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB',esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])),colors=['#5bdcff','#ff8fbc','#ffd166','#a682ff'];
q('title').textContent=state.title;q('source').textContent=state.source.path;q('timeBasis').textContent=state.timeline.time_basis;q('admission').textContent=state.publish_ready?'COCKPIT READY':'COCKPIT BLOCKED';q('admission').className='badge '+(state.publish_ready?'ok':'blocked');
function metric(id,value,ok){q(id).textContent=value;q(id).className='metric '+(ok===false?'danger':'')}metric('sourceStatus',state.source.identity_matches?'MATCH':'MISMATCH',state.source.identity_matches);q('sourceDetail').textContent=state.source.observed_sha256.slice(0,16)+'…';metric('publish',state.publish_ready?'READY':'BLOCKED',state.publish_ready);q('publishDetail').textContent=state.summary.source_packet_count+' admitted packets';metric('partition',state.partition_ready?'READY':'BLOCKED',state.partition_ready);q('partitionDetail').textContent=state.summary.completed_transfer_count+' / '+state.summary.transfer_count+' transfers';metric('mapping',state.mapping_admitted?'ADMITTED':'INSPECTION ONLY',state.mapping_admitted);q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'packet-frame coordinates';metric('frames',fmt(state.summary.frame_count),state.summary.frame_count>0);q('framesDetail').textContent=fmt(state.summary.total_point_count)+' source points';metric('samples',fmt(state.summary.sampled_point_count),state.summary.sampled_point_count>0);q('samplesDetail').textContent='bounded interaction points';q('bytes').textContent=fmtBytes(state.summary.payload_bytes)+' / '+fmtBytes(state.summary.counted_copy_bytes);
const frames=state.frames,slider=q('cursor');slider.max=Math.max(0,frames.length-1);let frameIndex=0,yaw=.62,pitch=-.34,zoom=1,dragging=false,moved=false,lastX=0,lastY=0,selected=null,measureEnd=null,playTimer=null;
const visible={};state.layers.forEach(l=>visible[l.id]=l.visible);q('layers').innerHTML=state.layers.filter(l=>l.kind==='point-cloud').map((l,i)=>'<label class="layer"><input type="checkbox" data-layer="'+esc(l.id)+'" '+(l.visible?'checked':'')+'><span class="swatch" style="background:'+colors[i%colors.length]+'"></span><span>'+esc(l.label)+'</span></label>').join('')||'<div class="empty">No point-cloud layers admitted</div>';
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{visible[input.dataset.layer]=input.checked;drawScene()}));
function activeFrame(){return frames[frameIndex]};function pointLayer(frame){return state.layers.find(l=>l.kind==='point-cloud'&&l.frame_ids.includes(frame.source_topic))};function frameColor(frame){const layer=pointLayer(frame),i=state.layers.indexOf(layer);return colors[Math.max(0,i)%colors.length]};
function bounds(frame){const pts=frame?frame.sampled_points.flatMap(p=>[p.x,p.y,p.z]):frames.flatMap(f=>f.sampled_points.flatMap(p=>[p.x,p.y,p.z]));if(!pts.length)return {cx:0,cy:0,cz:0,span:1};let min=[pts[0],pts[1],pts[2]],max=[pts[0],pts[1],pts[2]];for(let i=3;i<pts.length;i+=3)for(let a=0;a<3;a++){min[a]=Math.min(min[a],pts[i+a]);max[a]=Math.max(max[a],pts[i+a])}return {cx:(min[0]+max[0])/2,cy:(min[1]+max[1])/2,cz:(min[2]+max[2])/2,span:Math.max(max[0]-min[0],max[1]-min[1],max[2]-min[2],.001)}}
const scene=q('scene');const ctx=scene.getContext('2d');function resize(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=Math.max(1,Math.round(r.width*d));c.height=Math.max(1,Math.round(r.height*d));return [c.width,c.height,d]}function project(p,b,w,h){let x=p.x-b.cx,y=p.y-b.cy,z=p.z-b.cz;let rx=x*Math.cos(yaw)-y*Math.sin(yaw),ry=x*Math.sin(yaw)+y*Math.cos(yaw),rz=z;let py=ry*Math.cos(pitch)-rz*Math.sin(pitch),depth=ry*Math.sin(pitch)+rz*Math.cos(pitch);let s=Math.min(w,h)/b.span*.42*zoom,den=Math.max(.25,1+depth/(b.span*2.6));return [w/2+rx*s/den,h/2-py*s/den,depth]}
function drawScene(){const [w,h,d]=resize(scene);ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w/d,h/d);const cw=w/d,ch=h/d;ctx.fillStyle='#040b14';ctx.fillRect(0,0,cw,ch);const f=activeFrame(),b=bounds(f);ctx.strokeStyle='#12324a';ctx.lineWidth=1;for(let i=-5;i<=5;i++){let x=cw/2+i*cw/12,y=ch/2+i*ch/12;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,ch);ctx.stroke();ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cw,y);ctx.stroke()}ctx.strokeStyle='#285773';ctx.beginPath();ctx.moveTo(cw/2,0);ctx.lineTo(cw/2,ch);ctx.moveTo(0,ch/2);ctx.lineTo(cw,ch/2);ctx.stroke();if(!f){ctx.fillStyle='#8ca9bd';ctx.textAlign='center';ctx.fillText('No admitted packet sample',cw/2,ch/2);return}f.sampled_points.forEach((p,i)=>{const layer=pointLayer(f);if(!layer||!visible[layer.id])return;const [x,y,depth]=project(p,b,cw,ch);const r=Math.max(1.5,3.5-depth/b.span);ctx.fillStyle=frameColor(f);ctx.globalAlpha=.55+Math.min(.45,Math.max(0,(depth/b.span+.5)));ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;if(selected&&selected.frameIndex===frameIndex&&selected.pointIndex===i){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,r+5,0,Math.PI*2);ctx.stroke()}});q('cursorReadout').textContent='#'+f.sequence+' · '+f.source_topic+' · '+(f.stamp_nanos/1e9).toFixed(6)+' s · '+fmt(f.point_count)+' pts';q('timelineBar').style.width=frames.length?((frameIndex+1)/frames.length*100)+'%':'0%';q('selection').innerHTML=selected?selectionHtml(selected):'<span class="small">Select a sampled point.</span>'}
function selectionHtml(s){const f=frames[s.frameIndex],p=f.sampled_points[s.pointIndex],idx=f.sampled_source_indices[s.pointIndex];let out='<div><strong>packet #'+f.sequence+' / source index '+idx+'</strong></div><div class="small">'+esc(f.source_topic)+' · '+esc(f.frame_id)+'</div><div class="mono">XYZ ['+p.x.toFixed(4)+', '+p.y.toFixed(4)+', '+p.z.toFixed(4)+']</div>';if(measureEnd){const g=frames[measureEnd.frameIndex].sampled_points[measureEnd.pointIndex],dist=Math.hypot(p.x-g.x,p.y-g.y,p.z-g.z);out+='<div class="measure">'+dist.toFixed(4)+' m</div><div class="small">shift-click another point to replace endpoint</div>'}return out}
function nearest(event){const f=activeFrame();if(!f)return null;const r=scene.getBoundingClientRect(),x=(event.clientX-r.left)/r.width*scene.width/(window.devicePixelRatio||1),y=(event.clientY-r.top)/r.height*scene.height/(window.devicePixelRatio||1),b=bounds(f);let best=null,bd=16;f.sampled_points.forEach((p,i)=>{const layer=pointLayer(f);if(!layer||!visible[layer.id])return;const [px,py]=project(p,b,scene.width/(window.devicePixelRatio||1),scene.height/(window.devicePixelRatio||1)),dist=Math.hypot(px-x,py-y);if(dist<bd){bd=dist;best={frameIndex,pointIndex:i}}});return best}
scene.addEventListener('pointerdown',e=>{dragging=true;moved=false;lastX=e.clientX;lastY=e.clientY;scene.classList.add('dragging');scene.setPointerCapture(e.pointerId)});scene.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;lastX=e.clientX;lastY=e.clientY;yaw+=dx*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.008));drawScene()});scene.addEventListener('pointerup',e=>{dragging=false;scene.classList.remove('dragging');if(!moved){const hit=nearest(e);if(hit){if(e.shiftKey&&selected)measureEnd=hit;else{selected=hit;measureEnd=null}drawScene()}}});scene.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(5,zoom*(e.deltaY<0?1.1:.9)));drawScene()},{passive:false});
slider.addEventListener('input',()=>{frameIndex=Number(slider.value);selected=null;measureEnd=null;drawScene()});q('reset').onclick=()=>{yaw=.62;pitch=-.34;zoom=1;drawScene()};q('play').onclick=()=>{if(playTimer){clearInterval(playTimer);playTimer=null;q('play').textContent='▶ Play'}else{q('play').textContent='⏸ Pause';playTimer=setInterval(()=>{if(!frames.length)return;frameIndex=(frameIndex+1)%frames.length;slider.value=frameIndex;drawScene()},700)}};q('timeline').textContent=frames.length?((state.timeline.start_nanos/1e9).toFixed(6)+' → '+(state.timeline.end_nanos/1e9).toFixed(6)+' s · '+fmt(frames.length)+' ordered packets'):'unavailable';q('topics').innerHTML=state.layers.filter(l=>l.kind==='point-cloud').map(l=>'<span class="chip">'+esc(l.label)+'</span>').join('')||'<span class="small">No admitted topics</span>';
const graph=q('graph'),gctx=graph.getContext('2d');function drawGraph(){const [w,h,d]=resize(graph);gctx.setTransform(d,0,0,d,0,0);const cw=w/d,ch=h/d;gctx.clearRect(0,0,cw,ch);gctx.fillStyle='#071522';gctx.fillRect(0,0,cw,ch);const nodeMap={};state.nodes.forEach(n=>nodeMap[n.id]=n);state.links.forEach(l=>{const a=nodeMap[l.from_node],b=nodeMap[l.to_node];if(!a||!b)return;gctx.strokeStyle='#63e7a5';gctx.globalAlpha=.72;gctx.lineWidth=Math.max(2,Math.min(8,l.transfer_count));gctx.beginPath();gctx.moveTo(a.display_x*cw,a.display_y*ch);gctx.lineTo(b.display_x*cw,b.display_y*ch);gctx.stroke();gctx.globalAlpha=1});state.nodes.forEach(n=>{const x=n.display_x*cw,y=n.display_y*ch;gctx.fillStyle=n.placement==='edge'?'#5bdcff':'#ff8fbc';gctx.beginPath();gctx.arc(x,y,9,0,Math.PI*2);gctx.fill();gctx.fillStyle='#eef9ff';gctx.textAlign='center';gctx.font='11px ui-monospace';gctx.fillText(n.id,x,y-16);gctx.fillStyle='#8ca9bd';gctx.fillText(n.placement,x,y+25)});q('graphDetail').textContent=state.links.length?state.links.map(l=>l.from_node+' → '+l.to_node+' · '+fmt(l.transfer_count)+' packets · '+fmtBytes(l.counted_copy_bytes)+' explicit copy').join(' | '):'No admitted transfer lanes'}
q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="pass">All gates passed</li>';q('artifacts').innerHTML=state.artifacts.map(a=>'<div class="artifact"><strong>'+esc(a.role)+'</strong> · '+fmt(a.size_bytes)+' bytes<br><span class="mono">'+esc(a.path)+'<br>'+esc(a.sha256)+'</span></div>').join('');q('raw').textContent=JSON.stringify(state,null,2);window.addEventListener('resize',()=>{drawScene();drawGraph()});drawScene();drawGraph();
</script></body></html>"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_cockpit_options() {
        let config = parse_args(
            [
                "/media/input.db3",
                "--live-publish-json",
                "/media/live.json",
                "--edge-partition-json",
                "/media/edge.json",
                "--calibration-readiness",
                "/media/readiness.json",
                "--calibration-evidence",
                "/media/evidence.json",
                "--output-dir",
                "/media/output",
                "--expected-input-sha256",
                SHA,
                "--sample-points",
                "128",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.sample_points, 128);
        assert_eq!(config.live_publish_json, PathBuf::from("/media/live.json"));
        assert_eq!(config.calibration_evidence, Some(PathBuf::from("/media/evidence.json")));
    }

    #[test]
    fn rejects_sample_budget_above_contract() {
        let config = Config {
            input: "/media/input.db3".into(),
            live_publish_json: "/media/live.json".into(),
            edge_partition_json: "/media/edge.json".into(),
            calibration_readiness: "/media/readiness.json".into(),
            calibration_evidence: None,
            output_dir: "/media/output".into(),
            expected_sha256: SHA.into(),
            sample_points: MISSION_COCKPIT_MAX_SAMPLED_POINTS + 1,
            min_output_free_bytes: 1,
        };
        assert!(validate_config(&config).is_err());
    }
}

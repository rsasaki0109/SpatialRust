//! Publish a bounded, source-bound PointCloud2 episode through an explicit adapter.
//!
//! The default adapter is an in-process loopback because native `rclrs` remains
//! outside the portable workspace boundary. The command still exercises the
//! same CDR encoder/decoder and records topic, frame, payload, queue, and
//! round-trip receipts. It never applies an implicit clock or TF transform.

use std::{
    collections::{BTreeMap, BTreeSet},
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
use spatialrust_ros2::{list_topics, Rosbag2PointCloudSource, Rosbag2Topic};
use spatialrust_runtime::{
    decode_point_cloud2_xyz, encode_point_cloud2_xyz, LoopbackRos2Node, PointCloud2Xyz,
    POINT_CLOUD2_TYPE,
};
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, EpisodeLimits, MemoryEpisode, MemoryEpisodeBuilder,
    StampedRecord, StampedTime,
};
use spatialrust_viewer::{
    LivePublishPacket, LivePublishState, LivePublishSummary, LivePublishTopic,
    LivePublishTransport, ReplayArtifact, StudioSource,
};

const CHUNK_POINTS: usize = 65_536;
const SOURCE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const EPISODE_MAX_POINTS: u64 = 2_000_000;
const EPISODE_MAX_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_MAX_RECORDS_PER_TOPIC: u64 = 2;
const DEFAULT_FRONT_TOPIC: &str = "/lidar_front/points_raw";
const DEFAULT_REAR_TOPIC: &str = "/lidar_rear/points_raw";
const DEFAULT_PUBLISH_PREFIX: &str = "/spatialrust";
const STATE_FILE: &str = "live-publish.json";
const HTML_FILE: &str = "live-publish.html";
const MANIFEST_FILE: &str = "live-publish.manifest.json";
const TIME_BASIS: &str =
    "PointCloud2 header stamp; ros2-external domain; no clock calibration applied";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    front_frame_id: String,
    rear_frame_id: String,
    calibration_readiness: PathBuf,
    publish_prefix: String,
    front_topic: String,
    rear_topic: String,
    max_records_per_topic: u64,
    min_output_free_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordSignature {
    stamp_nanos: u64,
    topic: String,
    frame_id: String,
    point_count: u64,
}

#[derive(Debug)]
struct SourceTopicRun {
    name: String,
    message_count: u64,
    retained_record_count: u64,
    retained_point_count: u64,
    frame_ids: Vec<String>,
    peak_source_bytes: u64,
}

#[derive(Debug)]
struct CollectedEpisode {
    episode: MemoryEpisode,
    topics: Vec<SourceTopicRun>,
    selected_record_count: u64,
    selected_point_count: u64,
    selected_bytes: u64,
    deterministic_order_verified: bool,
    frame_identity_match: bool,
    peak_source_bytes: u64,
}

#[derive(Debug, Default)]
struct PublishRun {
    packets: Vec<LivePublishPacket>,
    host_encode_bytes: u64,
    host_decode_bytes: u64,
}

#[derive(Debug)]
struct ReadinessGate {
    source_bound: bool,
    registration_ready: bool,
    blockers: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-live-publish: {error}");
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
    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let observed_sha256 = input.sha256.clone().ok_or("input checksum was not produced")?;
    let source_identity_match = observed_sha256 == config.expected_sha256;
    let input_display = config.input.display().to_string();
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        input_display.clone(),
        config.expected_sha256.clone(),
        observed_sha256.clone(),
        source_identity_match,
    )?;
    let readiness = read_json(&config.calibration_readiness)?;
    let readiness_gate =
        readiness_gate(&readiness, &input_display, &observed_sha256, input.size_bytes)?;
    let available_topics = list_topics(&config.input)?;
    let source_message_count = relevant_message_count(&available_topics, &config);

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
    blockers.extend(readiness_gate.blockers.iter().cloned());
    push_blocker(
        &mut blockers,
        "clock/TF calibration was not applied; published packets remain inspection-only",
    );

    let collected = if source_identity_match {
        collect_episode(&config, &available_topics)?
    } else {
        empty_collection(&available_topics, &config)
    };
    if collected.selected_record_count == 0 {
        push_blocker(&mut blockers, "bounded source episode retained no records");
    }
    if !collected.frame_identity_match {
        push_blocker(
            &mut blockers,
            format!(
                "frame identity mismatch: expected `{}` for `{}` and `{}` for `{}`",
                config.front_frame_id, config.front_topic, config.rear_frame_id, config.rear_topic
            ),
        );
    }
    if !collected.deterministic_order_verified {
        push_blocker(&mut blockers, "deterministic publish order verification failed");
    }
    if !readiness_gate.source_bound {
        push_blocker(&mut blockers, "calibration readiness receipt is not bound to this input");
    } else if !readiness_gate.registration_ready {
        push_blocker(
            &mut blockers,
            "calibration readiness registration is incomplete; mapping remains blocked",
        );
    }

    let can_publish = source_identity_match
        && collected.selected_record_count > 0
        && collected.frame_identity_match
        && collected.deterministic_order_verified;
    let publish_run = if can_publish {
        match publish_episode(&config, &collected.episode) {
            Ok(run) => run,
            Err(error) => {
                push_blocker(&mut blockers, format!("publish adapter failed: {error}"));
                PublishRun::default()
            }
        }
    } else {
        PublishRun::default()
    };
    let publish_topic_names = topic_names(&collected.topics, &config);
    let topics = build_topics(&collected.topics, &publish_topic_names, &publish_run.packets)?;
    let published_message_count = u64::try_from(publish_run.packets.len())?;
    let published_point_count = publish_run
        .packets
        .iter()
        .try_fold(0_u64, |total, packet| total.checked_add(packet.point_count))
        .ok_or("published point count overflow")?;
    let received_message_count =
        publish_run.packets.iter().filter(|packet| packet.roundtrip_verified).count() as u64;
    let transport = LivePublishTransport::try_new(
        "in-process-loopback",
        POINT_CLOUD2_TYPE,
        "replace-latest-per-topic; publish then take",
        1,
        published_message_count,
        received_message_count,
        0,
        publish_run.host_encode_bytes,
        publish_run.host_decode_bytes,
        0,
        0,
    )?;
    let summary = LivePublishSummary::try_new(
        source_message_count,
        collected.selected_record_count,
        collected.selected_point_count,
        collected.selected_bytes,
        collected.peak_source_bytes,
        published_message_count,
        received_message_count,
        published_point_count,
        collected.deterministic_order_verified,
        collected.frame_identity_match,
        readiness_gate.registration_ready && readiness_gate.source_bound,
        false,
        TIME_BASIS,
    )?;
    let expected_frame_ids = BTreeMap::from([
        (config.front_topic.clone(), config.front_frame_id.clone()),
        (config.rear_topic.clone(), config.rear_frame_id.clone()),
    ]);
    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    fs::create_dir_all(&config.output_dir)?;
    let input_artifact = replay_artifact("canonical-input", &input)?;
    let readiness_receipt =
        FileReceipt::from_path(ReceiptRole::Auxiliary, &config.calibration_readiness)?;
    let readiness_artifact = replay_artifact("calibration-readiness", &readiness_receipt)?;
    let state = LivePublishState::try_new(
        format!("ROS 2 Live Publish Bridge — {}", file_label(&config.input)),
        source,
        expected_frame_ids,
        TIME_BASIS,
        topics,
        transport,
        publish_run.packets,
        summary,
        vec![input_artifact, readiness_artifact],
        blockers,
    )?;
    state.validate()?;
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state)?)?;

    let mut manifest = DatasetManifest::new();
    manifest.entries.push(input);
    manifest.entries.push(readiness_receipt);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;
    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Live Publish Bridge: {} (publish_ready={}, mapping_admitted={})",
        state_path.display(),
        state.publish_ready,
        state.mapping_admitted
    );
    println!("Live Publish dashboard: {}", html_path.display());
    println!(
        "Live Publish manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.publish_ready {
        return Err("live publish failed its source, frame, or round-trip admission checks".into());
    }
    Ok(())
}

fn collect_episode(
    config: &Config,
    available_topics: &[Rosbag2Topic],
) -> Result<CollectedEpisode, Box<dyn Error>> {
    let mut blockers = Vec::new();
    let front_present = available_topics.iter().any(|topic| topic.name == config.front_topic);
    let rear_present = available_topics.iter().any(|topic| topic.name == config.rear_topic);
    if !front_present {
        blockers.push(format!("topic '{}' is missing", config.front_topic));
    }
    if !rear_present {
        blockers.push(format!("topic '{}' is missing", config.rear_topic));
    }
    if !blockers.is_empty() {
        return Ok(empty_collection(available_topics, config));
    }

    let max_records =
        config.max_records_per_topic.checked_mul(2).ok_or("episode record limit overflow")?;
    let mut builder = MemoryEpisodeBuilder::try_new(EpisodeLimits::new(
        max_records,
        EPISODE_MAX_POINTS,
        EPISODE_MAX_BYTES,
    ))?;
    let front = append_topic(
        &mut builder,
        &config.input,
        &config.front_topic,
        config.max_records_per_topic,
    )?;
    let rear = append_topic(
        &mut builder,
        &config.input,
        &config.rear_topic,
        config.max_records_per_topic,
    )?;
    let selected_record_count = u64::try_from(builder.len())?;
    let selected_point_count = builder.points();
    let selected_bytes = builder.bytes();
    let peak_source_bytes = front.peak_source_bytes.max(rear.peak_source_bytes);
    let topics = vec![front, rear];
    let episode = builder.finish();
    let expected = episode.records().iter().map(record_signature).collect::<Result<Vec<_>, _>>()?;
    let mut replayer = DeterministicReplayer::new(&episode);
    let mut observed = Vec::new();
    while let Some(record) = replayer.next_record() {
        observed.push(record_signature(record)?);
    }
    let deterministic_order_verified = expected == observed;
    let frame_identity_match = !expected.is_empty()
        && expected.iter().all(|signature| {
            expected_frame_for(config, &signature.topic)
                .is_some_and(|expected_frame| signature.frame_id == expected_frame)
        });
    Ok(CollectedEpisode {
        episode,
        topics,
        selected_record_count,
        selected_point_count,
        selected_bytes,
        deterministic_order_verified,
        frame_identity_match,
        peak_source_bytes,
    })
}

fn expected_frame_for<'a>(config: &'a Config, topic: &str) -> Option<&'a str> {
    if topic == config.front_topic {
        Some(&config.front_frame_id)
    } else if topic == config.rear_topic {
        Some(&config.rear_frame_id)
    } else {
        None
    }
}

fn append_topic(
    builder: &mut MemoryEpisodeBuilder,
    input: &Path,
    topic_name: &str,
    max_records: u64,
) -> Result<SourceTopicRun, Box<dyn Error>> {
    let options = StreamOptions::new(CHUNK_POINTS, MemoryBudget::new(SOURCE_MEMORY_BYTES)?)?;
    let mut source =
        Rosbag2PointCloudSource::open(input, topic_name, options, CancellationToken::default())?;
    let mut retained_records = 0_u64;
    let mut retained_points = 0_u64;
    let mut frame_ids = BTreeSet::new();
    while retained_records < max_records {
        let Some(chunk) = source.next_chunk() else {
            break;
        };
        let chunk = chunk?;
        let record = chunk.record().clone();
        frame_ids.insert(record.metadata().frame_id.0.clone());
        retained_points = retained_points
            .checked_add(u64::try_from(record.cloud().len())?)
            .ok_or("topic point count overflow")?;
        let stamp = StampedTime::exact("ros2", ClockDomain::External, record.metadata().timestamp);
        builder.push(StampedRecord::new(topic_name, stamp, record))?;
        retained_records =
            retained_records.checked_add(1).ok_or("topic retained record count overflow")?;
    }
    Ok(SourceTopicRun {
        name: topic_name.to_owned(),
        message_count: source.topic().message_count,
        retained_record_count: retained_records,
        retained_point_count: retained_points,
        frame_ids: frame_ids.into_iter().collect(),
        peak_source_bytes: source.memory_tracker().snapshot().peak_bytes,
    })
}

fn empty_collection(available_topics: &[Rosbag2Topic], config: &Config) -> CollectedEpisode {
    let topics = [config.front_topic.as_str(), config.rear_topic.as_str()]
        .into_iter()
        .filter_map(|name| {
            available_topics.iter().find(|topic| topic.name == name).map(|topic| SourceTopicRun {
                name: topic.name.clone(),
                message_count: topic.message_count,
                retained_record_count: 0,
                retained_point_count: 0,
                frame_ids: Vec::new(),
                peak_source_bytes: 0,
            })
        })
        .collect();
    CollectedEpisode {
        episode: MemoryEpisode::default(),
        topics,
        selected_record_count: 0,
        selected_point_count: 0,
        selected_bytes: 0,
        deterministic_order_verified: false,
        frame_identity_match: false,
        peak_source_bytes: 0,
    }
}

fn record_signature(record: &StampedRecord) -> Result<RecordSignature, Box<dyn Error>> {
    Ok(RecordSignature {
        stamp_nanos: record.stamp.as_nanos(),
        topic: record.topic.as_str().to_owned(),
        frame_id: record.record.metadata().frame_id.0.clone(),
        point_count: u64::try_from(record.record.cloud().len())?,
    })
}

fn publish_episode(config: &Config, episode: &MemoryEpisode) -> Result<PublishRun, Box<dyn Error>> {
    let mut node = LoopbackRos2Node::new();
    let mut replayer = DeterministicReplayer::new(episode);
    let mut run = PublishRun::default();
    while let Some(stamped) = replayer.next_record() {
        let source_topic = stamped.topic.as_str().to_owned();
        let publish_topic = publish_topic_name(&config.publish_prefix, &source_topic);
        let message = point_cloud_message(stamped)?;
        let payload = encode_point_cloud2_xyz(&message)?;
        let payload_bytes = u64::try_from(payload.len())?;
        node.publish(publish_topic.clone(), payload);
        let received = node
            .take(&publish_topic)
            .ok_or_else(|| format!("loopback did not receive topic '{publish_topic}'"))?;
        let roundtrip_payload_bytes = u64::try_from(received.len())?;
        let decoded = decode_point_cloud2_xyz(&received)?;
        let roundtrip_verified = decoded == message;
        let packet = LivePublishPacket::try_new(
            u64::try_from(run.packets.len())?,
            source_topic,
            publish_topic,
            message.frame_id.clone(),
            stamped.stamp.as_nanos(),
            u64::try_from(message.point_count())?,
            payload_bytes,
            roundtrip_payload_bytes,
            roundtrip_verified,
        )?;
        run.host_encode_bytes = run
            .host_encode_bytes
            .checked_add(payload_bytes)
            .ok_or("host encode byte count overflow")?;
        run.host_decode_bytes = run
            .host_decode_bytes
            .checked_add(roundtrip_payload_bytes)
            .ok_or("host decode byte count overflow")?;
        run.packets.push(packet);
    }
    Ok(run)
}

fn point_cloud_message(stamped: &StampedRecord) -> Result<PointCloud2Xyz, Box<dyn Error>> {
    let cloud = stamped.record.cloud();
    let x = cloud.field("x")?.as_f32()?;
    let y = cloud.field("y")?.as_f32()?;
    let z = cloud.field("z")?.as_f32()?;
    if x.len() != y.len() || x.len() != z.len() {
        return Err("source XYZ fields have inconsistent lengths".into());
    }
    let mut xyz = Vec::with_capacity(x.len().checked_mul(3).ok_or("XYZ capacity overflow")?);
    for index in 0..x.len() {
        xyz.extend_from_slice(&[x[index], y[index], z[index]]);
    }
    let intensity = match cloud.field("intensity") {
        Ok(buffer) => Some(buffer.as_f32()?.to_vec()),
        Err(_) => None,
    };
    let timestamp = stamped.stamp.as_nanos();
    let stamp_sec = i32::try_from(timestamp / 1_000_000_000)
        .map_err(|_| "PointCloud2 timestamp seconds exceed ROS i32 range")?;
    let stamp_nanosec = u32::try_from(timestamp % 1_000_000_000)?;
    Ok(match intensity {
        Some(intensity) => PointCloud2Xyz::try_new_with_intensity(
            cloud.metadata().frame_id.0.clone(),
            stamp_sec,
            stamp_nanosec,
            xyz,
            intensity,
        )?,
        None => PointCloud2Xyz::try_new(
            cloud.metadata().frame_id.0.clone(),
            stamp_sec,
            stamp_nanosec,
            xyz,
        )?,
    })
}

fn build_topics(
    source_topics: &[SourceTopicRun],
    publish_topic_names: &BTreeMap<String, String>,
    packets: &[LivePublishPacket],
) -> Result<Vec<LivePublishTopic>, Box<dyn Error>> {
    let mut counts = BTreeMap::<String, (u64, u64)>::new();
    for packet in packets {
        let entry = counts.entry(packet.source_topic.clone()).or_default();
        entry.0 = entry.0.checked_add(1).ok_or("topic message count overflow")?;
        entry.1 = entry.1.checked_add(packet.point_count).ok_or("topic point count overflow")?;
    }
    source_topics
        .iter()
        .map(|topic| {
            let (messages, points) = counts.remove(&topic.name).unwrap_or_default();
            let publish_topic = publish_topic_names
                .get(&topic.name)
                .cloned()
                .ok_or_else(|| format!("missing publish topic mapping for '{}'", topic.name))?;
            Ok(LivePublishTopic::try_new(
                topic.name.clone(),
                publish_topic,
                POINT_CLOUD2_TYPE,
                topic.message_count,
                topic.retained_record_count,
                topic.retained_point_count,
                messages,
                points,
                topic.frame_ids.clone(),
            )?)
        })
        .collect()
}

fn topic_names(source_topics: &[SourceTopicRun], config: &Config) -> BTreeMap<String, String> {
    source_topics
        .iter()
        .map(|topic| (topic.name.clone(), publish_topic_name(&config.publish_prefix, &topic.name)))
        .collect()
}

fn publish_topic_name(prefix: &str, source_topic: &str) -> String {
    format!("{}{}", prefix.trim_end_matches('/'), source_topic)
}

fn relevant_message_count(topics: &[Rosbag2Topic], config: &Config) -> u64 {
    topics
        .iter()
        .filter(|topic| topic.name == config.front_topic || topic.name == config.rear_topic)
        .map(|topic| topic.message_count)
        .sum()
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
    if !source_bound {
        blockers
            .push("calibration readiness input identity does not match the canonical bag".into());
    }
    let registration_ready =
        readiness.get("registration_ready").and_then(Value::as_bool).unwrap_or(false);
    for blocker in string_array(readiness.get("blockers")) {
        push_blocker(&mut blockers, format!("readiness: {blocker}"));
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

fn render_dashboard(state: &LivePublishState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#06111c;--panel:#0d2134;--line:#24506f;--muted:#8ea9bd;--cyan:#5bdcff;--green:#63e7a5;--red:#ff7184;--amber:#ffd166}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#1c5271 0,#06111c 45%);color:#eef9ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.title{font-size:30px;font-weight:750;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:750;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(16,43,67,.96),rgba(7,20,34,.96));border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 14px 32px #0004}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:25px;font-weight:750;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:#10324d;border:1px solid #2d638a;border-radius:999px;padding:6px 10px;font-size:12px}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #183b55;padding:9px 0}.row:last-child{border-bottom:0}.danger{color:var(--red)}.warning{color:var(--amber)}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.trace{max-height:280px;overflow:auto}.trace .row{display:grid;grid-template-columns:42px 1fr 1fr auto}.bar{height:9px;background:#06111c;border-radius:5px;overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}}
</style></head><body><main><section class="top"><div><div class="eyebrow">SpatialRust / ROS 2 live publish bridge</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section><section class="grid">
<article class="panel"><h2>Publish readiness</h2><div id="ready" class="metric"></div><div id="readyDetail" class="small"></div></article><article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article><article class="panel"><h2>Packets</h2><div id="packets" class="metric"></div><div id="points" class="small"></div></article><article class="panel"><h2>Round trips</h2><div id="roundtrips" class="metric"></div><div id="transport" class="small"></div></article><article class="panel wide"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="small mono"></div><div id="path" class="small mono"></div></article><article class="panel wide"><h2>Topic bridge</h2><div id="topics" class="chips"></div></article><article class="panel wide"><h2>Deterministic timeline</h2><div id="timeline" class="small"></div><div class="bar"><div id="timelineBar" class="fill"></div></div></article><article class="panel wide"><h2>Packet trace</h2><div id="trace" class="trace"></div></article><article class="panel wide"><h2>Transport receipt</h2><div id="resources"></div></article><article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article><article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:280px;overflow:auto;white-space:pre-wrap"></pre></article></section></main>
<script id="live-publish-state" type="application/json">__STATE_JSON__</script><script>
const state=JSON.parse(document.getElementById('live-publish-state').textContent),q=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),fmtBytes=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB',esc=v=>String(v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
q('title').textContent=state.title;q('source').textContent=state.source.path;q('admission').textContent=state.publish_ready?'PUBLISH READY':'PUBLISH BLOCKED';q('admission').className='badge '+(state.publish_ready?'ok':'blocked');q('ready').textContent=state.publish_ready?'READY':'BLOCKED';q('ready').className='metric '+(state.publish_ready?'':'danger');q('readyDetail').textContent=state.summary.frame_identity_match?'source/frame checks passed':'source/frame admission unavailable';q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';q('packets').textContent=fmt(state.summary.published_message_count)+' / '+fmt(state.summary.selected_record_count);q('points').textContent=fmt(state.summary.published_point_count)+' points · '+state.time_basis;q('roundtrips').textContent=fmt(state.summary.received_message_count)+' / '+fmt(state.summary.published_message_count);q('transport').textContent=state.transport.adapter+' · '+fmtBytes(state.transport.host_encode_bytes)+' encoded';q('identity').textContent=state.source.identity_matches?'MATCH':'MISMATCH';q('identity').className='metric '+(state.source.identity_matches?'':'danger');q('sha').textContent=state.source.observed_sha256;q('path').textContent=state.source.path;q('topics').innerHTML=state.topics.map(t=>'<span class="chip">'+esc(t.source_topic)+' → '+esc(t.publish_topic)+' · '+fmt(t.published_message_count)+' packets</span>').join('')||'<span class="small">No admitted topics</span>';
const p=state.packets,start=p.length?p[0].stamp_nanos:0,end=p.length?p[p.length-1].stamp_nanos:start;q('timeline').textContent=p.length?((start/1e9).toFixed(6)+' → '+(end/1e9).toFixed(6)+' s · '+fmt(p.length)+' ordered packets'): 'unavailable';q('timelineBar').style.width=p.length?((p.length/Math.max(1,state.summary.selected_record_count))*100)+'%':'0%';q('trace').innerHTML=p.map(v=>'<div class="row"><span class="mono">#'+v.sequence+'</span><span>'+esc(v.source_topic)+'</span><span>'+esc(v.publish_topic)+'</span><span class="small">'+fmt(v.point_count)+' pts · '+fmtBytes(v.payload_bytes)+' · '+(v.roundtrip_verified?'✓':'blocked')+'</span></div>').join('')||'<div class="small">No packets emitted</div>';q('resources').innerHTML='<div class="row"><span>host encode</span><span class="mono">'+fmtBytes(state.transport.host_encode_bytes)+'</span></div><div class="row"><span>host decode</span><span class="mono">'+fmtBytes(state.transport.host_decode_bytes)+'</span></div><div class="row"><span>device upload/readback</span><span class="mono">'+fmtBytes(state.transport.device_upload_bytes)+' / '+fmtBytes(state.transport.device_readback_bytes)+'</span></div><div class="row"><span>queue</span><span class="mono">'+state.transport.queue_policy+'</span></div>';q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="ok">All gates passed</li>';q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
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
        || !config.output_dir.is_absolute()
        || !config.calibration_readiness.is_absolute()
    {
        return Err(
            "input, --output-dir, and --calibration-readiness paths must be absolute".into()
        );
    }
    if !config.publish_prefix.starts_with('/') || config.publish_prefix == "/" {
        return Err("--publish-prefix must be an absolute non-root topic prefix".into());
    }
    if config.front_topic == config.rear_topic {
        return Err("front and rear topics must differ".into());
    }
    if config.max_records_per_topic == 0 {
        return Err("--max-records must be greater than zero".into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut front_frame_id = None;
    let mut rear_frame_id = None;
    let mut calibration_readiness = None;
    let mut publish_prefix = DEFAULT_PUBLISH_PREFIX.to_owned();
    let mut front_topic = DEFAULT_FRONT_TOPIC.to_owned();
    let mut rear_topic = DEFAULT_REAR_TOPIC.to_owned();
    let mut max_records_per_topic = DEFAULT_MAX_RECORDS_PER_TOPIC;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--front-frame-id" => front_frame_id = Some(next_value(&mut args, &flag)?),
            "--rear-frame-id" => rear_frame_id = Some(next_value(&mut args, &flag)?),
            "--calibration-readiness" => {
                calibration_readiness = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--publish-prefix" => publish_prefix = next_value(&mut args, &flag)?,
            "--front-topic" => front_topic = next_value(&mut args, &flag)?,
            "--rear-topic" => rear_topic = next_value(&mut args, &flag)?,
            "--max-records" => max_records_per_topic = parse_u64(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        input: PathBuf::from(input),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        front_frame_id: front_frame_id.ok_or("--front-frame-id is required")?,
        rear_frame_id: rear_frame_id.ok_or("--rear-frame-id is required")?,
        calibration_readiness: calibration_readiness
            .ok_or("--calibration-readiness is required")?,
        publish_prefix,
        front_topic,
        rear_topic,
        max_records_per_topic,
        min_output_free_bytes,
    })
}

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, Box<dyn Error>> {
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
    String::from(
        "usage: rosbag2_live_publish <absolute-input.db3> \\\n  --output-dir <absolute-new-dir> \\\n  --expected-input-sha256 <sha256> \\\n  --front-frame-id <frame> \\\n  --rear-frame-id <frame> \\\n  --calibration-readiness <absolute-readiness.json> [options]\n\noptions:\n  --publish-prefix <absolute-topic-prefix>\n  --front-topic <topic>\n  --rear-topic <topic>\n  --max-records <count>\n  --min-output-free-bytes <bytes>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_live_publish_options() {
        let config = parse_args(
            [
                "/media/input.db3",
                "--output-dir",
                "/media/output",
                "--expected-input-sha256",
                SHA,
                "--front-frame-id",
                "lidar_front",
                "--rear-frame-id",
                "lidar_rear",
                "--calibration-readiness",
                "/media/readiness.json",
                "--publish-prefix",
                "/bridge",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.publish_prefix, "/bridge");
        assert_eq!(config.front_frame_id, "lidar_front");
        assert_eq!(config.rear_frame_id, "lidar_rear");
        assert_eq!(config.max_records_per_topic, DEFAULT_MAX_RECORDS_PER_TOPIC);
    }

    #[test]
    fn rejects_bad_hash() {
        assert!(validate_sha256("not-a-sha").is_err());
    }

    #[test]
    fn maps_source_topic_without_hidden_rewrite() {
        assert_eq!(
            publish_topic_name("/spatialrust/", "/lidar_front/points_raw"),
            "/spatialrust/lidar_front/points_raw"
        );
    }
}

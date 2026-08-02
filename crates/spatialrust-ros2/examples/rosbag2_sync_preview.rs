//! Build a bounded deterministic front/rear rosbag2 synchronization preview.

use std::{collections::BTreeSet, env, error::Error, fs, path::Path};

use serde::Serialize;
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
};
use spatialrust_ros2::Rosbag2PointCloudSource;
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, EpisodeLimits, MemoryEpisodeBuilder, StampedRecord,
    StampedTime, SyncWindow, TopicId,
};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.sync-preview.receipt";
const RECEIPT_VERSION: u32 = 1;
const CHUNK_POINTS: usize = 65_536;
const SOURCE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_POINTS: u64 = 2_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
struct Config {
    input: String,
    receipt: String,
    front_topic: String,
    rear_topic: String,
    max_records_per_topic: u64,
    max_delta_ns: u64,
}

#[derive(Serialize)]
struct PreviewReceipt {
    schema: &'static str,
    version: u32,
    input: String,
    front_topic: String,
    rear_topic: String,
    clock: &'static str,
    time_basis: &'static str,
    max_records_per_topic: u64,
    max_delta_ns: u64,
    chunk_points: usize,
    source_memory_budget_bytes: u64,
    episode_max_points: u64,
    episode_max_bytes: u64,
    retained_records: u64,
    retained_points: u64,
    retained_bytes: u64,
    matched_bundles: u64,
    max_matched_delta_ns: u64,
    topics: Vec<TopicReceipt>,
}

#[derive(Serialize)]
struct TopicReceipt {
    topic: String,
    schema: String,
    bag_message_count: u64,
    retained_chunks: u64,
    retained_points: u64,
    peak_source_bytes: u64,
    frame_ids: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-sync-preview: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let max_records =
        config.max_records_per_topic.checked_mul(2).ok_or("episode record limit overflow")?;
    let limits = EpisodeLimits::new(max_records, MAX_POINTS, MAX_BYTES);
    let mut builder = MemoryEpisodeBuilder::try_new(limits)?;

    let front = append_topic(
        &mut builder,
        Path::new(&config.input),
        &config.front_topic,
        config.max_records_per_topic,
    )?;
    let rear = append_topic(
        &mut builder,
        Path::new(&config.input),
        &config.rear_topic,
        config.max_records_per_topic,
    )?;

    let retained_records = u64::try_from(builder.len())?;
    let retained_points = builder.points();
    let retained_bytes = builder.bytes();
    let episode = builder.finish();
    let mut replayer = DeterministicReplayer::new(&episode);
    let front_id = TopicId::new(config.front_topic.clone());
    let rear_id = TopicId::new(config.rear_topic.clone());
    let window = SyncWindow { max_delta_ns: config.max_delta_ns, max_uncertainty_ns: 0 };
    let mut matched_bundles = 0_u64;
    let mut max_matched_delta_ns = 0_u64;
    while let Some(bundle) =
        replayer.next_bundle(&front_id, std::slice::from_ref(&rear_id), window)?
    {
        let front_record = bundle.get(&front_id).ok_or("front bundle member missing")?;
        let rear_record = bundle.get(&rear_id).ok_or("rear bundle member missing")?;
        let delta = front_record.stamp.abs_delta_ns(&rear_record.stamp);
        matched_bundles = matched_bundles.checked_add(1).ok_or("bundle counter overflow")?;
        max_matched_delta_ns = max_matched_delta_ns.max(delta);
    }

    let receipt = PreviewReceipt {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input: config.input.clone(),
        front_topic: config.front_topic,
        rear_topic: config.rear_topic,
        clock: "ros2-external",
        time_basis: "PointCloud2 header stamp; preview assumes one external ROS time domain; no clock calibration applied",
        max_records_per_topic: config.max_records_per_topic,
        max_delta_ns: config.max_delta_ns,
        chunk_points: CHUNK_POINTS,
        source_memory_budget_bytes: SOURCE_MEMORY_BYTES,
        episode_max_points: MAX_POINTS,
        episode_max_bytes: MAX_BYTES,
        retained_records,
        retained_points,
        retained_bytes,
        matched_bundles,
        max_matched_delta_ns,
        topics: vec![front, rear],
    };
    write_json(Path::new(&config.receipt), &receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn append_topic(
    builder: &mut MemoryEpisodeBuilder,
    input: &Path,
    topic: &str,
    max_records: u64,
) -> Result<TopicReceipt, Box<dyn Error>> {
    let options = StreamOptions::new(CHUNK_POINTS, MemoryBudget::new(SOURCE_MEMORY_BYTES)?)?;
    let mut source =
        Rosbag2PointCloudSource::open(input, topic, options, CancellationToken::default())?;
    let schema = source.schema().id.as_str().to_owned();
    let mut retained_chunks = 0_u64;
    let mut retained_points = 0_u64;
    let mut frame_ids = BTreeSet::new();
    while retained_chunks < max_records {
        let Some(chunk) = source.next_chunk() else {
            break;
        };
        let chunk = chunk?;
        let record = chunk.record().clone();
        frame_ids.insert(record.metadata().frame_id.0.clone());
        let stamp = StampedTime::exact("ros2", ClockDomain::External, record.metadata().timestamp);
        retained_points = retained_points
            .checked_add(u64::try_from(record.cloud().len())?)
            .ok_or("topic point counter overflow")?;
        builder.push(StampedRecord::new(topic, stamp, record))?;
        retained_chunks = retained_chunks.checked_add(1).ok_or("topic chunk counter overflow")?;
    }
    let peak_source_bytes = source.memory_tracker().snapshot().peak_bytes;
    Ok(TopicReceipt {
        topic: topic.to_owned(),
        schema,
        bag_message_count: source.topic().message_count,
        retained_chunks,
        retained_points,
        peak_source_bytes,
        frame_ids: frame_ids.into_iter().collect(),
    })
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    let receipt = args.next().ok_or_else(usage)?;
    let front_topic = args.next().ok_or_else(usage)?;
    let rear_topic = args.next().ok_or_else(usage)?;
    if front_topic == rear_topic {
        return Err("front and rear topics must differ".into());
    }
    let max_records_per_topic = args.next().map(|value| value.parse()).transpose()?.unwrap_or(8);
    let max_delta_ns = args.next().map(|value| value.parse()).transpose()?.unwrap_or(100_000_000);
    if max_records_per_topic == 0 {
        return Err("MAX_RECORDS_PER_TOPIC must be greater than zero".into());
    }
    if args.next().is_some() {
        return Err(usage().into());
    }
    Ok(Config { input, receipt, front_topic, rear_topic, max_records_per_topic, max_delta_ns })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn usage() -> String {
    "usage: rosbag2_sync_preview INPUT_DB3 RECEIPT_JSON FRONT_TOPIC REAR_TOPIC [MAX_RECORDS_PER_TOPIC] [MAX_DELTA_NS]".into()
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    #[test]
    fn parses_bounded_preview_options() {
        let config = parse_args(
            ["bag.db3", "preview.json", "/front", "/rear", "4", "5000000"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.max_records_per_topic, 4);
        assert_eq!(config.max_delta_ns, 5_000_000);
    }

    #[test]
    fn rejects_same_topics() {
        let error = parse_args(
            ["bag.db3", "preview.json", "/front", "/front"].into_iter().map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.to_string().contains("must differ"));
    }
}

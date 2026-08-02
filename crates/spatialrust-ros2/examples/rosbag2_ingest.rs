//! Inventory and batch-convert rosbag2 PointCloud2 topics.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::Serialize;
use spatialrust_io::{DatasetManifest, LasChunkSink, LasWriteFormat, ReceiptRole, StorageRoots};
use spatialrust_records::{
    BoundedSpatialRecordSink, BoundedSpatialRecordSource, CancellationToken, MemoryBudget,
    StreamOptions, DEFAULT_STREAM_CHUNK_POINTS, DEFAULT_STREAM_MEMORY_BUDGET_BYTES,
};
use spatialrust_ros2::{list_topics, Rosbag2PointCloudSource, Rosbag2Topic};
use spatialrust_runtime::POINT_CLOUD2_TYPE;

const INVENTORY_SCHEMA: &str = "spatialrust.rosbag2.inventory";
const BATCH_RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.ingest.receipt";
const RECEIPT_VERSION: u32 = 1;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output_dir: Option<PathBuf>,
    topics: Vec<String>,
    list_topics: bool,
    receipt: Option<PathBuf>,
    manifest: Option<PathBuf>,
    input_root: Option<PathBuf>,
    output_root: Option<PathBuf>,
    chunk_points: usize,
    memory_bytes: u64,
}

#[derive(Serialize)]
struct Inventory {
    schema: &'static str,
    version: u32,
    input: String,
    topics: Vec<InventoryTopic>,
}

#[derive(Serialize)]
struct InventoryTopic {
    id: i64,
    name: String,
    type_name: String,
    serialization_format: String,
    message_count: u64,
    supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Serialize)]
struct BatchReceipt {
    schema: &'static str,
    version: u32,
    input: String,
    output_dir: String,
    chunk_points: usize,
    memory_budget_bytes: u64,
    converted_topics: u64,
    skipped_topics: u64,
    failed_topics: u64,
    topics: Vec<BatchTopic>,
}

#[derive(Serialize)]
struct BatchTopic {
    id: i64,
    topic: String,
    type_name: String,
    serialization_format: String,
    message_count: u64,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    points: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_tracked_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Serialize)]
struct TopicReceipt {
    schema: String,
    version: u32,
    source_schema: String,
    input: String,
    topic_id: i64,
    topic: String,
    output: String,
    topic_messages: u64,
    chunks: u64,
    points: u64,
    max_message_bytes: u64,
    max_chunk_bytes: u64,
    peak_tracked_bytes: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-ingest: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let roots = StorageRoots::new(config.input_root.clone(), config.output_root.clone());
    let input = roots.resolve_input(&config.input)?;
    let input_text = input.display().to_string();
    let topics = list_topics(&input)?;
    let inventory = inventory(&input_text, &topics);

    if config.list_topics {
        let json = json_text(&inventory)?;
        if let Some(path) = config.receipt.map(|path| roots.resolve_output(path)).transpose()? {
            write_json(&path, &inventory)?;
            eprintln!("wrote inventory {}", path.display());
        }
        println!("{json}");
        return Ok(());
    }

    let output_dir = config
        .output_dir
        .as_ref()
        .ok_or("--output-dir is required unless --list-topics is used")?;
    let output_dir = roots.resolve_output(output_dir)?;
    fs::create_dir_all(&output_dir)?;
    let receipt_path = roots.resolve_output(
        config.receipt.unwrap_or_else(|| output_dir.join("rosbag2.ingest.receipt.json")),
    )?;
    let manifest_path = roots.resolve_output(
        config.manifest.unwrap_or_else(|| output_dir.join("rosbag2.ingest.manifest.json")),
    )?;
    let selected = select_topics(&topics, &config.topics)?;
    let explicit_selection = !config.topics.is_empty();
    let options = StreamOptions::new(config.chunk_points, MemoryBudget::new(config.memory_bytes)?)?;

    let mut results = Vec::with_capacity(selected.len());
    let mut converted_topics = 0_u64;
    let mut skipped_topics = 0_u64;
    let mut failed_topics = 0_u64;
    let mut successful_outputs = Vec::new();
    let mut successful_receipts = Vec::new();

    for topic in selected {
        if let Err(reason) = topic_support(topic) {
            let status = if explicit_selection {
                failed_topics = failed_topics.checked_add(1).ok_or("topic counter overflow")?;
                "failed"
            } else {
                skipped_topics = skipped_topics.checked_add(1).ok_or("topic counter overflow")?;
                "skipped"
            };
            results.push(BatchTopic {
                id: topic.id,
                topic: topic.name.clone(),
                type_name: topic.type_name.clone(),
                serialization_format: topic.serialization_format.clone(),
                message_count: topic.message_count,
                status,
                source_schema: None,
                output: None,
                receipt: None,
                chunks: None,
                points: None,
                peak_tracked_bytes: None,
                reason: Some(reason),
            });
            continue;
        }

        let output = output_path(&output_dir, topic);
        let topic_receipt_path = topic_receipt_path(&output);
        match convert_topic(&input, topic, &output, &topic_receipt_path, &options) {
            Ok(receipt) => {
                converted_topics =
                    converted_topics.checked_add(1).ok_or("topic counter overflow")?;
                successful_outputs.push(output.clone());
                successful_receipts.push(topic_receipt_path.clone());
                results.push(BatchTopic {
                    id: topic.id,
                    topic: topic.name.clone(),
                    type_name: topic.type_name.clone(),
                    serialization_format: topic.serialization_format.clone(),
                    message_count: topic.message_count,
                    status: "converted",
                    source_schema: Some(receipt.source_schema.clone()),
                    output: Some(output.display().to_string()),
                    receipt: Some(topic_receipt_path.display().to_string()),
                    chunks: Some(receipt.chunks),
                    points: Some(receipt.points),
                    peak_tracked_bytes: Some(receipt.peak_tracked_bytes),
                    reason: None,
                });
            }
            Err(error) => {
                failed_topics = failed_topics.checked_add(1).ok_or("topic counter overflow")?;
                results.push(BatchTopic {
                    id: topic.id,
                    topic: topic.name.clone(),
                    type_name: topic.type_name.clone(),
                    serialization_format: topic.serialization_format.clone(),
                    message_count: topic.message_count,
                    status: "failed",
                    source_schema: None,
                    output: Some(output.display().to_string()),
                    receipt: None,
                    chunks: None,
                    points: None,
                    peak_tracked_bytes: None,
                    reason: Some(error.to_string()),
                });
            }
        }
    }

    let receipt = BatchReceipt {
        schema: BATCH_RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input: input_text,
        output_dir: output_dir.display().to_string(),
        chunk_points: config.chunk_points,
        memory_budget_bytes: config.memory_bytes,
        converted_topics,
        skipped_topics,
        failed_topics,
        topics: results,
    };
    write_json(&receipt_path, &receipt)?;
    println!("{}", json_text(&receipt)?);

    if failed_topics != 0 {
        return Err("one or more selected topics failed; see the batch receipt".into());
    }
    if converted_topics == 0 {
        return Err("no supported PointCloud2 topics were converted".into());
    }

    let mut manifest = DatasetManifest::new();
    manifest.add_file(ReceiptRole::Input, &input)?;
    manifest.add_file(ReceiptRole::Auxiliary, &receipt_path)?;
    for path in successful_outputs {
        manifest.add_file(ReceiptRole::Output, path)?;
    }
    for path in successful_receipts {
        manifest.add_file(ReceiptRole::Auxiliary, path)?;
    }
    manifest.write_json(&manifest_path)?;
    eprintln!("wrote receipt {}", receipt_path.display());
    eprintln!("wrote manifest {}", manifest_path.display());
    Ok(())
}

fn inventory(input: &str, topics: &[Rosbag2Topic]) -> Inventory {
    Inventory {
        schema: INVENTORY_SCHEMA,
        version: RECEIPT_VERSION,
        input: input.to_owned(),
        topics: topics
            .iter()
            .map(|topic| {
                let support = topic_support(topic);
                InventoryTopic {
                    id: topic.id,
                    name: topic.name.clone(),
                    type_name: topic.type_name.clone(),
                    serialization_format: topic.serialization_format.clone(),
                    message_count: topic.message_count,
                    supported: support.is_ok(),
                    reason: support.err(),
                }
            })
            .collect(),
    }
}

fn topic_support(topic: &Rosbag2Topic) -> Result<(), String> {
    if topic.type_name != POINT_CLOUD2_TYPE {
        return Err(format!(
            "topic type `{}` is outside PointCloud2 ingest scope",
            topic.type_name
        ));
    }
    if !topic.serialization_format.eq_ignore_ascii_case("cdr") {
        return Err(format!(
            "serialization `{}` is unsupported; only CDR is supported",
            topic.serialization_format
        ));
    }
    Ok(())
}

fn select_topics<'a>(
    topics: &'a [Rosbag2Topic],
    requested: &[String],
) -> Result<Vec<&'a Rosbag2Topic>, Box<dyn Error>> {
    if requested.is_empty() {
        return Ok(topics.iter().collect());
    }

    let mut selected: Vec<&Rosbag2Topic> = Vec::with_capacity(requested.len());
    for name in requested {
        let topic = topics
            .iter()
            .find(|topic| topic.name == *name)
            .ok_or_else(|| format!("topic `{name}` was not found"))?;
        if selected.iter().any(|candidate| candidate.id == topic.id) {
            continue;
        }
        selected.push(topic);
    }
    Ok(selected)
}

fn convert_topic(
    input: &Path,
    topic: &Rosbag2Topic,
    output: &Path,
    receipt_path: &Path,
    options: &StreamOptions,
) -> Result<TopicReceipt, Box<dyn Error>> {
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let cancellation = CancellationToken::default();
    let mut source =
        Rosbag2PointCloudSource::open(input, &topic.name, options.clone(), cancellation)?;
    let source_schema = source.schema().id.0.clone();
    let mut sink =
        LasChunkSink::create_open_ended(output, source.schema().clone(), LasWriteFormat::Las)?;
    let mut chunks = 0_u64;
    let mut points = 0_u64;

    while let Some(chunk) = source.next_chunk() {
        let chunk = chunk?;
        points = points
            .checked_add(u64::try_from(chunk.record().cloud().len())?)
            .ok_or("point counter overflow")?;
        chunks = chunks.checked_add(1).ok_or("chunk counter overflow")?;
        sink.write_chunk(&chunk)?;
    }
    sink.finish()?;

    let snapshot = source.memory_tracker().snapshot();
    let receipt = TopicReceipt {
        schema: format!("spatialrust.rosbag2.pointcloud2.ingest.topic.receipt.v{RECEIPT_VERSION}"),
        version: RECEIPT_VERSION,
        source_schema,
        input: input.display().to_string(),
        topic_id: topic.id,
        topic: topic.name.clone(),
        output: output.display().to_string(),
        topic_messages: topic.message_count,
        chunks,
        points,
        max_message_bytes: source.max_message_bytes(),
        max_chunk_bytes: source.max_chunk_bytes(),
        peak_tracked_bytes: snapshot.peak_bytes,
    };
    write_json(receipt_path, &receipt)?;
    Ok(receipt)
}

fn output_path(output_dir: &Path, topic: &Rosbag2Topic) -> PathBuf {
    output_dir.join(format!("topic-{}-{}.las", topic.id, topic_slug(&topic.name)))
}

fn topic_receipt_path(output: &Path) -> PathBuf {
    output.with_extension("receipt.json")
}

fn topic_slug(name: &str) -> String {
    let mut slug = String::new();
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('_') {
            slug.push('_');
        }
    }
    let slug = slug.trim_matches('_');
    if slug.is_empty() {
        "topic".to_owned()
    } else {
        slug.to_owned()
    }
}

fn json_text<T: Serialize>(value: &T) -> Result<String, Box<dyn Error>> {
    Ok(serde_json::to_string_pretty(value)?)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", json_text(value)?))?;
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    if input == "-h" || input == "--help" {
        return Err(usage().into());
    }
    let mut config = Config {
        input: PathBuf::from(input),
        output_dir: None,
        topics: Vec::new(),
        list_topics: false,
        receipt: None,
        manifest: None,
        input_root: None,
        output_root: None,
        chunk_points: DEFAULT_STREAM_CHUNK_POINTS,
        memory_bytes: DEFAULT_STREAM_MEMORY_BUDGET_BYTES,
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--list-topics" => config.list_topics = true,
            "--output-dir" => {
                config.output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--topic" => config.topics.push(next_value(&mut args, &flag)?),
            "--receipt" => config.receipt = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--manifest" => config.manifest = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--input-root" => {
                config.input_root = Some(PathBuf::from(next_value(&mut args, &flag)?));
            }
            "--output-root" => {
                config.output_root = Some(PathBuf::from(next_value(&mut args, &flag)?));
            }
            "--chunk-points" => config.chunk_points = parse_one(&mut args, &flag)?,
            "--memory-budget" => config.memory_bytes = parse_one(&mut args, &flag)?,
            _ => return Err(format!("unknown option `{flag}`\n{}", usage()).into()),
        }
    }

    if config.list_topics && config.output_dir.is_some() {
        return Err("--list-topics cannot be combined with --output-dir".into());
    }
    if config.list_topics && config.manifest.is_some() {
        return Err("--manifest is only available for batch conversion".into());
    }
    Ok(config)
}

fn parse_one<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(next_value(args, flag)?.parse()?)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn usage() -> String {
    "usage: rosbag2_ingest INPUT_DB3 [--list-topics] [--output-dir DIR] \
     [--topic TOPIC]... [--receipt PATH] [--manifest PATH] \
     [--input-root DIR] [--output-root DIR] [--chunk-points N] [--memory-budget BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{output_path, parse_args, topic_receipt_path, topic_slug};
    use spatialrust_ros2::Rosbag2Topic;

    #[test]
    fn parses_inventory_and_batch_options() {
        let config = parse_args(
            [
                "bag.db3",
                "--output-dir",
                "runs",
                "--topic",
                "/lidar/front/points_raw",
                "--topic",
                "/lidar/rear/points_raw",
                "--input-root",
                "/media/input",
                "--output-root",
                "/media/output",
                "--chunk-points",
                "32",
                "--memory-budget",
                "4096",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(config.topics.len(), 2);
        assert_eq!(config.chunk_points, 32);
        assert_eq!(config.memory_bytes, 4096);
        assert_eq!(config.input_root.as_deref(), Some(std::path::Path::new("/media/input")));
    }

    #[test]
    fn rejects_output_dir_for_inventory_only() {
        let error = parse_args(
            ["bag.db3", "--list-topics", "--output-dir", "runs"].into_iter().map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cannot be combined"));
    }

    #[test]
    fn creates_collision_safe_topic_paths() {
        let topic = Rosbag2Topic {
            id: 7,
            name: "/lidar/front points_raw".into(),
            type_name: "sensor_msgs/msg/PointCloud2".into(),
            serialization_format: "cdr".into(),
            message_count: 1,
        };
        let output = output_path(std::path::Path::new("/media/output"), &topic);
        assert_eq!(
            output,
            std::path::PathBuf::from("/media/output/topic-7-lidar_front_points_raw.las")
        );
        assert_eq!(
            topic_receipt_path(&output),
            std::path::PathBuf::from("/media/output/topic-7-lidar_front_points_raw.receipt.json")
        );
    }

    #[test]
    fn normalizes_topic_slug() {
        assert_eq!(topic_slug("///LiDAR/front points_raw"), "lidar_front_points_raw");
        assert_eq!(topic_slug("///"), "topic");
    }
}

//! Decode a bounded rosbag2 TFMessage inventory with an input identity guard.
//!
//! The receipt is source-bound by the caller-provided input SHA-256. It records
//! frame edges without composing them and must not be reused for another bag
//! whose input identity or sensor frame names differ.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use spatialrust_io::{FileReceipt, ReceiptRole};
use spatialrust_ros2::{list_tf_messages, list_topics, Rosbag2TfMessage};
use spatialrust_runtime::TfTransform;

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.tf.inventory";
const RECEIPT_VERSION: u32 = 1;
const DEFAULT_TOPIC: &str = "/tf_static";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    expected_input_sha256: String,
    topic: String,
    max_messages: usize,
    required_frames: Vec<String>,
}

#[derive(Serialize)]
struct InventoryReceipt {
    schema: &'static str,
    version: u32,
    input: FileReceipt,
    source_identity: SourceIdentityReceipt,
    topic: TopicReceipt,
    messages: Vec<MessageReceipt>,
    observed_frames: Vec<String>,
    required_frames: Vec<String>,
    required_frames_present: bool,
    blockers: Vec<String>,
    passed: bool,
}

#[derive(Serialize)]
struct SourceIdentityReceipt {
    expected_input_sha256: String,
    observed_input_sha256: String,
    matches: bool,
}

#[derive(Serialize)]
struct TopicReceipt {
    requested_topic: String,
    present: bool,
    type_name: Option<String>,
    serialization_format: Option<String>,
    bag_message_count: u64,
    decoded_message_count: u64,
    truncated: bool,
}

#[derive(Serialize)]
struct MessageReceipt {
    bag_timestamp: u64,
    transforms: Vec<TransformReceipt>,
}

#[derive(Serialize)]
struct TransformReceipt {
    stamp_sec: i32,
    stamp_nanosec: u32,
    frame_id: String,
    child_frame_id: String,
    translation_m: [f64; 3],
    rotation_xyzw: [f64; 4],
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    if config.output.exists() {
        return Err(
            format!("TF inventory output '{}' already exists", config.output.display()).into()
        );
    }
    if !config.input.is_file() {
        return Err(format!("input bag '{}' is not a regular file", config.input.display()).into());
    }

    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let observed_input_sha256 = input.sha256.clone().ok_or("input checksum was not produced")?;
    let identity_matches = observed_input_sha256 == config.expected_input_sha256;
    let topics = list_topics(&config.input)?;
    let topic = topics.iter().find(|topic| topic.name == config.topic);
    let mut blockers = Vec::new();
    if !identity_matches {
        blockers.push(format!(
            "input SHA-256 mismatch: expected {}, observed {}",
            config.expected_input_sha256, observed_input_sha256
        ));
    }

    let mut messages = Vec::new();
    let (topic_receipt, observed_frames) = if !identity_matches {
        (
            TopicReceipt {
                requested_topic: config.topic.clone(),
                present: topic.is_some(),
                type_name: topic.map(|topic| topic.type_name.clone()),
                serialization_format: topic.map(|topic| topic.serialization_format.clone()),
                bag_message_count: topic.map_or(0, |topic| topic.message_count),
                decoded_message_count: 0,
                truncated: false,
            },
            BTreeSet::new(),
        )
    } else {
        let Some(topic) = topic else {
            blockers.push(format!("topic '{}' is missing", config.topic));
            return write_failed_receipt(
                &config,
                input,
                observed_input_sha256,
                identity_matches,
                TopicReceipt {
                    requested_topic: config.topic.clone(),
                    present: false,
                    type_name: None,
                    serialization_format: None,
                    bag_message_count: 0,
                    decoded_message_count: 0,
                    truncated: false,
                },
                messages,
                BTreeSet::new(),
                blockers,
            );
        };
        let decoded = list_tf_messages(&config.input, &config.topic, config.max_messages)?;
        if topic.message_count == 0 {
            blockers.push(format!("topic '{}' has no messages", config.topic));
        }
        let truncated = topic.message_count > u64::try_from(decoded.len()).unwrap_or(u64::MAX);
        if truncated {
            blockers.push(format!(
                "topic '{}' is truncated at {} of {} messages",
                config.topic,
                decoded.len(),
                topic.message_count
            ));
        }
        let mut frames = BTreeSet::new();
        messages = decoded.iter().map(message_receipt).collect();
        for message in &decoded {
            for transform in &message.transforms {
                frames.insert(transform.frame_id.clone());
                frames.insert(transform.child_frame_id.clone());
            }
        }
        (
            TopicReceipt {
                requested_topic: config.topic.clone(),
                present: true,
                type_name: Some(topic.type_name.clone()),
                serialization_format: Some(topic.serialization_format.clone()),
                bag_message_count: topic.message_count,
                decoded_message_count: u64::try_from(decoded.len())?,
                truncated,
            },
            frames,
        )
    };

    let required_frames_present =
        config.required_frames.iter().all(|frame| observed_frames.contains(frame));
    for frame in &config.required_frames {
        if !observed_frames.contains(frame) {
            blockers.push(format!("required frame '{}' was not observed", frame));
        }
    }
    let passed = blockers.is_empty();
    let receipt = InventoryReceipt {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input,
        source_identity: SourceIdentityReceipt {
            expected_input_sha256: config.expected_input_sha256,
            observed_input_sha256,
            matches: identity_matches,
        },
        topic: topic_receipt,
        messages,
        observed_frames: observed_frames.into_iter().collect(),
        required_frames: config.required_frames,
        required_frames_present,
        blockers,
        passed,
    };
    write_receipt(&config.output, &receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !receipt.passed {
        return Err("TF inventory failed its source or frame admission checks".into());
    }
    Ok(())
}

fn write_failed_receipt(
    config: &Config,
    input: FileReceipt,
    observed_input_sha256: String,
    identity_matches: bool,
    topic: TopicReceipt,
    messages: Vec<MessageReceipt>,
    observed_frames: BTreeSet<String>,
    blockers: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let required_frames_present =
        config.required_frames.iter().all(|frame| observed_frames.contains(frame));
    let mut blockers = blockers;
    for frame in &config.required_frames {
        if !observed_frames.contains(frame) {
            blockers.push(format!("required frame '{}' was not observed", frame));
        }
    }
    let receipt = InventoryReceipt {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input,
        source_identity: SourceIdentityReceipt {
            expected_input_sha256: config.expected_input_sha256.clone(),
            observed_input_sha256,
            matches: identity_matches,
        },
        topic,
        messages,
        observed_frames: observed_frames.into_iter().collect(),
        required_frames: config.required_frames.clone(),
        required_frames_present,
        blockers,
        passed: false,
    };
    write_receipt(&config.output, &receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Err("TF inventory failed its source or frame admission checks".into())
}

fn message_receipt(message: &Rosbag2TfMessage) -> MessageReceipt {
    MessageReceipt {
        bag_timestamp: message.bag_timestamp,
        transforms: message.transforms.iter().map(transform_receipt).collect(),
    }
}

fn transform_receipt(transform: &TfTransform) -> TransformReceipt {
    TransformReceipt {
        stamp_sec: transform.stamp_sec,
        stamp_nanosec: transform.stamp_nanosec,
        frame_id: transform.frame_id.clone(),
        child_frame_id: transform.child_frame_id.clone(),
        translation_m: transform.translation,
        rotation_xyzw: transform.rotation_xyzw,
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut output = None;
    let mut expected_input_sha256 = None;
    let mut topic = DEFAULT_TOPIC.to_owned();
    let mut max_messages = 1_usize;
    let mut required_frames = Vec::new();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output" => output = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => {
                expected_input_sha256 = Some(validate_sha256(&next_value(&mut args, &flag)?)?)
            }
            "--topic" => topic = next_value(&mut args, &flag)?,
            "--max-messages" => {
                max_messages = next_value(&mut args, &flag)?.parse::<usize>()?;
                if max_messages == 0 {
                    return Err("--max-messages must be greater than zero".into());
                }
            }
            "--require-frame" => required_frames.push(next_value(&mut args, &flag)?),
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    let output = output.ok_or_else(|| "--output is required".to_owned())?;
    let expected_input_sha256 =
        expected_input_sha256.ok_or_else(|| "--expected-input-sha256 is required".to_owned())?;
    if !input.is_absolute() || !output.is_absolute() {
        return Err("input and --output paths must be absolute".into());
    }
    if topic.is_empty() {
        return Err("--topic must not be empty".into());
    }
    if required_frames.iter().any(|frame| frame.is_empty()) {
        return Err("--require-frame values must not be empty".into());
    }
    Ok(Config { input, output, expected_input_sha256, topic, max_messages, required_frames })
}

fn validate_sha256(value: &str) -> Result<String, Box<dyn Error>> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("SHA-256 must be exactly 64 hexadecimal characters".into());
    }
    Ok(value.to_ascii_lowercase())
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn write_receipt(path: &Path, receipt: &InventoryReceipt) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string_pretty(receipt)?;
    if path.exists() {
        return Err(format!("TF inventory output '{}' already exists", path.display()).into());
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("tf.inventory.json")
    ));
    fs::write(&temporary, format!("{json}\n"))?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn usage() -> String {
    "usage: rosbag2_tf_inventory INPUT_DB3 --output ABSOLUTE_RECEIPT_JSON \
     --expected-input-sha256 SHA256 [--topic TOPIC] [--max-messages N] \
     [--require-frame FRAME]..."
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, validate_sha256};

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_source_bound_inventory_options() {
        let config = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/output/tf.json",
                "--expected-input-sha256",
                HASH,
                "--topic",
                "/tf_static",
                "--max-messages",
                "2",
                "--require-frame",
                "base_link",
                "--require-frame",
                "lidar_front",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.max_messages, 2);
        assert_eq!(config.required_frames, ["base_link", "lidar_front"]);
    }

    #[test]
    fn rejects_missing_identity_and_relative_paths() {
        let missing = parse_args(
            ["/media/input/bag.db3", "--output", "/media/output/tf.json"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(missing.to_string().contains("expected-input-sha256"));

        let relative = parse_args(
            ["bag.db3", "--output", "/media/output/tf.json", "--expected-input-sha256", HASH]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(relative.to_string().contains("absolute"));
    }

    #[test]
    fn validates_sha256_shape_and_normalizes_case() {
        assert_eq!(validate_sha256(&HASH.to_ascii_uppercase()).unwrap(), HASH);
        assert!(validate_sha256("not-a-sha256").is_err());
    }
}

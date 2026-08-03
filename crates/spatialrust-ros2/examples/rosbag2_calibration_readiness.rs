//! Produce a fail-closed calibration/frame readiness receipt for one rosbag2 bag.
//!
//! This is an inventory gate, not a calibration solver. It records the bag's
//! relevant ROS topics and optionally registers caller-supplied clock/frame
//! artifacts by path, size, and SHA-256. Artifact contents are intentionally
//! opaque until their source format is explicitly registered; this example
//! never invents a transform or applies a clock correction.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use spatialrust_io::{FileReceipt, ReceiptRole};
use spatialrust_ros2::{list_topics, Rosbag2Topic};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.calibration-readiness";
const RECEIPT_VERSION: u32 = 1;
const POINT_CLOUD2_TYPE: &str = "sensor_msgs/msg/PointCloud2";
const REQUIRED_ROS_TOPICS: [&str; 4] = ["/clock", "/tf", "/tf_static", "/odom"];
const DEFAULT_FRONT_TOPIC: &str = "/lidar_front/points_raw";
const DEFAULT_REAR_TOPIC: &str = "/lidar_rear/points_raw";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    front_topic: String,
    rear_topic: String,
    clock_artifact: Option<PathBuf>,
    frame_artifact: Option<PathBuf>,
}

#[derive(Serialize)]
struct ReadinessReceipt {
    schema: &'static str,
    version: u32,
    input: FileReceipt,
    sensor_topics: Vec<SensorTopicReceipt>,
    relevant_topics: Vec<RelevantTopicReceipt>,
    calibration_artifacts: CalibrationArtifactReceipt,
    artifact_policy: &'static str,
    registration_ready: bool,
    blockers: Vec<String>,
}

#[derive(Serialize)]
struct SensorTopicReceipt {
    role: &'static str,
    requested_topic: String,
    present: bool,
    supported_pointcloud2: bool,
    message_count: u64,
    type_name: Option<String>,
    serialization_format: Option<String>,
}

#[derive(Serialize)]
struct RelevantTopicReceipt {
    topic: &'static str,
    present: bool,
    observed_types: Vec<String>,
    message_count: u64,
}

#[derive(Serialize)]
struct CalibrationArtifactReceipt {
    clock: ArtifactReceipt,
    frame: ArtifactReceipt,
}

#[derive(Serialize)]
struct ArtifactReceipt {
    kind: &'static str,
    status: &'static str,
    path: Option<PathBuf>,
    file: Option<FileReceipt>,
    reason: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    if config.output.exists() {
        return Err(format!("readiness output '{}' already exists", config.output.display()).into());
    }
    if !config.input.is_file() {
        return Err(format!("input bag '{}' is not a regular file", config.input.display()).into());
    }

    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let topics = list_topics(&config.input)?;
    let mut sorted_topics = topics;
    sorted_topics.sort_by(|left, right| left.name.cmp(&right.name).then(left.id.cmp(&right.id)));

    let sensor_topics = vec![
        sensor_topic_receipt("front", &config.front_topic, &sorted_topics),
        sensor_topic_receipt("rear", &config.rear_topic, &sorted_topics),
    ];
    let relevant_topics = REQUIRED_ROS_TOPICS
        .into_iter()
        .map(|topic| relevant_topic_receipt(topic, &sorted_topics))
        .collect::<Vec<_>>();
    let calibration_artifacts = CalibrationArtifactReceipt {
        clock: artifact_receipt("clock_calibration", config.clock_artifact.as_deref()),
        frame: artifact_receipt("frame_calibration", config.frame_artifact.as_deref()),
    };

    let mut blockers = Vec::new();
    for sensor in &sensor_topics {
        if !sensor.present {
            blockers.push(format!(
                "{} sensor topic '{}' is missing",
                sensor.role, sensor.requested_topic
            ));
        } else if !sensor.supported_pointcloud2 {
            blockers.push(format!(
                "{} sensor topic '{}' is not a CDR sensor_msgs/msg/PointCloud2 topic",
                sensor.role, sensor.requested_topic
            ));
        }
    }
    if calibration_artifacts.clock.status != "registered" {
        blockers
            .push(format!("clock calibration artifact is {}", calibration_artifacts.clock.status));
    }
    if calibration_artifacts.frame.status != "registered" {
        blockers
            .push(format!("frame calibration artifact is {}", calibration_artifacts.frame.status));
    }

    let receipt = ReadinessReceipt {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input,
        sensor_topics,
        relevant_topics,
        calibration_artifacts,
        artifact_policy: "Only absolute regular non-empty files are registered by path, size, and SHA-256; artifact contents are not interpreted and no clock/frame transform is applied",
        registration_ready: blockers.is_empty(),
        blockers,
    };
    let json = serde_json::to_string_pretty(&receipt)?;
    write_json_atomically(&config.output, &json)?;
    println!("{json}");
    if !receipt.registration_ready {
        return Err("calibration readiness gate failed; see the external receipt blockers".into());
    }
    Ok(())
}

fn sensor_topic_receipt(
    role: &'static str,
    requested_topic: &str,
    topics: &[Rosbag2Topic],
) -> SensorTopicReceipt {
    let topic = topics.iter().find(|topic| topic.name == requested_topic);
    SensorTopicReceipt {
        role,
        requested_topic: requested_topic.to_owned(),
        present: topic.is_some(),
        supported_pointcloud2: topic.is_some_and(|topic| {
            topic.type_name == POINT_CLOUD2_TYPE
                && topic.serialization_format.eq_ignore_ascii_case("cdr")
        }),
        message_count: topic.map_or(0, |topic| topic.message_count),
        type_name: topic.map(|topic| topic.type_name.clone()),
        serialization_format: topic.map(|topic| topic.serialization_format.clone()),
    }
}

fn relevant_topic_receipt(
    topic_name: &'static str,
    topics: &[Rosbag2Topic],
) -> RelevantTopicReceipt {
    let matches = topics.iter().filter(|topic| topic.name == topic_name).collect::<Vec<_>>();
    let observed_types = matches
        .iter()
        .map(|topic| topic.type_name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    RelevantTopicReceipt {
        topic: topic_name,
        present: !matches.is_empty(),
        observed_types,
        message_count: matches.iter().map(|topic| topic.message_count).sum(),
    }
}

fn artifact_receipt(kind: &'static str, path: Option<&Path>) -> ArtifactReceipt {
    let Some(path) = path else {
        return ArtifactReceipt {
            kind,
            status: "not_registered",
            path: None,
            file: None,
            reason: Some("no artifact path was supplied".into()),
        };
    };
    let path = path.to_path_buf();
    match fs::metadata(&path) {
        Ok(metadata) if !metadata.is_file() => ArtifactReceipt {
            kind,
            status: "invalid",
            path: Some(path),
            file: None,
            reason: Some("artifact path is not a regular file".into()),
        },
        Ok(metadata) if metadata.len() == 0 => ArtifactReceipt {
            kind,
            status: "invalid",
            path: Some(path),
            file: None,
            reason: Some("artifact file is empty".into()),
        },
        Ok(_) => match FileReceipt::from_path(ReceiptRole::Auxiliary, &path) {
            Ok(file) => ArtifactReceipt {
                kind,
                status: "registered",
                path: Some(path),
                file: Some(file),
                reason: None,
            },
            Err(error) => ArtifactReceipt {
                kind,
                status: "invalid",
                path: Some(path),
                file: None,
                reason: Some(format!("artifact could not be hashed: {error}")),
            },
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ArtifactReceipt {
            kind,
            status: "missing",
            path: Some(path),
            file: None,
            reason: Some("artifact file does not exist".into()),
        },
        Err(error) => ArtifactReceipt {
            kind,
            status: "invalid",
            path: Some(path),
            file: None,
            reason: Some(format!("artifact metadata could not be read: {error}")),
        },
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut output = None;
    let mut front_topic = DEFAULT_FRONT_TOPIC.to_owned();
    let mut rear_topic = DEFAULT_REAR_TOPIC.to_owned();
    let mut clock_artifact = None;
    let mut frame_artifact = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output" => output = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--front-topic" => front_topic = next_value(&mut args, &flag)?,
            "--rear-topic" => rear_topic = next_value(&mut args, &flag)?,
            "--clock-artifact" => {
                clock_artifact = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--frame-artifact" => {
                frame_artifact = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }

    let output = output.ok_or_else(|| "--output is required".to_owned())?;
    if !input.is_absolute() || !output.is_absolute() {
        return Err("input and --output paths must be absolute".into());
    }
    if front_topic == rear_topic {
        return Err("front and rear topics must differ".into());
    }
    for (kind, path) in [("clock", clock_artifact.as_ref()), ("frame", frame_artifact.as_ref())] {
        if path.is_some_and(|path| !path.is_absolute()) {
            return Err(format!("{kind} artifact path must be absolute").into());
        }
    }
    Ok(Config { input, output, front_topic, rear_topic, clock_artifact, frame_artifact })
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn write_json_atomically(path: &Path, json: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("readiness output '{}' already exists", path.display()).into());
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("readiness.json")
    ));
    fs::write(&temporary, format!("{json}\n"))?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn usage() -> String {
    "usage: rosbag2_calibration_readiness INPUT_DB3 --output ABSOLUTE_RECEIPT_JSON \
     [--front-topic TOPIC] [--rear-topic TOPIC] [--clock-artifact ABSOLUTE_FILE] \
     [--frame-artifact ABSOLUTE_FILE]"
        .into()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{artifact_receipt, parse_args, relevant_topic_receipt, Rosbag2Topic};

    #[test]
    fn parses_absolute_paths_and_defaults() {
        let config = parse_args(
            ["/media/input/bag.db3", "--output", "/media/output/readiness.json"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.input, PathBuf::from("/media/input/bag.db3"));
        assert_eq!(config.front_topic, "/lidar_front/points_raw");
        assert_eq!(config.rear_topic, "/lidar_rear/points_raw");
        assert!(config.clock_artifact.is_none());
    }

    #[test]
    fn rejects_relative_paths_and_duplicate_sensor_topics() {
        let relative = parse_args(
            ["bag.db3", "--output", "/media/output/readiness.json"].into_iter().map(str::to_owned),
        )
        .unwrap_err();
        assert!(relative.to_string().contains("absolute"));

        let duplicate = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/output/readiness.json",
                "--front-topic",
                "/same",
                "--rear-topic",
                "/same",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("must differ"));
    }

    #[test]
    fn reports_missing_topic_and_unregistered_artifact() {
        let topics = vec![Rosbag2Topic {
            id: 1,
            name: "/clock".into(),
            type_name: "rosgraph_msgs/msg/Clock".into(),
            serialization_format: "cdr".into(),
            message_count: 3,
        }];
        let clock = relevant_topic_receipt("/clock", &topics);
        assert!(clock.present);
        assert_eq!(clock.message_count, 3);
        let artifact = artifact_receipt("clock_calibration", None);
        assert_eq!(artifact.status, "not_registered");
    }

    #[test]
    fn registers_non_empty_artifact_with_checksum_and_rejects_empty_file() {
        let directory = tempfile::tempdir().unwrap();
        let populated = directory.path().join("clock.json");
        std::fs::write(&populated, b"clock calibration").unwrap();
        let registered = artifact_receipt("clock_calibration", Some(&populated));
        assert_eq!(registered.status, "registered");
        assert_eq!(registered.file.as_ref().unwrap().size_bytes, Some(17));
        assert!(registered.file.as_ref().unwrap().sha256.is_some());

        let empty = directory.path().join("frame.json");
        std::fs::write(&empty, []).unwrap();
        let invalid = artifact_receipt("frame_calibration", Some(&empty));
        assert_eq!(invalid.status, "invalid");
        assert_eq!(invalid.reason.as_deref(), Some("artifact file is empty"));
    }
}

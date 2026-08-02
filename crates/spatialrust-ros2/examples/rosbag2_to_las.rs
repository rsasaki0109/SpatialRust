//! Convert one rosbag2 PointCloud2 topic to a bounded LAS stream.

use std::{env, error::Error, fs, path::Path};

use serde::Serialize;
use spatialrust_io::{DatasetManifest, LasChunkSink, LasWriteFormat, ReceiptRole};
use spatialrust_records::{
    BoundedSpatialRecordSink, BoundedSpatialRecordSource, CancellationToken, MemoryBudget,
    StreamOptions,
};
use spatialrust_ros2::Rosbag2PointCloudSource;

#[derive(Serialize)]
struct Receipt {
    schema: &'static str,
    input: String,
    topic: String,
    output: String,
    topic_messages: u64,
    chunks: u64,
    points: u64,
    max_message_bytes: u64,
    max_chunk_bytes: u64,
    peak_tracked_bytes: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: {} INPUT_DB3 TOPIC OUTPUT_LAS RECEIPT_JSON MANIFEST_JSON", args[0]);
        return Err("expected five arguments".into());
    }
    let input = Path::new(&args[1]);
    let topic = &args[2];
    let output = Path::new(&args[3]);
    let receipt_path = Path::new(&args[4]);
    let manifest_path = Path::new(&args[5]);

    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let options = StreamOptions::new(16_384, MemoryBudget::new(256 * 1024 * 1024)?)?;
    let cancellation = CancellationToken::default();
    let mut source = Rosbag2PointCloudSource::open(input, topic, options, cancellation)?;
    let schema = source.schema().clone();
    let mut sink = LasChunkSink::create_open_ended(output, schema, LasWriteFormat::Las)?;

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
    let receipt = Receipt {
        schema: "spatialrust.rosbag2.pointcloud2.xyz.receipt",
        input: input.display().to_string(),
        topic: topic.clone(),
        output: output.display().to_string(),
        topic_messages: source.topic().message_count,
        chunks,
        points,
        max_message_bytes: source.max_message_bytes(),
        max_chunk_bytes: source.max_chunk_bytes(),
        peak_tracked_bytes: snapshot.peak_bytes,
    };

    if let Some(parent) = receipt_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(receipt_path, format!("{}\n", serde_json::to_string_pretty(&receipt)?))?;

    let mut manifest = DatasetManifest::new();
    manifest.add_file(ReceiptRole::Input, input)?;
    manifest.add_file(ReceiptRole::Output, output)?;
    manifest.write_json(manifest_path)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

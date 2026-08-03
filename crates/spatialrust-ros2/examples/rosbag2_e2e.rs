//! Run a bounded, receipt-producing rosbag2-to-Viewer/glTF E2E smoke.
//!
//! This example deliberately keeps the episode small. It exercises the public
//! contracts in order, while leaving the canonical bag and all derived data on
//! the configured external result filesystem.

use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use spatialrust_interchange::{export_triangle_mesh_gltf_json, import_triangle_mesh_gltf_json};
use spatialrust_io::{
    DatasetManifest, ReceiptRole, StoragePreflight, StorageRoots, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_mapping::{IcpScanMatcher, ScanOdometry, ScanOdometryConfig};
use spatialrust_math::Vec3;
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
    DEFAULT_STREAM_MEMORY_BUDGET_BYTES,
};
use spatialrust_registration::IcpConfig;
use spatialrust_ros2::Rosbag2PointCloudSource;
use spatialrust_scene::TsdfVolume;
use spatialrust_semantic::{OpenVocabLabel, SpatialRecordEntity};
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, EpisodeLimits, MemoryEpisode, MemoryEpisodeBuilder,
    StampedRecord, StampedTime, SyncWindow, TopicId,
};
use spatialrust_viewer::{mesh_visual, spatial_record_entity_visual};
use spatialrust_viz::{LayerId, LinearRgba, VisualScene};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const RECEIPT_VERSION: u32 = 1;
const CHECKPOINT_SCHEMA: &str = "spatialrust.rosbag2.e2e.checkpoint";
const CHECKPOINT_VERSION: u32 = 1;
const DEFAULT_CHUNK_POINTS: usize = 65_536;
const DEFAULT_MAX_RECORDS_PER_TOPIC: u64 = 2;
const DEFAULT_MAX_DELTA_NS: u64 = 100_000_000;
const DEFAULT_EPISODE_MAX_POINTS: u64 = 2_000_000;
const DEFAULT_EPISODE_MAX_BYTES: u64 = 256 * 1024 * 1024;
const TSDF_ORIGIN: Vec3<f32> = Vec3::new(-50.0, -50.0, -10.0);
const TSDF_VOXEL_SIZE: f32 = 0.5;
const TSDF_DIMS: [usize; 3] = [200, 200, 80];
const TSDF_TRUNCATION: f32 = 1.0;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output_root: PathBuf,
    output_dir: PathBuf,
    front_topic: String,
    rear_topic: String,
    max_records_per_topic: u64,
    max_delta_ns: u64,
    chunk_points: usize,
    source_memory_bytes: u64,
    min_output_free_bytes: u64,
    verify_manifest: bool,
    resume: bool,
}

#[derive(Serialize)]
struct E2eReceipt {
    schema: &'static str,
    version: u32,
    input: String,
    output_dir: String,
    front_topic: String,
    rear_topic: String,
    storage: StorageReceipt,
    ingest: IngestReceipt,
    sync: SyncReceipt,
    odometry: OdometryReceipt,
    tsdf: TsdfReceipt,
    semantic: SemanticReceipt,
    viewer: ViewerReceipt,
    interchange: InterchangeReceipt,
    checkpoint: CheckpointReceipt,
    manifest_verified: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckpointStage {
    Created,
    Ingested,
    Synchronized,
    Odometry,
    Tsdf,
    Interchange,
    Viewer,
    Receipt,
    ManifestVerified,
    Complete,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct RunCheckpoint {
    schema: String,
    version: u32,
    input: String,
    output_dir: String,
    stage: CheckpointStage,
    artifacts: Vec<String>,
    temporary_files_removed: u64,
}

#[derive(Serialize)]
struct CheckpointReceipt {
    path: String,
    stage: &'static str,
    resume_mode: &'static str,
    temporary_files_removed: u64,
}

#[derive(Serialize)]
struct StorageReceipt {
    root: String,
    required_free_bytes: u64,
    available_before_bytes: u64,
    available_after_pipeline_bytes: u64,
}

#[derive(Serialize)]
struct IngestReceipt {
    chunk_points: usize,
    source_memory_budget_bytes: u64,
    episode_max_points: u64,
    episode_max_bytes: u64,
    retained_records: u64,
    retained_points: u64,
    retained_bytes: u64,
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

#[derive(Serialize)]
struct SyncReceipt {
    clock: &'static str,
    time_basis: &'static str,
    max_records_per_topic: u64,
    max_delta_ns: u64,
    matched_bundles: u64,
    max_matched_delta_ns: u64,
}

#[derive(Serialize)]
struct OdometryReceipt {
    topic: String,
    frame_id: String,
    scans: u64,
    motions: u64,
    truncated: bool,
    matcher: &'static str,
    pose_graph_nodes: u64,
    pose_graph_edges: u64,
}

#[derive(Serialize)]
struct TsdfReceipt {
    frame_id: String,
    origin: [f32; 3],
    voxel_size: f32,
    dims: [usize; 3],
    truncation: f32,
    integrated_records: u64,
    integrated_points: u64,
    mesh_vertices: u64,
    mesh_triangles: u64,
}

#[derive(Serialize)]
struct SemanticReceipt {
    entities: u64,
    visible_entities: u64,
    frame_ids: Vec<String>,
    runtime: &'static str,
}

#[derive(Serialize)]
struct ViewerReceipt {
    layers: u64,
    device_upload_bytes: u64,
    mesh: ViewerLayerReceipt,
    semantic: ViewerLayerReceipt,
}

#[derive(Serialize)]
struct ViewerLayerReceipt {
    id: String,
    source_count: u64,
    output_count: u64,
    generated_bytes: u64,
}

#[derive(Serialize)]
struct InterchangeReceipt {
    format: &'static str,
    path: String,
    bytes: u64,
    vertices: u64,
    indices: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-e2e: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let roots = StorageRoots::new(None, Some(config.output_root.clone()));
    let before = roots.preflight_output(config.min_output_free_bytes)?;
    let input = config.input.clone();
    let output_dir = roots.resolve_output(&config.output_dir)?;
    let checkpoint_path = output_dir.join("rosbag2.e2e.checkpoint.json");
    let output_exists = output_dir.exists();
    let temporary_files_removed =
        if output_exists && config.resume { remove_checkpoint_temp(&checkpoint_path)? } else { 0 };
    if output_exists {
        if !config.resume {
            return Err(format!(
                "run output directory '{}' already exists; choose a new run id or use --resume",
                output_dir.display()
            )
            .into());
        }
        let checkpoint = read_checkpoint(&checkpoint_path)?;
        validate_checkpoint(&checkpoint, &config.input, &output_dir, &checkpoint_path)?;
        if !matches!(checkpoint.stage, CheckpointStage::Complete) {
            return Err(format!(
                "run '{}' is incomplete at stage {}; it was retained and will not be overwritten",
                output_dir.display(),
                checkpoint_stage_label(&checkpoint.stage)
            )
            .into());
        }
        let manifest_path = output_dir.join("rosbag2.e2e.manifest.json");
        let manifest = DatasetManifest::read_json(&manifest_path)?;
        let validation = manifest.validate_local_files()?;
        eprintln!(
            "resumed complete E2E run: local_files={} uri_entries={} total_bytes={} temporary_files_removed={}",
            validation.checked_local_files,
            validation.uri_entries,
            validation.total_bytes,
            temporary_files_removed
        );
        return Ok(());
    }
    if config.resume {
        return Err(format!(
            "--resume requires an existing run directory '{}', but it was not found",
            output_dir.display()
        )
        .into());
    }
    fs::create_dir_all(&output_dir)?;
    let mut checkpoint = RunCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        version: CHECKPOINT_VERSION,
        input: config.input.display().to_string(),
        output_dir: output_dir.display().to_string(),
        stage: CheckpointStage::Created,
        artifacts: Vec::new(),
        temporary_files_removed,
    };
    write_checkpoint(&checkpoint_path, &checkpoint)?;

    let max_records =
        config.max_records_per_topic.checked_mul(2).ok_or("episode record limit overflow")?;
    let limits =
        EpisodeLimits::new(max_records, DEFAULT_EPISODE_MAX_POINTS, DEFAULT_EPISODE_MAX_BYTES);
    let mut builder = MemoryEpisodeBuilder::try_new(limits)?;
    let front = append_topic(&mut builder, &input, &config, &config.front_topic)?;
    let rear = append_topic(&mut builder, &input, &config, &config.rear_topic)?;
    let retained_records = u64::try_from(builder.len())?;
    let retained_points = builder.points();
    let retained_bytes = builder.bytes();
    let episode = builder.finish();
    advance_checkpoint(&mut checkpoint, &checkpoint_path, CheckpointStage::Ingested, &[])?;

    let front_id = TopicId::new(config.front_topic.clone());
    let rear_id = TopicId::new(config.rear_topic.clone());
    let window = SyncWindow { max_delta_ns: config.max_delta_ns, max_uncertainty_ns: 0 };
    let mut replayer = DeterministicReplayer::new(&episode);
    let mut matched_front_stamps = BTreeSet::new();
    let mut matched_bundles = 0_u64;
    let mut max_matched_delta_ns = 0_u64;
    while let Some(bundle) =
        replayer.next_bundle(&front_id, std::slice::from_ref(&rear_id), window)?
    {
        let front_record = bundle.get(&front_id).ok_or("front bundle member missing")?;
        let rear_record = bundle.get(&rear_id).ok_or("rear bundle member missing")?;
        matched_front_stamps.insert(front_record.stamp.as_nanos());
        matched_bundles = matched_bundles.checked_add(1).ok_or("bundle counter overflow")?;
        max_matched_delta_ns =
            max_matched_delta_ns.max(front_record.stamp.abs_delta_ns(&rear_record.stamp));
    }
    if matched_bundles == 0 {
        return Err("no front/rear bundles matched within the configured window".into());
    }
    advance_checkpoint(&mut checkpoint, &checkpoint_path, CheckpointStage::Synchronized, &[])?;

    let front_records: Vec<StampedRecord> = episode
        .records()
        .iter()
        .filter(|record| {
            record.topic == front_id && matched_front_stamps.contains(&record.stamp.as_nanos())
        })
        .cloned()
        .collect();
    if front_records.len() < 2 {
        return Err("at least two synchronized front records are required for odometry".into());
    }
    let front_episode = MemoryEpisode::from_records(front_records);
    let odometry =
        ScanOdometry::try_new(ScanOdometryConfig::new(front_episode.records().len(), 3))?;
    let matcher = IcpScanMatcher::new(IcpConfig {
        max_iterations: 10,
        max_correspondence_distance: 2.0,
        ..IcpConfig::default()
    });
    let odometry_result = odometry.estimate(&front_episode, &front_id, &matcher)?;
    advance_checkpoint(&mut checkpoint, &checkpoint_path, CheckpointStage::Odometry, &[])?;

    let mut volume = TsdfVolume::try_new(TSDF_ORIGIN, TSDF_VOXEL_SIZE, TSDF_DIMS, TSDF_TRUNCATION)?;
    let mut integrated_points = 0_u64;
    for (record, pose) in front_episode.records().iter().zip(odometry_result.trajectory.samples()) {
        integrated_points = integrated_points
            .checked_add(u64::try_from(volume.integrate_cloud_with_pose(
                record.record.cloud(),
                pose.pose.isometry,
                Vec3::new(0.0, 0.0, 0.0),
            )?)?)
            .ok_or("TSDF point counter overflow")?;
    }
    let mesh = volume.extract_mesh(1.0);
    advance_checkpoint(&mut checkpoint, &checkpoint_path, CheckpointStage::Tsdf, &[])?;
    let gltf_json = export_triangle_mesh_gltf_json(&mesh)?;
    let (gltf_vertices, gltf_indices) = import_triangle_mesh_gltf_json(&gltf_json)?;
    let gltf_path = output_dir.join("tsdf.mesh.gltf");
    fs::write(&gltf_path, format!("{gltf_json}\n"))?;
    advance_checkpoint(
        &mut checkpoint,
        &checkpoint_path,
        CheckpointStage::Interchange,
        std::slice::from_ref(&gltf_path),
    )?;

    let mut entities = Vec::with_capacity(front_episode.records().len());
    for record in front_episode.records() {
        entities.push(SpatialRecordEntity::try_from_record(
            &record.record,
            vec![OpenVocabLabel { text: "lidar_surface".into(), confidence: 1.0 }],
            None,
        )?);
    }
    let semantic_visual = spatial_record_entity_visual("rosbag2-e2e", &entities)?;
    let (mesh_layer, mesh_adapter) = mesh_visual(
        LayerId::try_new("scene/rosbag2-e2e/tsdf")?,
        "TSDF mesh",
        &mesh,
        LinearRgba::WHITE,
    )?;
    let semantic_layer = semantic_visual.as_layer()?;
    let semantic_adapter = semantic_visual.receipt;
    let mut visual_scene = VisualScene::new();
    visual_scene.add_layer(mesh_layer)?;
    visual_scene.add_layer(semantic_layer)?;
    advance_checkpoint(&mut checkpoint, &checkpoint_path, CheckpointStage::Viewer, &[])?;

    let after = StoragePreflight::check(&config.output_root, config.min_output_free_bytes)?;
    let receipt_path = output_dir.join("rosbag2.e2e.receipt.json");
    let manifest_path = output_dir.join("rosbag2.e2e.manifest.json");
    let receipt = E2eReceipt {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        input: input.display().to_string(),
        output_dir: output_dir.display().to_string(),
        front_topic: config.front_topic.clone(),
        rear_topic: config.rear_topic.clone(),
        storage: StorageReceipt {
            root: before.root.display().to_string(),
            required_free_bytes: before.required_free_bytes,
            available_before_bytes: before.available_bytes,
            available_after_pipeline_bytes: after.available_bytes,
        },
        ingest: IngestReceipt {
            chunk_points: config.chunk_points,
            source_memory_budget_bytes: config.source_memory_bytes,
            episode_max_points: DEFAULT_EPISODE_MAX_POINTS,
            episode_max_bytes: DEFAULT_EPISODE_MAX_BYTES,
            retained_records,
            retained_points,
            retained_bytes,
            topics: vec![front, rear],
        },
        sync: SyncReceipt {
            clock: "ros2-external",
            time_basis: "PointCloud2 header stamp; no clock calibration applied",
            max_records_per_topic: config.max_records_per_topic,
            max_delta_ns: config.max_delta_ns,
            matched_bundles,
            max_matched_delta_ns,
        },
        odometry: OdometryReceipt {
            topic: odometry_result.topic.as_str().to_owned(),
            frame_id: odometry_result.frame_id.0.clone(),
            scans: u64::try_from(odometry_result.trajectory.samples().len())?,
            motions: u64::try_from(odometry_result.motions.len())?,
            truncated: odometry_result.truncated,
            matcher: "point-to-point ICP; bounded ten-iteration smoke",
            pose_graph_nodes: u64::try_from(odometry_result.pose_graph.nodes().len())?,
            pose_graph_edges: u64::try_from(odometry_result.pose_graph.edges().len())?,
        },
        tsdf: TsdfReceipt {
            frame_id: odometry_result.frame_id.0.clone(),
            origin: [TSDF_ORIGIN.x, TSDF_ORIGIN.y, TSDF_ORIGIN.z],
            voxel_size: TSDF_VOXEL_SIZE,
            dims: TSDF_DIMS,
            truncation: TSDF_TRUNCATION,
            integrated_records: u64::try_from(front_episode.records().len())?,
            integrated_points,
            mesh_vertices: u64::try_from(mesh.vertex_count())?,
            mesh_triangles: u64::try_from(mesh.triangle_count())?,
        },
        semantic: SemanticReceipt {
            entities: u64::try_from(entities.len())?,
            visible_entities: u64::try_from(
                entities.iter().filter(|entity| entity.entity.centroid.is_some()).count(),
            )?,
            frame_ids: entities
                .iter()
                .map(|entity| entity.frame_id.0.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            runtime: "deterministic label adapter; no model runtime",
        },
        viewer: ViewerReceipt {
            layers: u64::try_from(visual_scene.layers().len())?,
            device_upload_bytes: 0,
            mesh: ViewerLayerReceipt {
                id: "scene/rosbag2-e2e/tsdf".into(),
                source_count: u64::try_from(mesh_adapter.source_count)?,
                output_count: u64::try_from(mesh_adapter.output_count)?,
                generated_bytes: u64::try_from(mesh_adapter.generated_bytes)?,
            },
            semantic: ViewerLayerReceipt {
                id: semantic_visual.id.as_str().to_owned(),
                source_count: u64::try_from(semantic_adapter.source_count)?,
                output_count: u64::try_from(semantic_adapter.output_count)?,
                generated_bytes: u64::try_from(semantic_adapter.generated_bytes)?,
            },
        },
        interchange: InterchangeReceipt {
            format: "glTF 2.0 JSON with embedded base64 buffers",
            path: gltf_path.display().to_string(),
            bytes: u64::try_from(gltf_json.len())?,
            vertices: u64::try_from(gltf_vertices)?,
            indices: u64::try_from(gltf_indices)?,
        },
        checkpoint: CheckpointReceipt {
            path: checkpoint_path.display().to_string(),
            stage: "complete",
            resume_mode: "complete-run manifest verification; partial runs are never overwritten",
            temporary_files_removed,
        },
        manifest_verified: config.verify_manifest,
    };
    write_json(&receipt_path, &receipt)?;
    advance_checkpoint(
        &mut checkpoint,
        &checkpoint_path,
        CheckpointStage::Receipt,
        &[gltf_path.clone(), receipt_path.clone()],
    )?;

    let mut manifest = DatasetManifest::new();
    manifest.add_file(ReceiptRole::Input, &input)?;
    manifest.add_file(ReceiptRole::Output, &gltf_path)?;
    manifest.add_file(ReceiptRole::Auxiliary, &receipt_path)?;
    if config.verify_manifest {
        let validation = manifest.validate_local_files()?;
        eprintln!(
            "validated E2E manifest: local_files={} uri_entries={} total_bytes={}",
            validation.checked_local_files, validation.uri_entries, validation.total_bytes
        );
        advance_checkpoint(
            &mut checkpoint,
            &checkpoint_path,
            CheckpointStage::ManifestVerified,
            &[gltf_path.clone(), receipt_path.clone()],
        )?;
    }
    manifest.write_json(&manifest_path)?;
    let completed_artifacts = [gltf_path.clone(), receipt_path.clone(), manifest_path.clone()];
    advance_checkpoint(
        &mut checkpoint,
        &checkpoint_path,
        CheckpointStage::Complete,
        &completed_artifacts,
    )?;
    eprintln!("wrote E2E receipt {}", receipt_path.display());
    eprintln!("wrote E2E manifest {}", manifest_path.display());
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn append_topic(
    builder: &mut MemoryEpisodeBuilder,
    input: &Path,
    config: &Config,
    topic: &str,
) -> Result<TopicReceipt, Box<dyn Error>> {
    let options =
        StreamOptions::new(config.chunk_points, MemoryBudget::new(config.source_memory_bytes)?)?;
    let mut source =
        Rosbag2PointCloudSource::open(input, topic, options, CancellationToken::default())?;
    let schema = source.schema().id.as_str().to_owned();
    let mut retained_chunks = 0_u64;
    let mut retained_points = 0_u64;
    let mut frame_ids = BTreeSet::new();
    while retained_chunks < config.max_records_per_topic {
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
    if input == "-h" || input == "--help" {
        return Err(usage().into());
    }
    let mut output_root = None;
    let mut output_dir = PathBuf::from("v1-3/143a-e2e-smoke");
    let mut front_topic = "/lidar_front/points_raw".to_owned();
    let mut rear_topic = "/lidar_rear/points_raw".to_owned();
    let mut max_records_per_topic = DEFAULT_MAX_RECORDS_PER_TOPIC;
    let mut max_delta_ns = DEFAULT_MAX_DELTA_NS;
    let mut chunk_points = DEFAULT_CHUNK_POINTS;
    let mut source_memory_bytes = DEFAULT_STREAM_MEMORY_BUDGET_BYTES;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    let mut verify_manifest = false;
    let mut resume = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-root" => output_root = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--output-dir" => output_dir = PathBuf::from(next_value(&mut args, &flag)?),
            "--front-topic" => front_topic = next_value(&mut args, &flag)?,
            "--rear-topic" => rear_topic = next_value(&mut args, &flag)?,
            "--max-records" => max_records_per_topic = parse_one(&mut args, &flag)?,
            "--max-delta-ns" => max_delta_ns = parse_one(&mut args, &flag)?,
            "--chunk-points" => chunk_points = parse_one(&mut args, &flag)?,
            "--memory-budget" => source_memory_bytes = parse_one(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_one(&mut args, &flag)?,
            "--verify-manifest" => verify_manifest = true,
            "--resume" => resume = true,
            _ => return Err(format!("unknown option '{}'\n{}", flag, usage()).into()),
        }
    }

    let output_root = output_root.ok_or("--output-root is required")?;
    if front_topic == rear_topic {
        return Err("front and rear topics must differ".into());
    }
    if output_dir.is_absolute() {
        return Err("--output-dir must be relative to --output-root".into());
    }
    if max_records_per_topic < 2 {
        return Err("--max-records must be at least 2 for odometry".into());
    }
    if chunk_points == 0 {
        return Err("--chunk-points must be greater than zero".into());
    }
    Ok(Config {
        input: PathBuf::from(input),
        output_root,
        output_dir,
        front_topic,
        rear_topic,
        max_records_per_topic,
        max_delta_ns,
        chunk_points,
        source_memory_bytes,
        min_output_free_bytes,
        verify_manifest,
        resume,
    })
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

fn advance_checkpoint(
    checkpoint: &mut RunCheckpoint,
    path: &Path,
    stage: CheckpointStage,
    artifacts: &[PathBuf],
) -> Result<(), Box<dyn Error>> {
    checkpoint.stage = stage;
    checkpoint.artifacts =
        artifacts.iter().map(|artifact| artifact.display().to_string()).collect();
    write_checkpoint(path, checkpoint)
}

fn write_checkpoint(path: &Path, checkpoint: &RunCheckpoint) -> Result<(), Box<dyn Error>> {
    let temporary = checkpoint_temp_path(path);
    write_json(&temporary, checkpoint)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn read_checkpoint(path: &Path) -> Result<RunCheckpoint, Box<dyn Error>> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read E2E checkpoint '{}': {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| {
        format!("cannot parse E2E checkpoint '{}': {error}", path.display()).into()
    })
}

fn validate_checkpoint(
    checkpoint: &RunCheckpoint,
    input: &Path,
    output_dir: &Path,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    if checkpoint.schema != CHECKPOINT_SCHEMA || checkpoint.version != CHECKPOINT_VERSION {
        return Err(format!(
            "unsupported E2E checkpoint schema/version in '{}': {}/{}",
            path.display(),
            checkpoint.schema,
            checkpoint.version
        )
        .into());
    }
    if checkpoint.input != input.display().to_string()
        || checkpoint.output_dir != output_dir.display().to_string()
    {
        return Err("E2E checkpoint does not match the requested input or output directory".into());
    }
    Ok(())
}

fn remove_checkpoint_temp(path: &Path) -> Result<u64, Box<dyn Error>> {
    let temporary = checkpoint_temp_path(path);
    if temporary.exists() {
        fs::remove_file(&temporary)?;
        Ok(1)
    } else {
        Ok(0)
    }
}

fn checkpoint_temp_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or("checkpoint.json");
    path.with_file_name(format!(".{file_name}.tmp"))
}

fn checkpoint_stage_label(stage: &CheckpointStage) -> &'static str {
    match stage {
        CheckpointStage::Created => "created",
        CheckpointStage::Ingested => "ingested",
        CheckpointStage::Synchronized => "synchronized",
        CheckpointStage::Odometry => "odometry",
        CheckpointStage::Tsdf => "tsdf",
        CheckpointStage::Interchange => "interchange",
        CheckpointStage::Viewer => "viewer",
        CheckpointStage::Receipt => "receipt",
        CheckpointStage::ManifestVerified => "manifest_verified",
        CheckpointStage::Complete => "complete",
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn usage() -> String {
    "usage: rosbag2_e2e INPUT_DB3 --output-root DIR [--output-dir RUN_DIR] \
     [--front-topic TOPIC] [--rear-topic TOPIC] [--max-records N] \
     [--max-delta-ns N] [--chunk-points N] [--memory-budget BYTES] \
     [--min-output-free-bytes BYTES] [--verify-manifest] [--resume]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{
        checkpoint_stage_label, checkpoint_temp_path, parse_args, read_checkpoint,
        remove_checkpoint_temp, validate_checkpoint, write_checkpoint, CheckpointStage,
        RunCheckpoint,
    };

    #[test]
    fn parses_bounded_e2e_options() {
        let config = parse_args(
            [
                "bag.db3",
                "--output-root",
                "/media/output",
                "--output-dir",
                "v1-3/test-run",
                "--front-topic",
                "/front",
                "--rear-topic",
                "/rear",
                "--max-records",
                "3",
                "--max-delta-ns",
                "5000000",
                "--chunk-points",
                "32",
                "--memory-budget",
                "4096",
                "--min-output-free-bytes",
                "100",
                "--verify-manifest",
                "--resume",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(config.max_records_per_topic, 3);
        assert_eq!(config.max_delta_ns, 5_000_000);
        assert_eq!(config.chunk_points, 32);
        assert_eq!(config.source_memory_bytes, 4096);
        assert_eq!(config.min_output_free_bytes, 100);
        assert!(config.verify_manifest);
        assert!(config.resume);
        assert_eq!(config.front_topic, "/front");
        assert_eq!(config.rear_topic, "/rear");
    }

    #[test]
    fn rejects_unsafe_or_unbounded_options() {
        let absolute_dir = parse_args(
            ["bag.db3", "--output-root", "/media/output", "--output-dir", "/tmp/run"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(absolute_dir.to_string().contains("must be relative"));

        let one_record = parse_args(
            ["bag.db3", "--output-root", "/media/output", "--max-records", "1"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(one_record.to_string().contains("at least 2"));
    }

    #[test]
    fn checkpoint_roundtrips_atomically_and_cleans_stale_temp() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.checkpoint.json");
        let checkpoint = RunCheckpoint {
            schema: "spatialrust.rosbag2.e2e.checkpoint".into(),
            version: 1,
            input: "/media/input/bag.db3".into(),
            output_dir: directory.path().display().to_string(),
            stage: CheckpointStage::Interchange,
            artifacts: vec!["/media/output/mesh.gltf".into()],
            temporary_files_removed: 0,
        };
        write_checkpoint(&path, &checkpoint).unwrap();
        assert_eq!(read_checkpoint(&path).unwrap(), checkpoint);
        validate_checkpoint(
            &checkpoint,
            std::path::Path::new("/media/input/bag.db3"),
            directory.path(),
            &path,
        )
        .unwrap();
        assert!(validate_checkpoint(
            &checkpoint,
            std::path::Path::new("/media/input/other.db3"),
            directory.path(),
            &path,
        )
        .is_err());
        assert_eq!(checkpoint_stage_label(&CheckpointStage::Interchange), "interchange");

        let temporary = checkpoint_temp_path(&path);
        std::fs::write(&temporary, "stale").unwrap();
        assert_eq!(remove_checkpoint_temp(&path).unwrap(), 1);
        assert!(!temporary.exists());
        assert_eq!(remove_checkpoint_temp(&path).unwrap(), 0);
    }
}

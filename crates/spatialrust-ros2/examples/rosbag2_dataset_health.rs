//! Build a source-bound Dataset Health dashboard from canonical receipts.
//!
//! The command re-validates the canonical E2E manifest once, checks the
//! source-bound calibration readiness receipt, validates the 145A-145F stage
//! snapshots, and emits a portable JSON/HTML/manifest bundle. A wrong
//! canonical SHA produces a dashboard with no admitted stage coverage and a
//! non-zero exit status; no alternate source is substituted.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_viewer::{
    CalibrationObservatoryState, DatasetHealthCheck, DatasetHealthStage, DatasetHealthState,
    DatasetHealthSummary, DatasetHealthTopic, DigitalTwinState, MapDiffState, ReplayArtifact,
    ReplayDemoState, SemanticOverlayState, StudioSource, StudioState,
};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const READINESS_SCHEMA: &str = "spatialrust.rosbag2.calibration-readiness";
const STATE_FILE: &str = "dataset-health.json";
const HTML_FILE: &str = "dataset-health.html";
const MANIFEST_FILE: &str = "dataset-health.manifest.json";

#[derive(Debug)]
struct Config {
    run_dir: PathBuf,
    results_root: PathBuf,
    readiness_path: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    expected_frame_id: String,
    min_output_free_bytes: u64,
}

struct LoadedCanonical {
    source: StudioSource,
    frame_id: String,
    time_basis: String,
    topics: Vec<DatasetHealthTopic>,
    source_message_count: u64,
    retained_record_count: u64,
    retained_point_count: u64,
    input_receipt: FileReceipt,
    map_receipt: FileReceipt,
    receipt_path: PathBuf,
    manifest_path: PathBuf,
    mesh_vertices: u64,
    mesh_triangles: u64,
    interchange_bytes: u64,
    interchange_vertices: u64,
    interchange_indices: u64,
}

struct StageSpec {
    id: String,
    label: String,
    state_path: PathBuf,
    dashboard_path: PathBuf,
    manifest_path: PathBuf,
    kind: &'static str,
    extra_files: Vec<(String, PathBuf)>,
}

#[derive(Debug, Deserialize)]
struct E2eReceipt {
    schema: String,
    version: u32,
    input: String,
    front_topic: String,
    rear_topic: String,
    ingest: IngestReceipt,
    sync: SyncReceipt,
    tsdf: TsdfReceipt,
    interchange: InterchangeReceipt,
}

#[derive(Debug, Deserialize)]
struct IngestReceipt {
    retained_records: u64,
    retained_points: u64,
    topics: Vec<IngestTopic>,
}

#[derive(Debug, Deserialize)]
struct IngestTopic {
    topic: String,
    schema: String,
    bag_message_count: u64,
    retained_chunks: u64,
    retained_points: u64,
    peak_source_bytes: u64,
    frame_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SyncReceipt {
    time_basis: String,
}

#[derive(Debug, Deserialize)]
struct TsdfReceipt {
    frame_id: String,
    mesh_vertices: u64,
    mesh_triangles: u64,
}

#[derive(Debug, Deserialize)]
struct InterchangeReceipt {
    path: String,
    bytes: u64,
    vertices: u64,
    indices: u64,
}

#[derive(Debug, Deserialize)]
struct CalibrationReadiness {
    schema: String,
    version: u32,
    input: FileReceipt,
    sensor_topics: Vec<SensorReadiness>,
    relevant_topics: Vec<RelevantTopic>,
    calibration_artifacts: CalibrationArtifacts,
    registration_ready: bool,
    blockers: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SensorReadiness {
    role: String,
    requested_topic: String,
    present: bool,
    supported_pointcloud2: bool,
    message_count: u64,
}

#[derive(Debug, Deserialize)]
struct RelevantTopic {
    topic: String,
    present: bool,
    observed_types: Vec<String>,
    message_count: u64,
}

#[derive(Debug, Deserialize)]
struct CalibrationArtifacts {
    clock: ReadinessArtifact,
    frame: ReadinessArtifact,
}

#[derive(Debug, Deserialize)]
struct ReadinessArtifact {
    status: String,
    path: Option<String>,
    file: Option<FileReceipt>,
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LegacyStageManifest {
    schema: String,
    version: u32,
    state: FileReceipt,
    dashboard: FileReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-dataset-health: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    validate_config(&config)?;

    let output_parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent")?;
    let preflight = StoragePreflight::check(output_parent, config.min_output_free_bytes)?;
    if config.output_dir.exists() {
        return Err(format!(
            "Dataset Health output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }

    let canonical = load_canonical(&config)?;
    let readiness: CalibrationReadiness = read_json(&config.readiness_path)?;
    validate_readiness(&readiness, &canonical)?;
    let source_identity_match = canonical.source.identity_matches;
    let frame_identity_match = canonical.frame_id == config.expected_frame_id;

    let mut artifacts = Vec::new();
    let mut manifest_entries = Vec::new();
    register_receipt(
        canonical.input_receipt.clone(),
        "canonical-source",
        &mut artifacts,
        &mut manifest_entries,
    )?;
    register_receipt(
        rebind_role(&canonical.map_receipt, ReceiptRole::Auxiliary),
        "canonical-map",
        &mut artifacts,
        &mut manifest_entries,
    )?;
    register_path(
        "canonical-e2e-receipt",
        ReceiptRole::Auxiliary,
        &canonical.receipt_path,
        &mut artifacts,
        &mut manifest_entries,
    )?;
    register_path(
        "canonical-e2e-manifest",
        ReceiptRole::Auxiliary,
        &canonical.manifest_path,
        &mut artifacts,
        &mut manifest_entries,
    )?;
    register_path(
        "calibration-readiness",
        ReceiptRole::Auxiliary,
        &config.readiness_path,
        &mut artifacts,
        &mut manifest_entries,
    )?;

    let specs = stage_specs(&config.results_root);
    let mut stages = Vec::new();
    let mut checks = Vec::new();
    let mut blockers = Vec::new();

    add_check(
        &mut checks,
        "canonical-source",
        "Canonical source identity",
        if source_identity_match { "pass" } else { "blocked" },
        true,
        &canonical.source.observed_sha256,
        &canonical.source.expected_sha256,
        if source_identity_match {
            "input manifest SHA-256 matches the operation contract"
        } else {
            "input manifest SHA-256 differs; all downstream stage admission is withheld"
        },
    )?;
    add_check(
        &mut checks,
        "canonical-manifest",
        "Canonical manifest integrity",
        "pass",
        true,
        "verified",
        "verified",
        "143A E2E manifest was re-hashed before health aggregation",
    )?;
    add_check(
        &mut checks,
        "canonical-frame",
        "Canonical frame identity",
        if frame_identity_match { "pass" } else { "blocked" },
        true,
        &canonical.frame_id,
        &config.expected_frame_id,
        if frame_identity_match {
            "TSDF and stage frame match the requested health frame"
        } else {
            "frame mismatch; cross-stage health admission is withheld"
        },
    )?;

    let topic_inventory_valid =
        canonical.topics.iter().all(|topic| topic.status == "pass") && canonical.topics.len() == 2;
    add_check(
        &mut checks,
        "topic-inventory",
        "Front/rear PointCloud2 inventory",
        if topic_inventory_valid { "pass" } else { "blocked" },
        true,
        &format!(
            "{} topics; {} source messages",
            canonical.topics.len(),
            canonical.source_message_count
        ),
        "two non-empty PointCloud2 topics",
        "front/rear topic presence, retained records, points, and frame IDs are receipt-backed",
    )?;

    let mesh_counts_valid = canonical.mesh_vertices > 0
        && canonical.mesh_triangles > 0
        && canonical.interchange_vertices == canonical.mesh_vertices
        && canonical.interchange_indices == canonical.mesh_triangles.saturating_mul(3);
    add_check(
        &mut checks,
        "mesh-counts",
        "Canonical TSDF/glTF geometry",
        if mesh_counts_valid { "pass" } else { "blocked" },
        true,
        &format!(
            "{} vertices / {} triangles; {} interchange bytes",
            canonical.mesh_vertices, canonical.mesh_triangles, canonical.interchange_bytes
        ),
        "non-empty TSDF mesh with receipt-backed interchange",
        "geometry counts were checked against the E2E receipt and manifest",
    )?;

    let readiness_source_match = readiness.input.path == canonical.source_path()
        && readiness.input.sha256 == canonical.input_receipt.sha256
        && readiness.input.size_bytes == canonical.input_receipt.size_bytes;
    add_check(
        &mut checks,
        "calibration-source",
        "Calibration receipt source binding",
        if readiness_source_match { "pass" } else { "blocked" },
        false,
        &readiness.input.path.display().to_string(),
        &canonical.source.path,
        if readiness_source_match {
            "calibration readiness receipt is bound to the canonical input"
        } else {
            "calibration readiness receipt belongs to another source and is not accepted"
        },
    )?;

    let sensor_inventory_valid = readiness.sensor_topics.len() == 2
        && readiness.sensor_topics.iter().all(|sensor| {
            sensor.present && sensor.supported_pointcloud2 && sensor.message_count > 0
        });
    add_check(
        &mut checks,
        "readiness-sensors",
        "Calibration sensor topic support",
        if sensor_inventory_valid { "pass" } else { "blocked" },
        true,
        &format!(
            "{} sensor declarations; {} ready",
            readiness.sensor_topics.len(),
            readiness
                .sensor_topics
                .iter()
                .filter(|sensor| sensor.present && sensor.supported_pointcloud2)
                .count()
        ),
        "front and rear PointCloud2 declarations supported",
        &sensor_detail(&readiness.sensor_topics),
    )?;

    let clock_topic = relevant_topic(&readiness, "/clock");
    let tf_topic = relevant_topic(&readiness, "/tf");
    let tf_static_topic = relevant_topic(&readiness, "/tf_static");
    let odom_topic = relevant_topic(&readiness, "/odom");
    let clock_present = clock_topic.map(|topic| topic.present).unwrap_or(false);
    let tf_present = tf_topic.map(|topic| topic.present).unwrap_or(false)
        || tf_static_topic.map(|topic| topic.present).unwrap_or(false);
    let odom_present = odom_topic.map(|topic| topic.present).unwrap_or(false);

    add_check(
        &mut checks,
        "clock-topic",
        "Clock basis",
        if clock_present { "pass" } else { "blocked" },
        false,
        &topic_detail(clock_topic),
        "registered /clock or equivalent source clock",
        if clock_present {
            "source clock topic is available for calibration"
        } else {
            "no /clock topic; header-stamp time remains uncalibrated"
        },
    )?;
    add_check(
        &mut checks,
        "tf-topics",
        "TF/frame evidence",
        if tf_present { "pass" } else { "blocked" },
        false,
        &format!("{}, {}", topic_detail(tf_topic), topic_detail(tf_static_topic)),
        "registered /tf or /tf_static evidence",
        if tf_present {
            "TF topic evidence is available for frame composition"
        } else {
            "no /tf or /tf_static topic; frame composition remains withheld"
        },
    )?;
    add_check(
        &mut checks,
        "odom-topic",
        "Odometry auxiliary evidence",
        if odom_present { "pass" } else { "warning" },
        false,
        &topic_detail(odom_topic),
        "optional /odom evidence",
        if odom_present {
            "odometry topic is present as auxiliary evidence"
        } else {
            "no /odom topic; ICP/replay evidence remains source-local"
        },
    )?;

    let calibration_ready = source_identity_match
        && frame_identity_match
        && readiness_source_match
        && readiness.registration_ready;
    add_check(
        &mut checks,
        "calibration-registration",
        "Clock/frame calibration registration",
        if calibration_ready { "pass" } else { "blocked" },
        false,
        &format!(
            "clock={}, frame={}, registration_ready={}",
            readiness.calibration_artifacts.clock.status,
            readiness.calibration_artifacts.frame.status,
            readiness.registration_ready
        ),
        "source-bound registered clock and frame artifacts",
        &calibration_artifact_detail(&readiness.calibration_artifacts),
    )?;

    if !source_identity_match {
        push_blocker(
            &mut blockers,
            "canonical input SHA-256 mismatch; stage aggregation and mapping are fail-closed",
        );
    }
    if !frame_identity_match {
        push_blocker(
            &mut blockers,
            "canonical frame mismatch; cross-stage health comparison is withheld",
        );
    }
    if !readiness_source_match {
        push_blocker(
            &mut blockers,
            "calibration readiness source identity does not match the canonical input",
        );
    }
    if !readiness.registration_ready {
        for blocker in &readiness.blockers {
            push_blocker(&mut blockers, blocker.clone());
        }
        push_blocker(
            &mut blockers,
            "mapping admission requires registered source-bound clock and frame calibration",
        );
    }
    if !clock_present {
        push_blocker(
            &mut blockers,
            "no /clock evidence; time basis remains PointCloud2 header stamp",
        );
    }
    if !tf_present {
        push_blocker(&mut blockers, "no /tf or /tf_static evidence; TF composition is not applied");
    }
    if !odom_present {
        push_blocker(
            &mut blockers,
            "no /odom evidence; odometry remains bounded inspection output",
        );
    }

    if source_identity_match {
        for spec in &specs {
            let required_files = [
                spec.state_path.as_path(),
                spec.dashboard_path.as_path(),
                spec.manifest_path.as_path(),
            ];
            if required_files.iter().any(|path| !path.is_file()) {
                add_check(
                    &mut checks,
                    &format!("stage-{}", spec.id),
                    &format!("{} stage snapshot", spec.label),
                    "blocked",
                    true,
                    "missing stage file",
                    "JSON, HTML, and manifest",
                    "stage coverage is withheld until all receipt-backed files exist",
                )?;
                push_blocker(&mut blockers, format!("{} stage evidence is incomplete", spec.id));
                continue;
            }

            if let Err(error) = verify_stage_manifest(&spec.manifest_path, &canonical.input_receipt)
            {
                add_check(
                    &mut checks,
                    &format!("stage-{}", spec.id),
                    &format!("{} stage snapshot", spec.label),
                    "blocked",
                    true,
                    "manifest verification failed",
                    "all local receipts unchanged",
                    format!("stage manifest withheld: {error}"),
                )?;
                push_blocker(
                    &mut blockers,
                    format!("{} stage manifest is not trustworthy", spec.id),
                );
                continue;
            }

            match load_stage(spec, &canonical.source, &config.expected_frame_id) {
                Ok(stage) => {
                    let status = stage.status.clone();
                    let critical = status == "blocked";
                    add_check(
                        &mut checks,
                        &format!("stage-{}", spec.id),
                        &format!("{} stage snapshot", spec.label),
                        &status,
                        critical,
                        &format!(
                            "ready={}, mapping_admitted={}",
                            stage.ready, stage.mapping_admitted
                        ),
                        "source-bound validated stage state",
                        &stage.detail,
                    )?;
                    if status == "warning" {
                        push_blocker(&mut blockers, format!("{}: {}", spec.id, stage.detail));
                    }
                    stages.push(stage);
                    register_path(
                        &format!("{}-state", spec.id),
                        ReceiptRole::Auxiliary,
                        &spec.state_path,
                        &mut artifacts,
                        &mut manifest_entries,
                    )?;
                    register_path(
                        &format!("{}-dashboard", spec.id),
                        ReceiptRole::Auxiliary,
                        &spec.dashboard_path,
                        &mut artifacts,
                        &mut manifest_entries,
                    )?;
                    register_path(
                        &format!("{}-manifest", spec.id),
                        ReceiptRole::Auxiliary,
                        &spec.manifest_path,
                        &mut artifacts,
                        &mut manifest_entries,
                    )?;
                    for (role, path) in &spec.extra_files {
                        register_path(
                            role,
                            ReceiptRole::Auxiliary,
                            path,
                            &mut artifacts,
                            &mut manifest_entries,
                        )?;
                    }
                }
                Err(error) => {
                    add_check(
                        &mut checks,
                        &format!("stage-{}", spec.id),
                        &format!("{} stage snapshot", spec.label),
                        "blocked",
                        true,
                        "state validation failed",
                        "validated portable state",
                        format!("stage state withheld: {error}"),
                    )?;
                    push_blocker(&mut blockers, format!("{} stage state is not valid", spec.id));
                }
            }
        }
    } else {
        add_check(
            &mut checks,
            "stage-coverage",
            "145A-145F stage coverage",
            "blocked",
            true,
            "withheld after source mismatch",
            "six source-bound stage snapshots",
            "alternate stage sources are never substituted for a mismatched canonical input",
        )?;
    }

    let stage_coverage_ok = stages.len() == specs.len();
    if source_identity_match {
        add_check(
            &mut checks,
            "stage-coverage",
            "145A-145F stage coverage",
            if stage_coverage_ok { "pass" } else { "blocked" },
            true,
            &format!("{} of {} stages validated", stages.len(), specs.len()),
            "six source-bound stage snapshots",
            if stage_coverage_ok {
                "all visual slices are represented by validated receipts"
            } else {
                "one or more stage snapshots are missing or invalid"
            },
        )?;
    }

    if !stage_coverage_ok {
        push_blocker(
            &mut blockers,
            format!("stage coverage incomplete: {} of {} validated", stages.len(), specs.len()),
        );
    }

    let artifact_bytes = artifacts.iter().try_fold(0_u64, |sum, artifact| {
        sum.checked_add(artifact.size_bytes)
            .ok_or_else(|| "Dataset Health artifact byte count overflow".to_string())
    })?;
    let source_message_count = canonical.source_message_count;
    let retained_record_count = canonical.retained_record_count;
    let retained_point_count = canonical.retained_point_count;
    let (pass_count, warning_count, blocked_count, critical_block_count) = check_counts(&checks);
    let summary = DatasetHealthSummary::try_new(
        canonical.input_receipt.size_bytes.ok_or("canonical input receipt has no size")?,
        source_message_count,
        retained_record_count,
        retained_point_count,
        u64::try_from(artifacts.len())?,
        artifact_bytes,
        u64::try_from(canonical.topics.len())?,
        u64::try_from(stages.len())?,
        u64::try_from(checks.len())?,
        pass_count,
        warning_count,
        blocked_count,
        critical_block_count,
        source_identity_match,
        frame_identity_match,
        calibration_ready,
    )?;

    fs::create_dir_all(&config.output_dir)?;
    let state = DatasetHealthState::try_new(
        "SpatialRust Dataset Health",
        canonical.source,
        canonical.frame_id,
        config.expected_frame_id,
        canonical.time_basis,
        canonical.topics,
        stages,
        artifacts,
        checks,
        summary,
        blockers,
    )?;
    state.validate()?;

    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state)?)?;
    manifest_entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest_entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let manifest = DatasetManifest { version: 1, entries: manifest_entries };
    let manifest_validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Dataset Health: {} (dataset_ready={}, mapping_admitted={})",
        state_path.display(),
        state.dataset_ready,
        state.mapping_admitted
    );
    println!("Dataset Health dashboard: {}", html_path.display());
    println!(
        "Dataset Health manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        manifest_validation.checked_local_files,
        manifest_validation.total_bytes,
        preflight.available_bytes
    );
    if !state.dataset_ready {
        return Err("Dataset Health failed its canonical source/frame/stage gates".into());
    }
    Ok(())
}

fn load_canonical(config: &Config) -> Result<LoadedCanonical, Box<dyn Error>> {
    let receipt_path = config.run_dir.join("rosbag2.e2e.receipt.json");
    let manifest_path = config.run_dir.join("rosbag2.e2e.manifest.json");
    let receipt: E2eReceipt = read_json(&receipt_path)?;
    if receipt.schema != RECEIPT_SCHEMA || !(1..=2).contains(&receipt.version) {
        return Err(format!(
            "unsupported E2E receipt schema/version in '{}': {}/{}",
            receipt_path.display(),
            receipt.schema,
            receipt.version
        )
        .into());
    }
    let manifest = DatasetManifest::read_json(&manifest_path)?;
    manifest.validate_local_files()?;

    let input_path = PathBuf::from(&receipt.input);
    if !input_path.is_absolute() {
        return Err(format!("canonical input path '{}' is not absolute", receipt.input).into());
    }
    let input_receipt = manifest_entry(&manifest, ReceiptRole::Input, &input_path)?;
    let observed_sha256 =
        input_receipt.sha256.clone().ok_or("canonical input receipt has no SHA-256")?;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        receipt.input,
        &config.expected_sha256,
        observed_sha256,
        input_receipt.sha256.as_deref() == Some(config.expected_sha256.as_str()),
    )?;

    let map_path = PathBuf::from(&receipt.interchange.path);
    if !map_path.is_absolute() {
        return Err("canonical interchange path must be absolute".into());
    }
    let map_receipt = manifest_entry(&manifest, ReceiptRole::Output, &map_path)?;
    if map_receipt.size_bytes.unwrap_or(0) == 0 || receipt.interchange.bytes == 0 {
        return Err("canonical glTF receipt and interchange bytes must be non-zero".into());
    }
    if receipt.interchange.vertices == 0
        || receipt.interchange.indices == 0
        || receipt.tsdf.mesh_vertices == 0
        || receipt.tsdf.mesh_triangles == 0
    {
        return Err("canonical E2E geometry counts must be non-empty".into());
    }

    let mut topics = Vec::new();
    for topic in &receipt.ingest.topics {
        let role = if topic.topic == receipt.front_topic {
            "front"
        } else if topic.topic == receipt.rear_topic {
            "rear"
        } else {
            "unknown"
        };
        let healthy = topic.schema.contains("PointCloud2")
            && topic.bag_message_count > 0
            && topic.retained_chunks > 0
            && topic.retained_points > 0
            && topic.peak_source_bytes > 0
            && !topic.frame_ids.is_empty();
        topics.push(DatasetHealthTopic::try_new(
            &topic.topic,
            role,
            topic.bag_message_count,
            if healthy { topic.retained_chunks } else { 0 },
            topic.retained_points,
            topic.frame_ids.clone(),
            if healthy { "pass" } else { "blocked" },
        )?);
    }

    Ok(LoadedCanonical {
        source,
        frame_id: receipt.tsdf.frame_id,
        time_basis: receipt.sync.time_basis,
        topics,
        source_message_count: receipt
            .ingest
            .topics
            .iter()
            .try_fold(0_u64, |sum, topic| sum.checked_add(topic.bag_message_count))
            .ok_or("canonical message count overflow")?,
        retained_record_count: receipt.ingest.retained_records,
        retained_point_count: receipt.ingest.retained_points,
        input_receipt,
        map_receipt,
        receipt_path,
        manifest_path,
        mesh_vertices: receipt.tsdf.mesh_vertices,
        mesh_triangles: receipt.tsdf.mesh_triangles,
        interchange_bytes: receipt.interchange.bytes,
        interchange_vertices: receipt.interchange.vertices,
        interchange_indices: receipt.interchange.indices,
    })
}

impl LoadedCanonical {
    fn source_path(&self) -> PathBuf {
        PathBuf::from(&self.source.path)
    }
}

fn validate_readiness(
    readiness: &CalibrationReadiness,
    canonical: &LoadedCanonical,
) -> Result<(), Box<dyn Error>> {
    if readiness.schema != READINESS_SCHEMA || readiness.version != 1 {
        return Err(format!(
            "unsupported calibration readiness schema/version: {}/{}",
            readiness.schema, readiness.version
        )
        .into());
    }
    if readiness.input.path != canonical.source_path() {
        return Err("calibration readiness input path is not absolute canonical input".into());
    }
    Ok(())
}

fn load_stage(
    spec: &StageSpec,
    canonical_source: &StudioSource,
    expected_frame_id: &str,
) -> Result<DatasetHealthStage, Box<dyn Error>> {
    let (stage_source, ready, mapping_admitted, frame_identity_match, detail) = match spec.kind {
        "studio" => {
            let state: StudioState = read_json(&spec.state_path)?;
            state.validate()?;
            (
                state.source,
                true,
                state.mapping_admitted,
                None,
                format!("{} layers; time basis {}", state.layers.len(), state.timeline.time_basis),
            )
        }
        "observatory" => {
            let state: CalibrationObservatoryState = read_json(&spec.state_path)?;
            state.validate()?;
            (
                state.source,
                state.calibration_admitted,
                state.calibration_admitted,
                None,
                format!(
                    "{} frame edges; {} rejected; clock {}",
                    state.edges.len(),
                    state.rejected_edge_count,
                    state.clock.status
                ),
            )
        }
        "replay" => {
            let state: ReplayDemoState = read_json(&spec.state_path)?;
            state.validate()?;
            let frame_match = state
                .topics
                .iter()
                .any(|topic| topic.frame_ids.iter().any(|frame_id| frame_id == expected_frame_id));
            (
                state.source,
                state.replay_ready,
                state.mapping_admitted,
                Some(frame_match),
                format!(
                    "{} replay records; {} matched bundles; {}",
                    state.summary.replayed_record_count,
                    state.summary.matched_bundle_count,
                    state.summary.time_basis
                ),
            )
        }
        "map-diff" => {
            let state: MapDiffState = read_json(&spec.state_path)?;
            state.validate()?;
            let frame_match = state.base.frame_id == expected_frame_id
                && state.candidate.frame_id == expected_frame_id;
            (
                state.base.source,
                state.compare_ready,
                state.mapping_admitted,
                Some(frame_match),
                format!(
                    "{} stable-index vertices compared; {} heatmap cells",
                    state.summary.compared_vertex_count, state.summary.cell_count
                ),
            )
        }
        "semantic" => {
            let state: SemanticOverlayState = read_json(&spec.state_path)?;
            state.validate()?;
            let frame_match = state.frame_id == expected_frame_id;
            (
                state.source,
                state.overlay_ready,
                state.mapping_admitted,
                Some(frame_match),
                format!(
                    "{} semantic entities; {} classes; model {}",
                    state.summary.entity_count, state.summary.class_count, state.model.model_id
                ),
            )
        }
        "digital-twin" => {
            let state: DigitalTwinState = read_json(&spec.state_path)?;
            state.validate()?;
            let frame_match = state.frame_id == expected_frame_id;
            (
                state.source,
                state.twin_ready,
                state.mapping_admitted,
                Some(frame_match),
                format!(
                    "{} assets; semantic layer {}; {} vertices",
                    state.assets.len(),
                    state.summary.semantic_layer_present,
                    state.summary.source_vertex_count
                ),
            )
        }
        _ => return Err(format!("unknown Dataset Health stage kind '{}'", spec.kind).into()),
    };
    let source_identity_match = same_source(&stage_source, canonical_source);
    let status = if !source_identity_match || frame_identity_match == Some(false) {
        "blocked"
    } else if ready {
        "pass"
    } else {
        "warning"
    };
    DatasetHealthStage::try_new(
        &spec.id,
        &spec.label,
        status,
        ready,
        mapping_admitted,
        source_identity_match,
        frame_identity_match,
        u64::try_from(3 + spec.extra_files.len())?,
        detail,
    )
    .map_err(Into::into)
}

fn stage_specs(results_root: &Path) -> Vec<StageSpec> {
    vec![
        stage_spec(
            results_root,
            "145A",
            "Spatial Studio",
            "145a-spatial-studio-v2",
            "spatial-studio.json",
            "spatial-studio.html",
            "studio",
            Vec::new(),
        ),
        stage_spec(
            results_root,
            "145B",
            "TF / Calibration Observatory",
            "145b-tf-calibration-observatory",
            "calibration-observatory.json",
            "calibration-observatory.html",
            "observatory",
            Vec::new(),
        ),
        stage_spec(
            results_root,
            "145C",
            "One-command Replay Demo",
            "145c-one-command-replay-demo",
            "replay-demo.json",
            "replay-demo.html",
            "replay",
            Vec::new(),
        ),
        stage_spec(
            results_root,
            "145D",
            "Map Diff",
            "145d-map-diff-v2",
            "map-diff.json",
            "map-diff.html",
            "map-diff",
            Vec::new(),
        ),
        stage_spec(
            results_root,
            "145E",
            "AI Semantic Overlay",
            "145e-ai-semantic-overlay",
            "semantic-overlay.json",
            "semantic-overlay.html",
            "semantic",
            Vec::new(),
        ),
        stage_spec(
            results_root,
            "145F",
            "glTF / USD Digital Twin",
            "145f-digital-twin",
            "digital-twin.json",
            "digital-twin.html",
            "digital-twin",
            vec![
                ("145F-gltf".into(), results_root.join("145f-digital-twin/digital-twin.gltf")),
                ("145F-usda".into(), results_root.join("145f-digital-twin/digital-twin.usda")),
            ],
        ),
    ]
}

fn stage_spec(
    results_root: &Path,
    id: &str,
    label: &str,
    directory: &str,
    state_file: &str,
    dashboard_file: &str,
    kind: &'static str,
    extra_files: Vec<(String, PathBuf)>,
) -> StageSpec {
    let directory = results_root.join(directory);
    StageSpec {
        id: id.into(),
        label: label.into(),
        state_path: directory.join(state_file),
        dashboard_path: directory.join(dashboard_file),
        manifest_path: directory.join(match kind {
            "studio" => "spatial-studio.manifest.json",
            "observatory" => "calibration-observatory.manifest.json",
            "replay" => "replay-demo.manifest.json",
            "map-diff" => "map-diff.manifest.json",
            "semantic" => "semantic-overlay.manifest.json",
            "digital-twin" => "digital-twin.manifest.json",
            _ => "stage.manifest.json",
        }),
        kind,
        extra_files,
    }
}

fn verify_stage_manifest(
    path: &Path,
    canonical_input: &FileReceipt,
) -> Result<u64, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    if let Ok(manifest) = serde_json::from_str::<DatasetManifest>(&text) {
        if manifest.version != 1 {
            return Err(format!("unsupported stage manifest version {}", manifest.version).into());
        }
        for entry in &manifest.entries {
            if entry.path == canonical_input.path {
                if entry.size_bytes != canonical_input.size_bytes
                    || entry.sha256 != canonical_input.sha256
                {
                    return Err(
                        format!("canonical input receipt differs in '{}'", path.display()).into()
                    );
                }
                continue;
            }
            verify_receipt_entry(entry)?;
        }
        return Ok(u64::try_from(manifest.entries.len())?);
    }

    let legacy: LegacyStageManifest = serde_json::from_str(&text).map_err(|error| {
        format!(
            "stage manifest '{}' is neither a DatasetManifest nor a legacy stage manifest: {error}",
            path.display()
        )
    })?;
    if legacy.schema.trim().is_empty() || legacy.version != 1 {
        return Err(format!(
            "unsupported legacy stage manifest '{}': {}/{}",
            path.display(),
            legacy.schema,
            legacy.version
        )
        .into());
    }
    verify_receipt_entry(&legacy.state)?;
    verify_receipt_entry(&legacy.dashboard)?;
    Ok(2)
}

fn verify_receipt_entry(entry: &FileReceipt) -> Result<(), Box<dyn Error>> {
    match (&entry.size_bytes, &entry.sha256) {
        (Some(expected_size), Some(expected_sha256)) => {
            let observed = FileReceipt::from_path(entry.role, &entry.path)?;
            if observed.size_bytes.as_ref() != Some(expected_size)
                || observed.sha256.as_deref() != Some(expected_sha256)
            {
                return Err(format!(
                    "stage manifest receipt changed for '{}'",
                    entry.path.display()
                )
                .into());
            }
        }
        (None, None) if entry.path.to_string_lossy().contains("://") => {}
        _ => {
            return Err(format!(
                "stage manifest entry '{}' has an incomplete receipt",
                entry.path.display()
            )
            .into());
        }
    }
    Ok(())
}

fn relevant_topic<'a>(
    readiness: &'a CalibrationReadiness,
    name: &str,
) -> Option<&'a RelevantTopic> {
    readiness.relevant_topics.iter().find(|topic| topic.topic == name)
}

fn topic_detail(topic: Option<&RelevantTopic>) -> String {
    topic
        .map(|topic| {
            format!(
                "{} present={} messages={} types={}",
                topic.topic,
                topic.present,
                topic.message_count,
                topic.observed_types.len()
            )
        })
        .unwrap_or_else(|| "not declared".into())
}

fn sensor_detail(sensors: &[SensorReadiness]) -> String {
    sensors
        .iter()
        .map(|sensor| {
            format!(
                "{} {} present={} supported={} messages={}",
                sensor.role,
                sensor.requested_topic,
                sensor.present,
                sensor.supported_pointcloud2,
                sensor.message_count
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn calibration_artifact_detail(artifacts: &CalibrationArtifacts) -> String {
    fn detail(artifact: &ReadinessArtifact) -> String {
        format!(
            "status={} path={} file_registered={} reason={}",
            artifact.status,
            artifact.path.as_deref().unwrap_or("<none>"),
            artifact.file.is_some(),
            artifact.reason.as_deref().unwrap_or("<none>")
        )
    }
    format!("clock [{}]; frame [{}]", detail(&artifacts.clock), detail(&artifacts.frame))
}

fn check_counts(checks: &[DatasetHealthCheck]) -> (u64, u64, u64, u64) {
    let mut pass = 0_u64;
    let mut warning = 0_u64;
    let mut blocked = 0_u64;
    let mut critical = 0_u64;
    for check in checks {
        match check.status.as_str() {
            "pass" => pass = pass.saturating_add(1),
            "warning" => warning = warning.saturating_add(1),
            "blocked" => {
                blocked = blocked.saturating_add(1);
                if check.critical {
                    critical = critical.saturating_add(1);
                }
            }
            _ => {}
        }
    }
    (pass, warning, blocked, critical)
}

fn add_check(
    checks: &mut Vec<DatasetHealthCheck>,
    id: &str,
    label: &str,
    status: &str,
    critical: bool,
    observed: &str,
    expected: &str,
    detail: impl Into<String>,
) -> Result<(), Box<dyn Error>> {
    checks.push(DatasetHealthCheck::try_new(
        id, label, status, critical, observed, expected, detail,
    )?);
    Ok(())
}

fn same_source(left: &StudioSource, right: &StudioSource) -> bool {
    left.identity_matches
        && right.identity_matches
        && left.path == right.path
        && left.expected_sha256 == right.expected_sha256
        && left.observed_sha256 == right.observed_sha256
}

fn manifest_entry(
    manifest: &DatasetManifest,
    role: ReceiptRole,
    path: &Path,
) -> Result<FileReceipt, Box<dyn Error>> {
    manifest
        .entries
        .iter()
        .find(|entry| entry.role == role && entry.path == path)
        .cloned()
        .ok_or_else(|| format!("manifest has no {role:?} entry for '{}'", path.display()).into())
}

fn artifact_from_receipt(
    role: &str,
    receipt: &FileReceipt,
) -> Result<ReplayArtifact, Box<dyn Error>> {
    ReplayArtifact::try_new(
        role,
        receipt.path.display().to_string(),
        receipt.size_bytes.ok_or("artifact receipt has no size")?,
        receipt.sha256.clone().ok_or("artifact receipt has no SHA-256")?,
    )
    .map_err(Into::into)
}

fn register_receipt(
    receipt: FileReceipt,
    role: &str,
    artifacts: &mut Vec<ReplayArtifact>,
    manifest_entries: &mut Vec<FileReceipt>,
) -> Result<(), Box<dyn Error>> {
    artifacts.push(artifact_from_receipt(role, &receipt)?);
    manifest_entries.push(receipt);
    Ok(())
}

fn register_path(
    role: &str,
    receipt_role: ReceiptRole,
    path: &Path,
    artifacts: &mut Vec<ReplayArtifact>,
    manifest_entries: &mut Vec<FileReceipt>,
) -> Result<(), Box<dyn Error>> {
    let receipt = FileReceipt::from_path(receipt_role, path)?;
    register_receipt(receipt, role, artifacts, manifest_entries)
}

fn rebind_role(receipt: &FileReceipt, role: ReceiptRole) -> FileReceipt {
    let mut receipt = receipt.clone();
    receipt.role = role;
    receipt
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    write_text_atomically(path, &format!("{}\n", serde_json::to_string_pretty(value)?))
}

fn write_text_atomically(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("output '{}' already exists", path.display()).into());
    }
    let file_name = path.file_name().ok_or("output path has no file name")?;
    let temp = path.with_file_name(format!(".{}.tmp", file_name.to_string_lossy()));
    if temp.exists() {
        return Err(format!("temporary output '{}' already exists", temp.display()).into());
    }
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

fn render_dashboard(state: &DatasetHealthState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#070b16;--panel:#101b2f;--line:#294767;--muted:#91a9c2;--cyan:#69e8ff;--green:#6cf5a7;--red:#ff7382;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#28436e 0,#070b16 48%);color:#edf7ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:30px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}.title{font-size:31px;font-weight:780;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:10px 15px;font-weight:780;white-space:nowrap}.pass{color:var(--green);border-color:#237a50}.warning{color:var(--amber);border-color:#876b2e}.blocked{color:var(--red);border-color:#873443}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.panel{background:linear-gradient(145deg,rgba(18,38,65,.96),rgba(8,15,28,.96));border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 16px 36px #0005}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 12px}.metric{font-size:27px;font-weight:780;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:14px;border-bottom:1px solid #1c3552;padding:9px 0}.row:last-child{border-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;overflow-wrap:anywhere}.danger{color:var(--red)}.warningText{color:var(--amber)}.list{margin:0;padding:0;list-style:none}.list li{border-bottom:1px solid #1c3552;padding:10px 0}.list li:last-child{border-bottom:0}.pill{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:2px 7px;font-size:11px;margin-right:7px}.rail{display:grid;grid-template-columns:repeat(6,1fr);gap:7px}.rail div{min-height:72px;border:1px solid var(--line);border-radius:10px;padding:9px;background:#0b1930}.rail strong{display:block;color:var(--cyan);font-size:15px}.meter{height:9px;border-radius:8px;background:#1b304b;overflow:hidden;margin-top:12px}.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));border-radius:8px}.checkGrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.check{border:1px solid #254363;border-radius:10px;padding:10px;background:#0b1930}.check .label{font-weight:700}.artifact{font-size:11px;line-height:1.35;margin:5px 0;padding:7px;border-left:2px solid var(--cyan);background:#0b1930;overflow-wrap:anywhere}@media(max-width:1050px){.grid{grid-template-columns:1fr 1fr}.rail{grid-template-columns:repeat(3,1fr)}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}.rail,.checkGrid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / dataset health</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Dataset gate</h2><div id="dataset" class="metric"></div><div id="datasetDetail" class="small"></div></article>
<article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Canonical source</h2><div id="sourceBytes" class="metric"></div><div id="sourceDetail" class="small"></div></article>
<article class="panel"><h2>Checks</h2><div id="checksCount" class="metric"></div><div id="checksDetail" class="small"></div></article>
<article class="panel full"><h2>Visual slice coverage</h2><div id="rail" class="rail"></div><div class="meter"><i id="coverage"></i></div></article>
<article class="panel wide"><h2>Topic health</h2><ul id="topics" class="list"></ul></article>
<article class="panel wide"><h2>Integrity checks</h2><div id="checks" class="checkGrid"></div></article>
<article class="panel wide"><h2>Dataset metrics</h2><div id="metrics"></div></article>
<article class="panel wide"><h2>Fail-closed blockers</h2><ul id="blockers" class="list"></ul></article>
<article class="panel full"><h2>Artifact lineage</h2><div id="artifacts"></div></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:320px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="dataset-health-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('dataset-health-state').textContent),q=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])),statusClass=s=>s==='pass'?'pass':s==='warning'?'warning':'blocked';
q('title').textContent=state.title;q('source').textContent=state.source.path+' · '+state.source.observed_sha256;q('admission').textContent=state.dataset_ready?'DATASET READY':'DATASET BLOCKED';q('admission').className='badge '+(state.dataset_ready?'pass':'blocked');
q('dataset').textContent=state.dataset_ready?'READY':'BLOCKED';q('dataset').className='metric '+(state.dataset_ready?'':'danger');q('datasetDetail').textContent=state.summary.critical_block_count+' critical blocked checks';
q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_ready?'calibration registered':'inspection-only · calibration absent';
q('sourceBytes').textContent=(state.summary.source_bytes/1073741824).toFixed(2)+' GB';q('sourceDetail').textContent=fmt(state.summary.source_message_count)+' source messages · '+fmt(state.summary.retained_point_count)+' retained points';
const s=state.summary;q('checksCount').textContent=fmt(s.pass_count)+' / '+fmt(s.check_count);q('checksDetail').textContent=fmt(s.warning_count)+' warnings · '+fmt(s.blocked_count)+' blocked';
q('rail').innerHTML=state.stages.map(stage=>'<div class="'+statusClass(stage.status)+'"><strong>'+esc(stage.id)+'</strong><span>'+esc(stage.label)+'</span><br><span class="pill">'+esc(stage.status.toUpperCase())+'</span><span class="small">'+(stage.ready?'ready':'gate blocked')+'</span></div>').join('')||'<div class="blocked">No stage snapshots admitted</div>';q('coverage').style.width=(100*state.stages.length/6).toFixed(1)+'%';
q('topics').innerHTML=state.topics.map(t=>'<li><span class="pill '+statusClass(t.status)+'">'+esc(t.status.toUpperCase())+'</span><strong>'+esc(t.role)+'</strong> <span class="mono">'+esc(t.name)+'</span><div class="small">'+fmt(t.message_count)+' messages · '+fmt(t.retained_record_count)+' records · '+fmt(t.retained_point_count)+' points · '+esc(t.frame_ids.join(', '))+'</div></li>').join('')||'<li class="blocked">Topic inventory withheld</li>';
q('checks').innerHTML=state.checks.map(c=>'<div class="check '+statusClass(c.status)+'"><div class="label"><span class="pill">'+esc(c.status.toUpperCase())+'</span>'+esc(c.label)+'</div><div class="small">'+esc(c.observed)+' / '+esc(c.expected)+'</div><div class="small">'+esc(c.detail)+'</div></div>').join('');
q('metrics').innerHTML='<div class="row"><span>retained records / points</span><span class="mono">'+fmt(s.retained_record_count)+' / '+fmt(s.retained_point_count)+'</span></div><div class="row"><span>stage snapshots</span><span class="mono">'+fmt(s.stage_count)+' / 6</span></div><div class="row"><span>artifacts / bytes</span><span class="mono">'+fmt(s.artifact_count)+' / '+(s.artifact_bytes/1048576).toFixed(1)+' MiB</span></div><div class="row"><span>frame</span><span class="mono">'+esc(state.frame_id)+' / '+esc(state.expected_frame_id)+'</span></div><div class="row"><span>time basis</span><span class="mono">'+esc(state.time_basis)+'</span></div>';
q('blockers').innerHTML=state.blockers.map(v=>'<li><span class="pill blocked">BLOCKER</span>'+esc(v)+'</li>').join('')||'<li class="pass">All gates passed</li>';
q('artifacts').innerHTML=state.artifacts.map(a=>'<div class="artifact"><strong>'+esc(a.role)+'</strong> · '+fmt(a.size_bytes)+' bytes<br><span class="mono">'+esc(a.path)+'<br>'+esc(a.sha256)+'</span></div>').join('');
q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let run_dir = args.next().ok_or_else(usage)?;
    let mut results_root = None;
    let mut readiness_path = None;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut expected_frame_id = None;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--results-root" => results_root = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--calibration-readiness" => {
                readiness_path = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--expected-frame-id" => expected_frame_id = Some(next_value(&mut args, &flag)?),
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        run_dir: PathBuf::from(run_dir),
        results_root: results_root.ok_or("--results-root is required")?,
        readiness_path: readiness_path.ok_or("--calibration-readiness is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        expected_frame_id: expected_frame_id.ok_or("--expected-frame-id is required")?,
        min_output_free_bytes,
    })
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    for (label, path) in [
        ("run directory", &config.run_dir),
        ("results root", &config.results_root),
        ("readiness path", &config.readiness_path),
        ("output directory", &config.output_dir),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute").into());
        }
    }
    if !config.run_dir.is_dir() || !config.results_root.is_dir() {
        return Err("run directory and results root must exist as directories".into());
    }
    if !config.readiness_path.is_file() {
        return Err("calibration readiness path must be a regular file".into());
    }
    if config.output_dir == Path::new("/") {
        return Err("--output-dir must not be the filesystem root".into());
    }
    let parent = config.output_dir.parent().ok_or("--output-dir must have an existing parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent '{}' is not a directory", parent.display()).into());
    }
    if config.expected_frame_id.trim().is_empty() || config.min_output_free_bytes == 0 {
        return Err("expected frame and free-space floor must be non-empty/non-zero".into());
    }
    validate_sha256(&config.expected_sha256)
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

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, Box<dyn Error>> {
    Ok(next_value(args, flag)?.parse()?)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn usage() -> String {
    "usage: rosbag2_dataset_health E2E_RUN_DIR \
     --results-root ABSOLUTE_RESULTS_ROOT \
     --calibration-readiness ABSOLUTE_READINESS_JSON \
     --output-dir ABSOLUTE_OUTPUT_DIR \
     --expected-input-sha256 SHA256 --expected-frame-id FRAME \
     [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, validate_sha256};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_dataset_health_options() {
        let config = parse_args(
            [
                "/media/e2e",
                "--results-root",
                "/media/results/v1-3",
                "--calibration-readiness",
                "/media/results/readiness.json",
                "--output-dir",
                "/media/results/health",
                "--expected-input-sha256",
                SHA,
                "--expected-frame-id",
                "lidar_front",
                "--min-output-free-bytes",
                "1024",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.expected_frame_id, "lidar_front");
        assert_eq!(config.min_output_free_bytes, 1024);
    }

    #[test]
    fn rejects_bad_hash() {
        assert!(validate_sha256("not-a-sha").is_err());
    }
}

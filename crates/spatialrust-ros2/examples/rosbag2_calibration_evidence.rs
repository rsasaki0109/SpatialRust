//! Register source-bound clock and front/rear extrinsic evidence.
//!
//! This example validates explicit JSON evidence documents; it does not solve
//! calibration, infer a root frame, or apply a timestamp/TF transform. A
//! blocked state is still written so missing or mismatched evidence remains
//! auditable on the external result disk.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_viewer::{
    CalibrationArtifact, CalibrationEvidenceClock, CalibrationEvidenceFrame,
    CalibrationEvidenceState, ClockCalibration, FrameTransform, StudioSource,
};

const CLOCK_SCHEMA: &str = "spatialrust.calibration.clock-evidence";
const FRAME_SCHEMA: &str = "spatialrust.calibration.frame-evidence";
const READINESS_SCHEMA: &str = "spatialrust.rosbag2.calibration-readiness";
const EVIDENCE_VERSION: u32 = 1;
const STATE_FILE: &str = "calibration-evidence.json";
const HTML_FILE: &str = "calibration-evidence.html";
const MANIFEST_FILE: &str = "calibration-evidence.manifest.json";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    readiness: PathBuf,
    clock_artifact: Option<PathBuf>,
    frame_artifact: Option<PathBuf>,
    output_dir: PathBuf,
    expected_sha256: String,
    root_frame: String,
    front_frame: String,
    rear_frame: String,
    min_output_free_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBindingDocument {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClockEvidenceDocument {
    schema: String,
    version: u32,
    source: SourceBindingDocument,
    source_domain: String,
    target_domain: String,
    method: String,
    time_basis: String,
    sample_count: u64,
    median_offset_nanos: Option<f64>,
    p95_abs_offset_nanos: Option<f64>,
    drift_ppm: Option<f64>,
    uncertainty_nanos: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameEvidenceDocument {
    schema: String,
    version: u32,
    source: SourceBindingDocument,
    method: String,
    root_frame: String,
    edges: Vec<FrameEvidenceEdgeDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameEvidenceEdgeDocument {
    parent_frame: String,
    child_frame: String,
    translation_m: [f64; 3],
    rotation_xyzw: [f64; 4],
    stamp_nanos: Option<u64>,
}

#[derive(Debug)]
struct ArtifactFile {
    receipt: Option<FileReceipt>,
    path: Option<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-calibration-evidence: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    validate_config(&config)?;
    ensure_outputs_absent(&config)?;
    if !config.input.is_file() {
        return Err(format!("input bag '{}' is not a regular file", config.input.display()).into());
    }

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

    let readiness_receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, &config.readiness)?;
    let readiness: Value = read_json(&config.readiness)?;
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
    inspect_readiness(&readiness, &input_path, input_size, &observed_sha256, &mut blockers)?;

    let clock_file = inspect_artifact_file(config.clock_artifact.as_deref())?;
    let frame_file = inspect_artifact_file(config.frame_artifact.as_deref())?;
    let (clock_artifact, clock, clock_blockers) =
        build_clock_evidence(&clock_file, &input_path, input_size, &observed_sha256)?;
    let (frame_artifact, frame, frame_blockers) = build_frame_evidence(
        &frame_file,
        &input_path,
        input_size,
        &observed_sha256,
        &config.root_frame,
        &config.front_frame,
        &config.rear_frame,
    )?;
    blockers.extend(clock_blockers);
    blockers.extend(frame_blockers);
    if !clock.registration_ready() {
        push_blocker(&mut blockers, "clock evidence registration is incomplete");
    }
    if !frame.registration_ready() {
        push_blocker(&mut blockers, "frame evidence has no complete root-to-front/rear path");
    }

    let state = CalibrationEvidenceState::try_new(
        format!("Calibration Evidence — {}", file_label(&config.input)),
        source,
        clock_artifact,
        frame_artifact,
        clock,
        frame,
        blockers,
    )?;
    state.validate()?;

    fs::create_dir_all(&config.output_dir)?;
    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state)?)?;

    let mut manifest = DatasetManifest::new();
    manifest.entries.push(input_receipt);
    manifest.entries.push(readiness_receipt);
    if let Some(receipt) = clock_file.receipt {
        manifest.entries.push(receipt);
    }
    if let Some(receipt) = frame_file.receipt {
        manifest.entries.push(receipt);
    }
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Calibration evidence receipt: {} (registration_ready={})",
        state_path.display(),
        state.registration_ready
    );
    println!("Calibration evidence dashboard: {}", html_path.display());
    println!(
        "Calibration evidence manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.registration_ready {
        return Err(
            "calibration evidence registration gate failed; see the external receipt blockers"
                .into(),
        );
    }
    Ok(())
}

fn build_clock_evidence(
    artifact_file: &ArtifactFile,
    input_path: &str,
    input_size: u64,
    observed_sha256: &str,
) -> Result<(CalibrationArtifact, CalibrationEvidenceClock, Vec<String>), Box<dyn Error>> {
    let Some(path) = artifact_file.path.as_deref() else {
        return Ok((
            CalibrationArtifact::try_new("clock_evidence", "not_registered", None, None, false)?,
            unregistered_clock(),
            vec!["clock evidence file was not supplied".into()],
        ));
    };
    let Some(receipt) = artifact_file.receipt.as_ref() else {
        return Ok((
            CalibrationArtifact::try_new(
                "clock_evidence",
                "missing",
                Some(path.to_owned()),
                None,
                false,
            )?,
            invalid_clock("clock evidence file is missing"),
            vec![format!("clock evidence file '{}' is missing", path)],
        ));
    };
    let artifact_path = Some(path.to_owned());
    let artifact_sha = receipt.sha256.clone();
    let document: ClockEvidenceDocument = match read_json(Path::new(path)) {
        Ok(document) => document,
        Err(error) => {
            return Ok((
                CalibrationArtifact::try_new(
                    "clock_evidence",
                    "invalid",
                    artifact_path,
                    artifact_sha,
                    false,
                )?,
                invalid_clock(format!("clock evidence JSON is invalid: {error}")),
                vec![format!("clock evidence JSON is invalid: {error}")],
            ));
        }
    };
    if document.schema != CLOCK_SCHEMA || document.version != EVIDENCE_VERSION {
        return Ok((
            CalibrationArtifact::try_new(
                "clock_evidence",
                "invalid",
                artifact_path,
                artifact_sha,
                false,
            )?,
            invalid_clock("clock evidence schema or version is unsupported"),
            vec!["clock evidence schema or version is unsupported".into()],
        ));
    }
    if !source_binding_matches(&document.source, input_path, input_size, observed_sha256) {
        return Ok((
            CalibrationArtifact::try_new(
                "clock_evidence",
                "source_mismatch",
                artifact_path,
                artifact_sha,
                false,
            )?,
            invalid_clock("clock evidence source identity does not match the canonical input"),
            vec!["clock evidence source identity does not match the canonical input".into()],
        ));
    }
    let calibration = match ClockCalibration::try_new(
        "registered",
        document.time_basis,
        document.sample_count,
        document.median_offset_nanos,
        document.p95_abs_offset_nanos,
        document.drift_ppm,
        document.uncertainty_nanos,
        true,
        false,
    ) {
        Ok(calibration) => calibration,
        Err(error) => {
            return Ok((
                CalibrationArtifact::try_new(
                    "clock_evidence",
                    "invalid",
                    artifact_path,
                    artifact_sha,
                    false,
                )?,
                invalid_clock(error.to_string()),
                vec![format!("clock evidence values are invalid: {error}")],
            ));
        }
    };
    let clock = match CalibrationEvidenceClock::try_new(
        document.source_domain,
        document.target_domain,
        document.method,
        calibration,
    ) {
        Ok(clock) => clock,
        Err(error) => {
            return Ok((
                CalibrationArtifact::try_new(
                    "clock_evidence",
                    "invalid",
                    artifact_path,
                    artifact_sha,
                    false,
                )?,
                invalid_clock(error.to_string()),
                vec![format!("clock evidence contract is invalid: {error}")],
            ));
        }
    };
    Ok((
        CalibrationArtifact::try_new(
            "clock_evidence",
            "registered",
            artifact_path,
            artifact_sha,
            true,
        )?,
        clock,
        Vec::new(),
    ))
}

fn build_frame_evidence(
    artifact_file: &ArtifactFile,
    input_path: &str,
    input_size: u64,
    observed_sha256: &str,
    root_frame: &str,
    front_frame: &str,
    rear_frame: &str,
) -> Result<(CalibrationArtifact, CalibrationEvidenceFrame, Vec<String>), Box<dyn Error>> {
    let required_frames = required_frames_for(front_frame, rear_frame);
    let Some(path) = artifact_file.path.as_deref() else {
        return Ok((
            CalibrationArtifact::try_new("frame_evidence", "not_registered", None, None, false)?,
            empty_frame("frame evidence file was not supplied", root_frame, required_frames)?,
            vec!["frame evidence file was not supplied".into()],
        ));
    };
    let Some(receipt) = artifact_file.receipt.as_ref() else {
        return Ok((
            CalibrationArtifact::try_new(
                "frame_evidence",
                "missing",
                Some(path.to_owned()),
                None,
                false,
            )?,
            empty_frame("frame evidence file is missing", root_frame, required_frames)?,
            vec![format!("frame evidence file '{}' is missing", path)],
        ));
    };
    let artifact_path = Some(path.to_owned());
    let artifact_sha = receipt.sha256.clone();
    let document: FrameEvidenceDocument = match read_json(Path::new(path)) {
        Ok(document) => document,
        Err(error) => {
            return Ok((
                CalibrationArtifact::try_new(
                    "frame_evidence",
                    "invalid",
                    artifact_path,
                    artifact_sha,
                    false,
                )?,
                empty_frame(
                    format!("frame evidence JSON is invalid: {error}"),
                    root_frame,
                    required_frames,
                )?,
                vec![format!("frame evidence JSON is invalid: {error}")],
            ));
        }
    };
    if document.schema != FRAME_SCHEMA || document.version != EVIDENCE_VERSION {
        return Ok((
            CalibrationArtifact::try_new(
                "frame_evidence",
                "invalid",
                artifact_path,
                artifact_sha,
                false,
            )?,
            empty_frame(
                "frame evidence schema or version is unsupported",
                root_frame,
                required_frames,
            )?,
            vec!["frame evidence schema or version is unsupported".into()],
        ));
    }
    if !source_binding_matches(&document.source, input_path, input_size, observed_sha256) {
        return Ok((
            CalibrationArtifact::try_new(
                "frame_evidence",
                "source_mismatch",
                artifact_path,
                artifact_sha,
                false,
            )?,
            empty_frame(
                "frame evidence source identity does not match the canonical input",
                root_frame,
                required_frames,
            )?,
            vec!["frame evidence source identity does not match the canonical input".into()],
        ));
    }
    if document.root_frame != root_frame {
        return Ok((
            CalibrationArtifact::try_new(
                "frame_evidence",
                "root_mismatch",
                artifact_path,
                artifact_sha,
                false,
            )?,
            empty_frame(
                "frame evidence root frame does not match the requested root",
                root_frame,
                required_frames,
            )?,
            vec!["frame evidence root frame does not match the requested root".into()],
        ));
    }
    let mut frame_ids = BTreeSet::from([document.root_frame.clone()]);
    frame_ids.extend(required_frames.values().cloned());
    let mut edges = Vec::with_capacity(document.edges.len());
    for edge in document.edges {
        frame_ids.insert(edge.parent_frame.clone());
        frame_ids.insert(edge.child_frame.clone());
        let transform = match FrameTransform::try_new(
            edge.parent_frame,
            edge.child_frame,
            edge.translation_m,
            edge.rotation_xyzw,
            edge.stamp_nanos,
            true,
            true,
        ) {
            Ok(transform) => transform,
            Err(error) => {
                return Ok((
                    CalibrationArtifact::try_new(
                        "frame_evidence",
                        "invalid",
                        artifact_path,
                        artifact_sha,
                        false,
                    )?,
                    empty_frame(
                        format!("frame evidence edge is invalid: {error}"),
                        root_frame,
                        required_frames,
                    )?,
                    vec![format!("frame evidence edge is invalid: {error}")],
                ));
            }
        };
        edges.push(transform);
    }
    let frame = match CalibrationEvidenceFrame::try_new(
        document.method,
        document.root_frame,
        required_frames,
        frame_ids.into_iter().collect(),
        edges,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Ok((
                CalibrationArtifact::try_new(
                    "frame_evidence",
                    "invalid",
                    artifact_path,
                    artifact_sha,
                    false,
                )?,
                empty_frame(
                    format!("frame evidence contract is invalid: {error}"),
                    root_frame,
                    required_frames_for(front_frame, rear_frame),
                )?,
                vec![format!("frame evidence contract is invalid: {error}")],
            ));
        }
    };
    let mut blockers = Vec::new();
    if !frame.registration_ready() {
        blockers.push("frame evidence graph does not connect root to both required sensors".into());
    }
    Ok((
        CalibrationArtifact::try_new(
            "frame_evidence",
            "registered",
            artifact_path,
            artifact_sha,
            true,
        )?,
        frame,
        blockers,
    ))
}

fn required_frames_for(front_frame: &str, rear_frame: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("front".to_owned(), front_frame.to_owned()),
        ("rear".to_owned(), rear_frame.to_owned()),
    ])
}

fn empty_frame(
    method: impl Into<String>,
    root_frame: &str,
    required_frames: BTreeMap<String, String>,
) -> Result<CalibrationEvidenceFrame, Box<dyn Error>> {
    Ok(CalibrationEvidenceFrame::try_new(
        method,
        root_frame,
        required_frames,
        Vec::new(),
        Vec::new(),
    )?)
}

fn unregistered_clock() -> CalibrationEvidenceClock {
    CalibrationEvidenceClock::try_new(
        "unknown",
        "uncalibrated",
        "not_registered",
        ClockCalibration::try_new(
            "not_registered",
            "PointCloud2 header stamp; no clock calibration applied",
            0,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .expect("static unregistered clock must be valid"),
    )
    .expect("static unregistered clock evidence must be valid")
}

fn invalid_clock(reason: impl Into<String>) -> CalibrationEvidenceClock {
    CalibrationEvidenceClock::try_new(
        "unknown",
        "uncalibrated",
        reason,
        ClockCalibration::try_new(
            "invalid",
            "PointCloud2 header stamp; clock evidence rejected",
            0,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .expect("static invalid clock must be valid"),
    )
    .expect("static invalid clock evidence must be valid")
}

fn inspect_artifact_file(path: Option<&Path>) -> Result<ArtifactFile, Box<dyn Error>> {
    let Some(path) = path else {
        return Ok(ArtifactFile { receipt: None, path: None });
    };
    let path_string = path.display().to_string();
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(ArtifactFile {
            receipt: Some(FileReceipt::from_path(ReceiptRole::Auxiliary, path)?),
            path: Some(path_string),
        }),
        Ok(_) => Ok(ArtifactFile { receipt: None, path: Some(path_string) }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(ArtifactFile { receipt: None, path: Some(path_string) })
        }
        Err(error) => Err(error.into()),
    }
}

fn source_binding_matches(
    source: &SourceBindingDocument,
    input_path: &str,
    input_size: u64,
    observed_sha256: &str,
) -> bool {
    source.path == input_path && source.size_bytes == input_size && source.sha256 == observed_sha256
}

fn inspect_readiness(
    readiness: &Value,
    input_path: &str,
    input_size: u64,
    observed_sha256: &str,
    blockers: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    if readiness.get("schema").and_then(Value::as_str) != Some(READINESS_SCHEMA) {
        push_blocker(blockers, "calibration readiness schema is unsupported");
    }
    if readiness.get("version").and_then(Value::as_u64) != Some(EVIDENCE_VERSION as u64) {
        push_blocker(blockers, "calibration readiness version is unsupported");
    }
    let path = string_at(readiness, &["input", "path"]);
    let sha256 = string_at(readiness, &["input", "sha256"]);
    let size_bytes =
        readiness.get("input").and_then(|input| input.get("size_bytes")).and_then(Value::as_u64);
    if path.as_deref() != Some(input_path)
        || sha256.as_deref() != Some(observed_sha256)
        || size_bytes != Some(input_size)
    {
        push_blocker(blockers, "calibration readiness receipt is not bound to the canonical input");
    }
    if !readiness.get("registration_ready").and_then(Value::as_bool).unwrap_or(false) {
        push_blocker(blockers, "calibration readiness registration is incomplete");
    }
    for blocker in string_array(readiness.get("blockers")) {
        push_blocker(blockers, format!("readiness: {blocker}"));
    }
    Ok(())
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut readiness = None;
    let mut clock_artifact = None;
    let mut frame_artifact = None;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut root_frame = None;
    let mut front_frame = None;
    let mut rear_frame = None;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--readiness" => readiness = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--clock-artifact" => {
                clock_artifact = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--frame-artifact" => {
                frame_artifact = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--root-frame" => root_frame = Some(next_value(&mut args, &flag)?),
            "--front-frame" => front_frame = Some(next_value(&mut args, &flag)?),
            "--rear-frame" => rear_frame = Some(next_value(&mut args, &flag)?),
            "--min-output-free-bytes" => {
                min_output_free_bytes = next_value(&mut args, &flag)?.parse()?
            }
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        input,
        readiness: readiness.ok_or("--readiness is required")?,
        clock_artifact,
        frame_artifact,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        root_frame: root_frame.ok_or("--root-frame is required")?,
        front_frame: front_frame.ok_or("--front-frame is required")?,
        rear_frame: rear_frame.ok_or("--rear-frame is required")?,
        min_output_free_bytes,
    })
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    let paths = [
        ("input", &config.input),
        ("readiness", &config.readiness),
        ("output", &config.output_dir),
    ];
    if paths.iter().any(|(_, path)| !path.is_absolute())
        || config.clock_artifact.as_ref().is_some_and(|path| !path.is_absolute())
        || config.frame_artifact.as_ref().is_some_and(|path| !path.is_absolute())
    {
        return Err("input, readiness, artifact, and output paths must be absolute".into());
    }
    for (label, value) in [
        ("root frame", config.root_frame.as_str()),
        ("front frame", config.front_frame.as_str()),
        ("rear frame", config.rear_frame.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label} must not be empty").into());
        }
    }
    if config.front_frame == config.rear_frame
        || config.root_frame == config.front_frame
        || config.root_frame == config.rear_frame
    {
        return Err("root, front, and rear frames must be distinct".into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
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

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    write_text_atomically(path, &format!("{}\n", serde_json::to_string_pretty(value)?))
}

fn write_text_atomically(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("output '{}' already exists", path.display()).into());
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
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

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
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

fn file_label(path: &Path) -> String {
    path.file_name().and_then(|name| name.to_str()).unwrap_or("input").to_owned()
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn render_dashboard(state: &CalibrationEvidenceState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#050a16;--panel:#101d36;--line:#2b416d;--muted:#91a2c4;--cyan:#6ce6ff;--green:#6ff0b1;--red:#ff7184;--amber:#ffd166;--violet:#ad8cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#342861 0,#080d1b 48%,#050a16 100%);color:#eef5ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}.title{font-size:30px;font-weight:800;margin-top:5px}.sub,.mono{color:var(--muted);font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:10px 15px;font-weight:800;white-space:nowrap}.good{color:var(--green);border-color:#287b59}.bad{color:var(--red);border-color:#8b3c50}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.panel{background:linear-gradient(145deg,#142341ee,#0d1529ee);border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 16px 35px #0004}.panel h2{font-size:11px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:24px;font-weight:800;color:var(--cyan)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:15px;padding:9px 0;border-bottom:1px solid #223253}.row:last-child{border-bottom:0}.edge{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:center;padding:10px;border:1px solid #30466e;border-radius:10px;background:#0b1730;margin:7px 0}.arrow{color:var(--cyan);font-size:20px;text-align:center}.blockers{margin:0;padding-left:20px}.blockers li{color:#ff9eaa;margin:7px 0}.empty{padding:24px;text-align:center;border:1px dashed #8b3c50;border-radius:12px;color:var(--red);letter-spacing:.12em}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:560px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:14px}}
</style></head><body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / 145J-B calibration evidence gate</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="mono"></div></article>
<article class="panel"><h2>Clock evidence</h2><div id="clock" class="metric"></div><div id="clockDetail" class="sub"></div></article>
<article class="panel"><h2>Frame graph</h2><div id="frame" class="metric"></div><div id="frameDetail" class="sub"></div></article>
<article class="panel"><h2>Registration</h2><div id="registration" class="metric"></div><div id="registrationDetail" class="sub"></div></article>
<article class="panel wide"><h2>Clock provenance</h2><div id="clockRows"></div></article>
<article class="panel wide"><h2>Required sensor paths</h2><div id="sensorPaths"></div></article>
<article class="panel full"><h2>Source-bound extrinsic edges</h2><div id="edges"></div></article>
<article class="panel wide"><h2>Registration blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel wide"><h2>Evidence receipts</h2><div id="artifacts"></div></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:280px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="evidence-state" type="application/json">__STATE_JSON__</script><script>
const state=JSON.parse(document.getElementById('evidence-state').textContent),q=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])),status=(id,value,ok)=>{q(id).textContent=value;q(id).className='metric '+(ok?'good':'bad')};
q('title').textContent=state.title;q('source').textContent=state.source.path;q('sha').textContent=state.source.observed_sha256;status('identity',state.source.identity_matches?'MATCH':'MISMATCH',state.source.identity_matches);status('clock',state.clock.registration_ready?'READY':'BLOCKED',state.clock.registration_ready);q('clockDetail').textContent=state.clock.source_domain+' → '+state.clock.target_domain+' · '+state.clock.calibration.sample_count+' samples';status('frame',state.frame.graph_ready?'READY':'BLOCKED',state.frame.graph_ready);q('frameDetail').textContent=state.frame.root_frame+' · '+state.frame.edges.length+' edges';status('registration',state.registration_ready?'READY':'BLOCKED',state.registration_ready);q('registrationDetail').textContent=state.registration_ready?'source-bound evidence admitted':'mapping remains fail-closed';q('admission').textContent=state.registration_ready?'REGISTRATION READY':'REGISTRATION BLOCKED';q('admission').className='badge '+(state.registration_ready?'good':'bad');
q('clockRows').innerHTML='<div class="row"><span>method</span><span>'+esc(state.clock.method)+'</span></div><div class="row"><span>time basis</span><span>'+esc(state.clock.calibration.time_basis)+'</span></div><div class="row"><span>median offset</span><span>'+String(state.clock.calibration.median_offset_nanos??'—')+' ns</span></div><div class="row"><span>p95 / uncertainty</span><span>'+String(state.clock.calibration.p95_abs_offset_nanos??'—')+' / '+String(state.clock.calibration.uncertainty_nanos??'—')+' ns</span></div>';
q('sensorPaths').innerHTML=Object.entries(state.frame.required_frames).map(([role,frame])=>'<div class="row"><span>'+esc(role)+'</span><span>'+esc(state.frame.root_frame)+' → '+esc(frame)+'</span></div>').join('');q('edges').innerHTML=state.frame.edges.length?state.frame.edges.map(e=>'<div class="edge"><span>'+esc(e.parent_frame)+'</span><span class="arrow">→</span><span>'+esc(e.child_frame)+'</span></div>').join(''):'<div class="empty">NO SOURCE-BOUND EDGES</div>';q('blockers').innerHTML=state.blockers.map(b=>'<li>'+esc(b)+'</li>').join('')||'<li class="good">All registration gates passed</li>';q('artifacts').innerHTML=[state.clock_artifact,state.frame_artifact].map(a=>'<div class="row"><span>'+esc(a.kind)+'</span><span>'+esc(a.status)+(a.source_bound?' · bound':'')+'</span></div>').join('');q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
}

fn usage() -> String {
    "usage: rosbag2_calibration_evidence INPUT_DB3 --readiness ABSOLUTE_READINESS_JSON \
     --output-dir ABSOLUTE_OUTPUT_DIR --expected-input-sha256 SHA256 \
     --root-frame FRAME --front-frame FRAME --rear-frame FRAME \
     [--clock-artifact ABSOLUTE_JSON] [--frame-artifact ABSOLUTE_JSON] \
     [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn args() -> Vec<String> {
        [
            "/media/input.db3",
            "--readiness",
            "/media/readiness.json",
            "--output-dir",
            "/media/evidence",
            "--expected-input-sha256",
            SHA,
            "--root-frame",
            "base_link",
            "--front-frame",
            "lidar_front",
            "--rear-frame",
            "lidar_rear",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn parses_explicit_frame_contract() {
        let config = parse_args(args()).unwrap();
        assert_eq!(config.root_frame, "base_link");
        assert_eq!(config.front_frame, "lidar_front");
        assert!(config.clock_artifact.is_none());
        validate_config(&config).unwrap();
    }

    #[test]
    fn rejects_relative_paths_and_colliding_frames() {
        let mut relative = args();
        relative[0] = "input.db3".into();
        assert!(parse_args(relative).and_then(|config| validate_config(&config)).is_err());

        let mut duplicate = args();
        let rear_index = duplicate.iter().position(|value| value == "lidar_rear").unwrap();
        duplicate[rear_index] = "lidar_front".into();
        let config = parse_args(duplicate).unwrap();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn source_binding_requires_path_size_and_sha() {
        let source = SourceBindingDocument {
            path: "/media/input.db3".into(),
            size_bytes: 42,
            sha256: SHA.into(),
        };
        assert!(source_binding_matches(&source, "/media/input.db3", 42, SHA));
        assert!(!source_binding_matches(&source, "/media/other.db3", 42, SHA));
        assert!(!source_binding_matches(&source, "/media/input.db3", 43, SHA));
    }

    #[test]
    fn missing_artifact_file_is_recorded_without_fabricating_evidence() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.json");
        let artifact = inspect_artifact_file(Some(&path)).unwrap();
        assert!(artifact.receipt.is_none());
        assert_eq!(artifact.path, Some(path.display().to_string()));
    }
}

//! Run a bounded deterministic semantic overlay over an existing E2E map.
//!
//! The command reuses a receipt-backed glTF artifact and runs only the
//! lightweight deterministic semantic mock profile. Model input/output
//! tensors are CPU-owned and every transfer is accounted for explicitly.
//! Source, frame, clock, TF, and production-model admission remain separate
//! gates so an attractive overlay cannot silently become calibrated mapping.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use spatialrust_ai::{
    CopyPolicy, InferenceBackend, MockInferenceBackend, MockProfile, ModelSource, NamedTensors,
    RunOptions, SessionOptions,
};
use spatialrust_interchange::decode_triangle_mesh_gltf_json;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_scene::TriangleMesh;
use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};
use spatialrust_viewer::{
    ReplayArtifact, SemanticOverlayClass, SemanticOverlayEntity, SemanticOverlayModel,
    SemanticOverlayState, SemanticOverlaySummary, StudioSource, SEMANTIC_CONFIDENCE_SCALE,
};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const DEFAULT_MAX_POINTS: usize = 4_096;
const MAX_MAX_POINTS: usize = 65_536;
const STATE_FILE: &str = "semantic-overlay.json";
const HTML_FILE: &str = "semantic-overlay.html";
const MANIFEST_FILE: &str = "semantic-overlay.manifest.json";

#[derive(Debug)]
struct Config {
    run_dir: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    expected_frame_id: String,
    max_points: usize,
    min_output_free_bytes: u64,
}

struct LoadedMap {
    source: StudioSource,
    mesh: TriangleMesh,
    input_receipt: FileReceipt,
    map_receipt: FileReceipt,
    input_artifact: ReplayArtifact,
    map_artifact: ReplayArtifact,
    receipt_path: PathBuf,
    manifest_path: PathBuf,
    frame_id: String,
    time_basis: String,
}

#[derive(Debug, Deserialize)]
struct E2eReceipt {
    schema: String,
    version: u32,
    input: String,
    sync: SyncReceipt,
    tsdf: TsdfReceipt,
    interchange: InterchangeReceipt,
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
    vertices: u64,
    indices: u64,
}

#[derive(Clone, Copy)]
struct ClassDefinition {
    id: u32,
    label: &'static str,
    color_rgb: [u8; 3],
}

const CLASS_DEFINITIONS: [ClassDefinition; 3] = [
    ClassDefinition { id: 0, label: "ground", color_rgb: [54, 211, 153] },
    ClassDefinition { id: 1, label: "structure", color_rgb: [81, 150, 255] },
    ClassDefinition { id: 2, label: "object", color_rgb: [255, 143, 83] },
];

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-semantic-overlay: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    validate_config(&config)?;
    let output_parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    let preflight = StoragePreflight::check(output_parent, config.min_output_free_bytes)?;
    if config.output_dir.exists() {
        return Err(format!(
            "semantic overlay output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }

    let loaded = load_map(&config.run_dir, &config.expected_sha256)?;
    let source_identity_match = loaded.source.identity_matches;
    let frame_identity_match = loaded.frame_id == config.expected_frame_id;
    let admission_identity_match = source_identity_match && frame_identity_match;
    let (model, classes, entities) =
        if admission_identity_match { run_model(&loaded.mesh, &config)? } else { empty_model()? };

    let mut blockers = Vec::new();
    if !source_identity_match {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch; semantic inference withheld (expected {}, observed {})",
                loaded.source.expected_sha256, loaded.source.observed_sha256
            ),
        );
    }
    if !frame_identity_match {
        push_blocker(
            &mut blockers,
            format!(
                "frame mismatch; expected '{}', observed '{}'",
                config.expected_frame_id, loaded.frame_id
            ),
        );
    }
    if !admission_identity_match {
        push_blocker(&mut blockers, "overlay admission requires source and frame identity");
    } else {
        push_blocker(&mut blockers, format!("time basis: {}", loaded.time_basis));
        push_blocker(
            &mut blockers,
            "clock calibration not applied; overlay remains in the header-stamp domain",
        );
        push_blocker(&mut blockers, "TF/frame composition not applied; overlay is inspection-only");
        push_blocker(
            &mut blockers,
            "production model receipt absent; deterministic mock is visualization-only",
        );
        push_blocker(
            &mut blockers,
            "mapping admission requires source-bound calibrated semantic evidence",
        );
    }

    let summary = build_summary(
        loaded.mesh.vertex_count(),
        entities.len(),
        &entities,
        source_identity_match,
        frame_identity_match,
    )?;
    let state = SemanticOverlayState::try_new(
        format!("AI Semantic Overlay — {}", loaded.source.label),
        loaded.source.clone(),
        loaded.frame_id.clone(),
        config.expected_frame_id.clone(),
        loaded.time_basis.clone(),
        model,
        classes,
        entities,
        vec![loaded.input_artifact.clone(), loaded.map_artifact.clone()],
        summary,
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
    manifest.entries.push(loaded.input_receipt);
    manifest.entries.push(loaded.map_receipt);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, &loaded.receipt_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, &loaded.manifest_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Semantic overlay: {} (overlay_ready={}, mapping_admitted={})",
        state_path.display(),
        state.overlay_ready,
        state.mapping_admitted
    );
    println!("Semantic overlay dashboard: {}", html_path.display());
    println!(
        "Semantic overlay manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.overlay_ready {
        return Err("semantic overlay failed its source/frame admission checks".into());
    }
    Ok(())
}

fn load_map(run_dir: &Path, expected_sha256: &str) -> Result<LoadedMap, Box<dyn Error>> {
    let receipt_path = run_dir.join("rosbag2.e2e.receipt.json");
    let manifest_path = run_dir.join("rosbag2.e2e.manifest.json");
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
        return Err(format!("E2E input path '{}' is not absolute", receipt.input).into());
    }
    let input_receipt = manifest_entry(&manifest, ReceiptRole::Input, &input_path)?;
    let observed_sha256 =
        input_receipt.sha256.clone().ok_or("E2E input manifest entry has no SHA-256")?;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        receipt.input.clone(),
        expected_sha256,
        observed_sha256.clone(),
        observed_sha256 == expected_sha256,
    )?;
    let input_artifact = ReplayArtifact::try_new(
        "input",
        input_path.display().to_string(),
        input_receipt.size_bytes.ok_or("E2E input receipt has no size")?,
        observed_sha256,
    )?;

    let map_path = PathBuf::from(&receipt.interchange.path);
    if !map_path.is_absolute() {
        return Err(
            format!("Map artifact path '{}' is not absolute", receipt.interchange.path).into()
        );
    }
    let map_receipt = manifest_entry(&manifest, ReceiptRole::Output, &map_path)?;
    let map_text = fs::read_to_string(&map_path)?;
    let mesh = decode_triangle_mesh_gltf_json(&map_text)?;
    let vertex_count = u64::try_from(mesh.vertex_count())?;
    let triangle_count = u64::try_from(mesh.triangle_count())?;
    if vertex_count != receipt.tsdf.mesh_vertices
        || triangle_count != receipt.tsdf.mesh_triangles
        || vertex_count != receipt.interchange.vertices
        || u64::try_from(mesh.indices.len())? != receipt.interchange.indices
    {
        return Err(format!(
            "Map artifact counts disagree with receipt in '{}'",
            receipt_path.display()
        )
        .into());
    }
    let map_artifact = ReplayArtifact::try_new(
        "map",
        map_path.display().to_string(),
        map_receipt.size_bytes.ok_or("Map receipt has no size")?,
        map_receipt.sha256.clone().ok_or("Map receipt has no SHA-256")?,
    )?;
    Ok(LoadedMap {
        source,
        mesh,
        input_receipt,
        map_receipt,
        input_artifact,
        map_artifact,
        receipt_path,
        manifest_path,
        frame_id: receipt.tsdf.frame_id,
        time_basis: receipt.sync.time_basis,
    })
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

fn run_model(
    mesh: &TriangleMesh,
    config: &Config,
) -> Result<
    (SemanticOverlayModel, Vec<SemanticOverlayClass>, Vec<SemanticOverlayEntity>),
    Box<dyn Error>,
> {
    let positions = mesh.positions.chunks_exact(3).collect::<Vec<_>>();
    if positions.len() != mesh.vertex_count() || positions.is_empty() {
        return Err("semantic overlay map contains no complete finite vertices".into());
    }
    let bounds = position_bounds(&positions)?;
    let indices = sample_indices(positions.len(), config.max_points)?;
    let count = indices.len();
    let mut features = Vec::with_capacity(count.checked_mul(4).ok_or("feature count overflow")?);
    for axis in 0..3 {
        for &index in &indices {
            features.push(normalize(positions[index][axis], bounds[axis], bounds[axis + 3]));
        }
    }
    for &index in &indices {
        let x = normalize(positions[index][0], bounds[0], bounds[3]);
        let y = normalize(positions[index][1], bounds[1], bounds[4]);
        features.push((x.mul_add(x, y * y)).sqrt().min(1.0));
    }
    let input_bytes = u64::try_from(features.len().checked_mul(4).ok_or("input bytes overflow")?)?;
    let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![1, 4, count], Device::CPU);
    let input = TensorBuffer::try_from_f32(features, descriptor)?;
    let mut inputs = NamedTensors::new();
    inputs.insert("features", input)?;
    let backend = MockInferenceBackend;
    let mut session = backend.create_session(
        &ModelSource::Mock(MockProfile::SemanticClasses),
        &SessionOptions { deterministic: true, ..SessionOptions::default() },
    )?;
    let outputs = session.run_with_options(
        inputs,
        RunOptions { input_copy: CopyPolicy::Forbid, output_copy: CopyPolicy::Allow },
    )?;
    let class_ids = outputs
        .get("class_ids")
        .and_then(TensorBuffer::shared_u32)
        .ok_or("semantic model did not return aligned u32 class IDs")?;
    let confidences = outputs
        .get("confidence")
        .and_then(TensorBuffer::shared_f32)
        .ok_or("semantic model did not return aligned f32 confidence")?;
    if class_ids.len() != count || confidences.len() != count {
        return Err("semantic model output count disagrees with sampled input".into());
    }
    let mut entities = Vec::with_capacity(count);
    for ((&source_index, &class_id), &confidence) in
        indices.iter().zip(class_ids.iter()).zip(confidences.iter())
    {
        let definition = class_definition(class_id)?;
        entities.push(SemanticOverlayEntity::try_new(
            format!("semantic:{source_index}"),
            u64::try_from(source_index)?,
            class_id,
            definition.label,
            quantize_confidence(confidence)?,
            quantized_position(positions[source_index])?,
        )?);
    }
    let input_descriptor_bytes = input_bytes;
    let output_bytes = u64::try_from(
        count
            .checked_mul(4)
            .ok_or("class output bytes overflow")?
            .checked_add(count.checked_mul(4).ok_or("confidence output bytes overflow")?)
            .ok_or("output bytes overflow")?,
    )?;
    let model = SemanticOverlayModel::try_new(
        session.model_info().name.clone().unwrap_or_else(|| "mock-semantic-classes".into()),
        session.backend_name(),
        "deterministic MockProfile::SemanticClasses; CPU tensors; no device copies",
        true,
        4,
        u32::try_from(CLASS_DEFINITIONS.len())?,
        input_descriptor_bytes,
        output_bytes,
        0,
        0,
    )?;
    Ok((model, class_legend(&entities)?, entities))
}

fn empty_model() -> Result<
    (SemanticOverlayModel, Vec<SemanticOverlayClass>, Vec<SemanticOverlayEntity>),
    Box<dyn Error>,
> {
    let model = SemanticOverlayModel::try_new(
        "mock-semantic-classes",
        "mock",
        "not run: source/frame admission failed",
        true,
        4,
        u32::try_from(CLASS_DEFINITIONS.len())?,
        0,
        0,
        0,
        0,
    )?;
    Ok((model, class_legend(&[])?, Vec::new()))
}

fn class_legend(
    entities: &[SemanticOverlayEntity],
) -> Result<Vec<SemanticOverlayClass>, Box<dyn Error>> {
    let mut statistics = BTreeMap::<u32, (u64, u64, u32)>::new();
    for entity in entities {
        let entry = statistics.entry(entity.class_id).or_default();
        entry.0 = entry.0.checked_add(1).ok_or("class count overflow")?;
        entry.1 = entry
            .1
            .checked_add(u64::from(entity.confidence_million))
            .ok_or("class confidence sum overflow")?;
        entry.2 = entry.2.max(entity.confidence_million);
    }
    CLASS_DEFINITIONS
        .iter()
        .map(|definition| {
            let (count, sum, max) = statistics.get(&definition.id).copied().unwrap_or_default();
            let mean = u32::try_from(sum.checked_div(count).unwrap_or(0))?;
            SemanticOverlayClass::try_new(
                definition.id,
                definition.label,
                definition.color_rgb,
                count,
                mean,
                max,
            )
            .map_err(Into::into)
        })
        .collect()
}

fn class_definition(class_id: u32) -> Result<ClassDefinition, Box<dyn Error>> {
    CLASS_DEFINITIONS
        .iter()
        .copied()
        .find(|definition| definition.id == class_id)
        .ok_or_else(|| format!("semantic model returned unknown class ID {class_id}").into())
}

fn build_summary(
    input_point_count: usize,
    sampled_point_count: usize,
    entities: &[SemanticOverlayEntity],
    source_identity_match: bool,
    frame_identity_match: bool,
) -> Result<SemanticOverlaySummary, Box<dyn Error>> {
    let mut confidence =
        entities.iter().map(|entity| entity.confidence_million).collect::<Vec<_>>();
    confidence.sort_unstable();
    let sum = confidence.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(u64::from(*value)).ok_or("confidence sum overflow")
    })?;
    let count = u64::try_from(entities.len())?;
    let mean = u32::try_from(sum.checked_div(count).unwrap_or(0))?;
    let p95 = confidence
        .get((confidence.len().saturating_mul(95)).div_ceil(100).saturating_sub(1))
        .copied()
        .unwrap_or(0);
    let coverage = if input_point_count == 0 {
        0
    } else {
        u32::try_from(
            (u128::from(sampled_point_count as u64) * u128::from(SEMANTIC_CONFIDENCE_SCALE)
                / u128::from(input_point_count as u64))
            .min(u128::from(SEMANTIC_CONFIDENCE_SCALE)),
        )?
    };
    let class_count = entities.iter().map(|entity| entity.class_id).collect::<BTreeSet<_>>().len();
    SemanticOverlaySummary::try_new(
        u64::try_from(input_point_count)?,
        u64::try_from(sampled_point_count)?,
        count,
        count,
        u64::try_from(class_count)?,
        mean,
        p95,
        coverage,
        source_identity_match,
        frame_identity_match,
        false,
    )
    .map_err(Into::into)
}

fn sample_indices(count: usize, max_points: usize) -> Result<Vec<usize>, Box<dyn Error>> {
    if count == 0 || max_points == 0 {
        return Err("semantic overlay sampling requires positive counts".into());
    }
    let step = count.saturating_add(max_points.saturating_sub(1)) / max_points;
    let step = step.max(1);
    Ok((0..count).step_by(step).collect())
}

fn position_bounds(positions: &[&[f32]]) -> Result<[f32; 6], Box<dyn Error>> {
    let first = positions.first().ok_or("semantic overlay map has no positions")?;
    if first.len() != 3 || first.iter().any(|value| !value.is_finite()) {
        return Err("semantic overlay map contains a non-finite position".into());
    }
    let mut bounds = [first[0], first[1], first[2], first[0], first[1], first[2]];
    for position in positions {
        if position.len() != 3 || position.iter().any(|value| !value.is_finite()) {
            return Err("semantic overlay map contains a non-finite position".into());
        }
        for axis in 0..3 {
            bounds[axis] = bounds[axis].min(position[axis]);
            bounds[axis + 3] = bounds[axis + 3].max(position[axis]);
        }
    }
    Ok(bounds)
}

fn normalize(value: f32, minimum: f32, maximum: f32) -> f32 {
    if maximum == minimum {
        0.5
    } else {
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
    }
}

fn quantize_confidence(value: f32) -> Result<u32, Box<dyn Error>> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("semantic model emitted confidence outside [0, 1]".into());
    }
    Ok((f64::from(value) * f64::from(SEMANTIC_CONFIDENCE_SCALE)).round() as u32)
}

fn quantized_position(position: &[f32]) -> Result<[i64; 3], Box<dyn Error>> {
    if position.len() != 3 || position.iter().any(|value| !value.is_finite()) {
        return Err("semantic overlay position is not finite".into());
    }
    let mut output = [0_i64; 3];
    for (axis, value) in position.iter().enumerate() {
        let scaled = f64::from(*value) * 1_000_000.0;
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err("semantic overlay position exceeds micrometre range".into());
        }
        output[axis] = scaled.round() as i64;
    }
    Ok(output)
}

fn render_dashboard(state: &SemanticOverlayState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#070d18;--panel:#101d31;--line:#284664;--muted:#8da8bf;--cyan:#63e6ff;--green:#64f2a3;--red:#ff7180;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#264873 0,#070d18 50%);color:#eff8ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.title{font-size:30px;font-weight:780;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:760;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(17,38,65,.97),rgba(8,16,29,.97));border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 14px 32px #0004}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:25px;font-weight:780;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #1c3552;padding:9px 0}.row:last-child{border-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.danger{color:var(--red)}.warning{color:var(--amber)}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.contract{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.chip{border:1px solid #254363;border-radius:10px;padding:10px}.chip b{display:block;color:var(--cyan);font-size:12px}.palette{display:flex;flex-wrap:wrap;gap:9px}.swatch{display:flex;gap:8px;align-items:center;border:1px solid #254363;border-radius:999px;padding:7px 10px}.dot{width:12px;height:12px;border-radius:50%}canvas{display:block;width:100%;height:auto;max-height:600px;background:#07111f;border:1px solid #254363;border-radius:10px}.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}.legend span{display:flex;gap:6px;align-items:center}.source{overflow-wrap:anywhere}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}.contract{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / AI semantic overlay</div><div class="title" id="title">__TITLE__</div><div class="sub source" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Overlay gate</h2><div id="overlay" class="metric"></div><div id="overlayDetail" class="small"></div></article>
<article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Predictions</h2><div id="predictions" class="metric"></div><div id="coverage" class="small"></div></article>
<article class="panel"><h2>Confidence</h2><div id="confidence" class="metric"></div><div id="p95" class="small"></div></article>
<article class="panel wide"><h2>Class palette</h2><div id="palette" class="palette"></div></article>
<article class="panel wide"><h2>Model contract / explicit transfers</h2><div id="contract" class="contract"></div></article>
<article class="panel wide"><h2>Top-down semantic point cloud</h2><canvas id="plot" width="900" height="620"></canvas><div id="plotDetail" class="small" style="margin-top:9px"></div><div id="legend" class="legend"></div></article>
<article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel full"><h2>Source / frame / calibration</h2><div id="identity"></div></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:300px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="semantic-overlay-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('semantic-overlay-state').textContent),q=id=>document.getElementById(id),fmt=n=>n.toLocaleString(),pct=n=>(n/10000).toFixed(2)+'%',esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const classes=new Map(state.classes.map(c=>[c.class_id,c])),s=state.summary,m=state.model;
q('title').textContent=state.title;q('source').textContent=state.source.path+' · '+state.source.observed_sha256;q('admission').textContent=state.overlay_ready?'OVERLAY READY':'OVERLAY BLOCKED';q('admission').className='badge '+(state.overlay_ready?'ok':'blocked');
q('overlay').textContent=state.overlay_ready?'READY':'BLOCKED';q('overlay').className='metric '+(state.overlay_ready?'':'danger');q('overlayDetail').textContent=state.source.identity_matches?'source SHA matched':'source SHA mismatch';
q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';
q('predictions').textContent=fmt(s.entity_count);q('coverage').textContent=fmt(s.sampled_point_count)+' sampled / '+fmt(s.input_point_count)+' input · '+pct(s.coverage_million);
q('confidence').textContent=pct(s.mean_confidence_million);q('p95').textContent='p95 '+pct(s.p95_confidence_million)+' · '+fmt(s.class_count)+' classes';
q('palette').innerHTML=state.classes.map(c=>'<div class="swatch"><i class="dot" style="background:rgb('+c.color_rgb.join(',')+')"></i><span>'+esc(c.label)+' · '+fmt(c.entity_count)+'</span></div>').join('');
q('contract').innerHTML='<div class="chip"><b>model</b><span class="mono">'+esc(m.model_id)+'</span></div><div class="chip"><b>backend / runtime</b><span class="mono">'+esc(m.backend)+' · '+esc(m.runtime)+'</span></div><div class="chip"><b>host tensors</b><span class="mono">'+fmt(m.input_host_bytes)+' B in / '+fmt(m.output_host_bytes)+' B out</span></div><div class="chip"><b>device copies</b><span class="mono">'+fmt(m.device_upload_bytes)+' B upload / '+fmt(m.device_readback_bytes)+' B readback</span></div>';
q('identity').innerHTML='<div class="row"><span>source identity</span><span class="mono '+(state.source.identity_matches?'':'danger')+'">'+(state.source.identity_matches?'MATCH':'MISMATCH')+' · expected '+esc(state.source.expected_sha256)+'</span></div><div class="row"><span>frame identity</span><span class="mono '+(s.frame_identity_match?'':'danger')+'">'+(s.frame_identity_match?'MATCH':'MISMATCH')+' · '+esc(state.frame_id)+' / expected '+esc(state.expected_frame_id)+'</span></div><div class="row"><span>time basis</span><span class="mono">'+esc(state.time_basis)+'</span></div><div class="row"><span>calibration / mapping</span><span class="mono '+(state.mapping_admitted?'':'warning')+'">'+(state.mapping_admitted?'APPLIED':'NOT APPLIED')+'</span></div>';
q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="ok">All gates passed</li>';
const canvas=q('plot'),ctx=canvas.getContext('2d'),pad=42,points=state.entities;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#294763';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,canvas.height-pad);ctx.lineTo(canvas.width-pad,canvas.height-pad);ctx.stroke();const xs=points.map(e=>e.centroid_um[0]),ys=points.map(e=>e.centroid_um[1]),minx=Math.min(...xs,0),maxx=Math.max(...xs,1),miny=Math.min(...ys,0),maxy=Math.max(...ys,1);for(const e of points){const x=pad+(e.centroid_um[0]-minx)/(maxx-minx||1)*(canvas.width-2*pad),y=canvas.height-pad-(e.centroid_um[1]-miny)/(maxy-miny||1)*(canvas.height-2*pad),c=classes.get(e.class_id),alpha=.25+.75*e.confidence_million/1000000;ctx.fillStyle='rgba('+c.color_rgb.join(',')+','+alpha.toFixed(3)+')';ctx.beginPath();ctx.arc(x,y,2.2,0,Math.PI*2);ctx.fill();}q('plotDetail').textContent=points.length?'X/Y projection · opacity encodes confidence · coordinates are micrometres':'No admitted predictions to render';
q('legend').innerHTML=state.classes.map(c=>'<span><i class="dot" style="background:rgb('+c.color_rgb.join(',')+')"></i>'+esc(c.label)+'</span>').join('');q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
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
    let file_name = path.file_name().ok_or("output path has no file name")?;
    let temporary = path.with_file_name(format!(".{}.tmp", file_name.to_string_lossy()));
    if temporary.exists() {
        return Err(format!("temporary output '{}' already exists", temporary.display()).into());
    }
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.run_dir.is_absolute() || !config.output_dir.is_absolute() {
        return Err("run directory and --output-dir must be absolute".into());
    }
    if !config.run_dir.is_dir() {
        return Err(
            format!("run directory '{}' is not a directory", config.run_dir.display()).into()
        );
    }
    if config.output_dir == Path::new("/") {
        return Err("--output-dir must not be the filesystem root".into());
    }
    let parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    if !parent.is_dir() {
        return Err(format!("output parent '{}' is not a directory", parent.display()).into());
    }
    if config.expected_frame_id.trim().is_empty() {
        return Err("--expected-frame-id must not be empty".into());
    }
    if config.max_points == 0 || config.max_points > MAX_MAX_POINTS {
        return Err(format!("--max-points must be between 1 and {MAX_MAX_POINTS}").into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let run_dir = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut expected_frame_id = None;
    let mut max_points = DEFAULT_MAX_POINTS;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--expected-frame-id" => expected_frame_id = Some(next_value(&mut args, &flag)?),
            "--max-points" => max_points = next_value(&mut args, &flag)?.parse()?,
            "--min-output-free-bytes" => {
                min_output_free_bytes = next_value(&mut args, &flag)?.parse()?
            }
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        run_dir: PathBuf::from(run_dir),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        expected_frame_id: expected_frame_id.ok_or("--expected-frame-id is required")?,
        max_points,
        min_output_free_bytes,
    })
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

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn usage() -> String {
    "usage: rosbag2_semantic_overlay RUN_DIR \
     --output-dir ABSOLUTE_OUTPUT_DIR --expected-input-sha256 SHA256 \
     --expected-frame-id FRAME [--max-points N] [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, sample_indices, validate_sha256};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_overlay_options() {
        let config = parse_args(
            [
                "/media/e2e",
                "--output-dir",
                "/media/results/overlay",
                "--expected-input-sha256",
                SHA,
                "--expected-frame-id",
                "lidar_front",
                "--max-points",
                "1024",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.max_points, 1024);
        assert_eq!(config.expected_frame_id, "lidar_front");
    }

    #[test]
    fn sampling_is_bounded_and_hash_is_strict() {
        let indices = sample_indices(10_000, 128).unwrap();
        assert!(indices.len() <= 128);
        assert_eq!(indices[0], 0);
        assert!(validate_sha256(SHA).is_ok());
        assert!(validate_sha256("BAD").is_err());
    }
}

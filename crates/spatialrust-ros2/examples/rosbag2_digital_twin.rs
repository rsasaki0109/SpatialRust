//! Export a receipt-backed glTF/USD Digital Twin bundle.
//!
//! The command consumes an existing SpatialRust E2E run directory and writes
//! only to an explicit external output directory. The glTF payload is copied
//! byte-for-byte. The USDA file is an ASCII companion/reference layer; this
//! example does not add an OpenUSD runtime or apply an implicit transform.
//!
//! Usage:
//!   rosbag2_digital_twin E2E_RUN_DIR --output-dir ABSOLUTE_OUTPUT_DIR \
//!     --expected-input-sha256 SHA256 --expected-frame-id FRAME
//!
//! An optional semantic overlay is attached only when its own source and frame
//! receipts match the same canonical identity.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use spatialrust_interchange::decode_triangle_mesh_gltf_json;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_viewer::{
    DigitalTwinAsset, DigitalTwinState, DigitalTwinSummary, ReplayArtifact, SemanticOverlayState,
    StudioSource,
};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const STATE_FILE: &str = "digital-twin.json";
const HTML_FILE: &str = "digital-twin.html";
const GLTF_FILE: &str = "digital-twin.gltf";
const USDA_FILE: &str = "digital-twin.usda";
const MANIFEST_FILE: &str = "digital-twin.manifest.json";

#[derive(Debug)]
struct Config {
    run_dir: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    expected_frame_id: String,
    semantic_overlay: Option<PathBuf>,
    min_output_free_bytes: u64,
}

struct LoadedMap {
    source: StudioSource,
    frame_id: String,
    time_basis: String,
    vertex_count: u64,
    triangle_count: u64,
    map_path: PathBuf,
    input_receipt: FileReceipt,
    map_receipt: FileReceipt,
    receipt_path: PathBuf,
    manifest_path: PathBuf,
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

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-digital-twin: {error}");
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
            "Digital Twin output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }

    let loaded = load_map(&config.run_dir, &config.expected_sha256)?;
    let source_identity_match = loaded.source.identity_matches;
    let frame_identity_match = loaded.frame_id == config.expected_frame_id;
    let export_admitted = source_identity_match && frame_identity_match;

    fs::create_dir_all(&config.output_dir)?;

    let mut blockers = Vec::new();
    if !source_identity_match {
        push_blocker(
            &mut blockers,
            "canonical input SHA-256 mismatch; Digital Twin export withheld",
        );
    }
    if !frame_identity_match {
        push_blocker(
            &mut blockers,
            "canonical map frame does not match the requested frame; export withheld",
        );
    }

    let mut assets = Vec::new();
    if export_admitted {
        let gltf_path = config.output_dir.join(GLTF_FILE);
        copy_identity_file(&loaded.map_path, &gltf_path)?;
        let gltf_receipt = FileReceipt::from_path(ReceiptRole::Output, &gltf_path)?;
        let gltf_artifact = artifact_from_receipt("gltf", &gltf_receipt)?;
        assets.push(DigitalTwinAsset::try_new(
            "canonical-mesh-gltf",
            "gltf",
            gltf_artifact,
            &loaded.frame_id,
            loaded.vertex_count,
            loaded.triangle_count,
            true,
            "byte-identical copy of canonical glTF",
        )?);

        let usda_path = config.output_dir.join(USDA_FILE);
        let usda = render_usda(
            &loaded.source.expected_sha256,
            &loaded.frame_id,
            &loaded.time_basis,
            loaded.vertex_count,
            loaded.triangle_count,
        );
        write_text_atomically(&usda_path, &usda)?;
        let usda_receipt = FileReceipt::from_path(ReceiptRole::Output, &usda_path)?;
        let usda_artifact = artifact_from_receipt("usda", &usda_receipt)?;
        assets.push(DigitalTwinAsset::try_new(
            "canonical-mesh-usda-companion",
            "usda",
            usda_artifact,
            &loaded.frame_id,
            loaded.vertex_count,
            loaded.triangle_count,
            true,
            "ASCII USDA companion layer referencing the glTF asset",
        )?);

        push_blocker(&mut blockers, format!("time basis: {}", loaded.time_basis));
        push_blocker(
            &mut blockers,
            "USDA is an explicit companion/reference layer; no OpenUSD runtime conversion was performed",
        );
        push_blocker(
            &mut blockers,
            "clock calibration not applied; Digital Twin remains in the source time domain",
        );
        push_blocker(
            &mut blockers,
            "TF/frame composition not applied; geometry is inspection-only in the source frame",
        );
        push_blocker(&mut blockers, "mapping admission requires source-bound calibrated evidence");
    } else {
        push_blocker(
            &mut blockers,
            "no glTF or USDA asset is emitted until source and frame identity both match",
        );
    }

    let mut semantic_layer = None;
    let mut semantic_receipt = None;
    if let Some(path) = &config.semantic_overlay {
        let receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, path)?;
        if export_admitted {
            match attach_semantic_layer(path, &config.expected_sha256, &config.expected_frame_id) {
                Ok(artifact) => semantic_layer = Some(artifact),
                Err(reason) => push_blocker(&mut blockers, reason),
            }
        } else {
            push_blocker(
                &mut blockers,
                "semantic overlay withheld because the canonical Digital Twin gate is blocked",
            );
        }
        semantic_receipt = Some(receipt);
    }

    let summary = DigitalTwinSummary::try_new(
        loaded.vertex_count,
        loaded.triangle_count,
        if export_admitted { loaded.vertex_count } else { 0 },
        if export_admitted { loaded.triangle_count } else { 0 },
        if export_admitted { loaded.vertex_count } else { 0 },
        if export_admitted { loaded.triangle_count } else { 0 },
        u64::try_from(assets.len())?,
        semantic_layer.is_some(),
        source_identity_match,
        frame_identity_match,
        export_admitted,
        false,
    )?;
    let state = DigitalTwinState::try_new(
        "SpatialRust Digital Twin",
        loaded.source,
        loaded.frame_id,
        config.expected_frame_id,
        loaded.time_basis,
        assets,
        semantic_layer,
        summary,
        blockers,
    )?;
    state.validate()?;

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
    if let Some(receipt) = semantic_receipt {
        manifest.entries.push(receipt);
    }
    if export_admitted {
        manifest
            .entries
            .push(FileReceipt::from_path(ReceiptRole::Output, config.output_dir.join(GLTF_FILE))?);
        manifest
            .entries
            .push(FileReceipt::from_path(ReceiptRole::Output, config.output_dir.join(USDA_FILE))?);
    }
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Digital Twin: {} (twin_ready={}, mapping_admitted={})",
        state_path.display(),
        state.twin_ready,
        state.mapping_admitted
    );
    println!("Digital Twin dashboard: {}", html_path.display());
    println!(
        "Digital Twin manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.twin_ready {
        return Err("Digital Twin failed its source/frame admission checks".into());
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
        receipt.input,
        expected_sha256,
        observed_sha256,
        input_receipt.sha256.as_deref() == Some(expected_sha256),
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

    Ok(LoadedMap {
        source,
        frame_id: receipt.tsdf.frame_id,
        time_basis: receipt.sync.time_basis,
        vertex_count,
        triangle_count,
        map_path,
        input_receipt,
        map_receipt,
        receipt_path,
        manifest_path,
    })
}

fn attach_semantic_layer(
    path: &Path,
    expected_sha256: &str,
    expected_frame_id: &str,
) -> Result<ReplayArtifact, String> {
    let semantic: SemanticOverlayState = read_json(path).map_err(|error| {
        format!("semantic overlay could not be parsed and was withheld: {error}")
    })?;
    semantic.validate().map_err(|error| {
        format!("semantic overlay failed state validation and was withheld: {error}")
    })?;
    if !semantic.overlay_ready
        || !semantic.source.identity_matches
        || semantic.source.expected_sha256 != expected_sha256
        || semantic.source.observed_sha256 != expected_sha256
        || semantic.frame_id != expected_frame_id
        || !semantic.summary.source_identity_match
        || !semantic.summary.frame_identity_match
    {
        return Err(
            "semantic overlay source/frame identity did not match the Digital Twin; layer withheld"
                .into(),
        );
    }
    let receipt = FileReceipt::from_path(ReceiptRole::Auxiliary, path)
        .map_err(|error| format!("semantic overlay receipt failed: {error}"))?;
    artifact_from_receipt("semantic-layer", &receipt)
        .map_err(|error| format!("semantic overlay artifact failed: {error}"))
}

fn artifact_from_receipt(
    role: &str,
    receipt: &FileReceipt,
) -> Result<ReplayArtifact, Box<dyn Error>> {
    ReplayArtifact::try_new(
        role,
        receipt.path.display().to_string(),
        receipt.size_bytes.ok_or("receipt has no size")?,
        receipt.sha256.clone().ok_or("receipt has no SHA-256")?,
    )
    .map_err(Into::into)
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

fn copy_identity_file(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    if destination.exists() {
        return Err(format!("output '{}' already exists", destination.display()).into());
    }
    let file_name = destination.file_name().ok_or("output path has no file name")?;
    let temp = destination.with_file_name(format!(".{}.tmp", file_name.to_string_lossy()));
    if temp.exists() {
        return Err(format!("temporary output '{}' already exists", temp.display()).into());
    }
    fs::copy(source, &temp)?;
    fs::rename(&temp, destination)?;
    Ok(())
}

fn render_usda(
    source_sha256: &str,
    frame_id: &str,
    time_basis: &str,
    vertex_count: u64,
    triangle_count: u64,
) -> String {
    format!(
        "#usda 1.0
(
    defaultPrim = \"DigitalTwin\"
    metersPerUnit = 1
    upAxis = \"Z\"
)
def Xform \"DigitalTwin\" (
    kind = \"assembly\"
) {{
    custom string sourceSha256 = \"{}\"
    custom string frameId = \"{}\"
    custom string timeBasis = \"{}\"
    custom int64 vertexCount = {}
    custom int64 triangleCount = {}
    custom asset gltfAsset = @digital-twin.gltf@
    def Scope \"Metadata\" {{
        custom string geometryMode = \"ASCII USDA companion layer; identity-preserving glTF reference\"
    }}
}}
",
        escape_usda(source_sha256),
        escape_usda(frame_id),
        escape_usda(time_basis),
        vertex_count,
        triangle_count
    )
}

fn render_dashboard(state: &DigitalTwinState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#070b16;--panel:#101b2f;--line:#294767;--muted:#91a9c2;--cyan:#69e8ff;--green:#6cf5a7;--red:#ff7382;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#28436e 0,#070b16 48%);color:#edf7ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.18em;text-transform:uppercase}.title{font-size:31px;font-weight:780;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:10px 15px;font-weight:780;white-space:nowrap}.ok{color:var(--green);border-color:#237a50}.blocked{color:var(--red);border-color:#873443}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.panel{background:linear-gradient(145deg,rgba(18,38,65,.96),rgba(8,15,28,.96));border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 16px 36px #0005}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 12px}.metric{font-size:27px;font-weight:780;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:14px;border-bottom:1px solid #1c3552;padding:9px 0}.row:last-child{border-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;overflow-wrap:anywhere}.danger{color:var(--red)}.warning{color:var(--amber)}.blockers{margin:0;padding-left:20px}.blockers li{margin:7px 0;color:#ff9ca6}.assets{display:grid;grid-template-columns:1fr 1fr;gap:11px}.asset{border:1px solid #254363;border-radius:11px;padding:12px;min-width:0}.asset h3{color:var(--cyan);font-size:15px;margin:0 0 8px}.pipeline{display:flex;align-items:center;gap:12px;margin:14px 0}.node{flex:1;border:1px solid #2b5878;border-radius:10px;padding:13px;background:#0b1930;text-align:center}.arrow{color:var(--amber);font-size:22px}.meter{height:9px;border-radius:8px;background:#1b304b;overflow:hidden;margin-top:9px}.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));border-radius:8px}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}.assets{grid-template-columns:1fr}.pipeline{display:block}.arrow{display:block;text-align:center;transform:rotate(90deg)}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / portable digital twin</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Twin gate</h2><div id="twin" class="metric"></div><div id="twinDetail" class="small"></div></article>
<article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Source geometry</h2><div id="geometry" class="metric"></div><div id="geometryDetail" class="small"></div></article>
<article class="panel"><h2>Semantic layer</h2><div id="semantic" class="metric"></div><div id="semanticDetail" class="small"></div></article>
<article class="panel wide"><h2>Portable twin pipeline</h2><div class="pipeline"><div class="node"><strong>Canonical glTF</strong><div class="small">byte-identical payload</div></div><div class="arrow">→</div><div class="node"><strong>USDA companion</strong><div class="small">explicit asset reference</div></div></div><div id="counts"></div></article>
<article class="panel wide"><h2>Assets</h2><div id="assets" class="assets"></div></article>
<article class="panel wide"><h2>Identity and calibration</h2><div id="identity"></div></article>
<article class="panel wide"><h2>Fail-closed notices</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:300px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="digital-twin-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('digital-twin-state').textContent),q=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
q('title').textContent=state.title;q('source').textContent=state.source.path+' · '+state.source.observed_sha256;
q('admission').textContent=state.twin_ready?'TWIN READY':'TWIN BLOCKED';q('admission').className='badge '+(state.twin_ready?'ok':'blocked');
q('twin').textContent=state.twin_ready?'READY':'BLOCKED';q('twin').className='metric '+(state.twin_ready?'':'danger');q('twinDetail').textContent=state.summary.geometry_identity_preserved?'identity-preserving bundle':'export withheld';
q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';
const s=state.summary;q('geometry').textContent=fmt(s.source_vertex_count);q('geometryDetail').textContent=fmt(s.source_triangle_count)+' triangles';q('semantic').textContent=s.semantic_layer_present?'ATTACHED':'NONE';q('semantic').className='metric '+(s.semantic_layer_present?'':'warning');q('semanticDetail').textContent=s.semantic_layer_present?'source/frame checked':'optional layer withheld or not requested';
q('counts').innerHTML='<div class="row"><span>glTF vertices / triangles</span><span class="mono">'+fmt(s.gltf_vertex_count)+' / '+fmt(s.gltf_triangle_count)+'</span></div><div class="row"><span>USDA represented geometry</span><span class="mono">'+fmt(s.usd_vertex_count)+' / '+fmt(s.usd_triangle_count)+'</span></div><div class="meter"><i style="width:'+(s.geometry_identity_preserved?'100':'0')+'%"></i></div>';
q('assets').innerHTML=state.assets.map(a=>'<div class="asset"><h3>'+esc(a.format.toUpperCase())+'</h3><div class="row"><span>mode</span><span class="mono">'+esc(a.geometry_mode)+'</span></div><div class="row"><span>frame</span><span class="mono">'+esc(a.frame_id)+'</span></div><div class="row"><span>vertices / triangles</span><span class="mono">'+fmt(a.vertex_count)+' / '+fmt(a.triangle_count)+'</span></div><div class="small mono">'+esc(a.artifact.path)+'<br>'+esc(a.artifact.sha256)+'</div></div>').join('')||'<div class="small">No assets emitted</div>';
q('identity').innerHTML='<div class="row"><span>source identity</span><span class="mono">'+(s.source_identity_match?'MATCH':'MISMATCH')+'</span></div><div class="row"><span>frame identity</span><span class="mono">'+(s.frame_identity_match?'MATCH':'MISMATCH')+'</span></div><div class="row"><span>observed / expected frame</span><span class="mono">'+esc(state.frame_id)+' / '+esc(state.expected_frame_id)+'</span></div><div class="row"><span>time basis</span><span class="mono">'+esc(state.time_basis)+'</span></div><div class="row"><span>calibration applied</span><span class="mono">'+(s.calibration_applied?'YES':'NO')+'</span></div>';
q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="ok">All gates passed</li>';q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
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
    if config.expected_frame_id.trim().is_empty() || config.min_output_free_bytes == 0 {
        return Err("expected frame and free-space floor must be non-empty/non-zero".into());
    }
    if let Some(path) = &config.semantic_overlay {
        if !path.is_absolute() {
            return Err("--semantic-overlay must be an absolute path".into());
        }
        if !path.is_file() {
            return Err(format!("semantic overlay '{}' is not a file", path.display()).into());
        }
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let run_dir = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut expected_frame_id = None;
    let mut semantic_overlay = None;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--expected-frame-id" => expected_frame_id = Some(next_value(&mut args, &flag)?),
            "--semantic-overlay" => {
                semantic_overlay = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        run_dir: PathBuf::from(run_dir),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        expected_frame_id: expected_frame_id.ok_or("--expected-frame-id is required")?,
        semantic_overlay,
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

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn escape_usda(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn usage() -> String {
    "usage: rosbag2_digital_twin E2E_RUN_DIR \
     --output-dir ABSOLUTE_OUTPUT_DIR --expected-input-sha256 SHA256 \
     --expected-frame-id FRAME [--semantic-overlay ABSOLUTE_JSON] \
     [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{parse_args, render_usda, validate_sha256};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_digital_twin_options() {
        let config = parse_args(
            [
                "/media/e2e",
                "--output-dir",
                "/media/results/digital-twin",
                "--expected-input-sha256",
                SHA,
                "--expected-frame-id",
                "lidar_front",
                "--semantic-overlay",
                "/media/results/semantic-overlay.json",
                "--min-output-free-bytes",
                "1024",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.expected_frame_id, "lidar_front");
        assert_eq!(
            config.semantic_overlay.unwrap(),
            PathBuf::from("/media/results/semantic-overlay.json")
        );
        assert_eq!(config.min_output_free_bytes, 1024);
    }

    #[test]
    fn renders_explicit_companion_reference() {
        let usda = render_usda(SHA, "lidar_front", "header stamp", 10, 4);
        assert!(usda.starts_with("#usda 1.0"));
        assert!(usda.contains("@digital-twin.gltf@"));
        assert!(usda.contains("identity-preserving glTF reference"));
    }

    #[test]
    fn rejects_bad_hash() {
        assert!(validate_sha256("not-a-sha").is_err());
    }
}

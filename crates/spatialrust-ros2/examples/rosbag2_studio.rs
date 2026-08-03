//! Build a portable Spatial Studio state and a static dashboard from rosbag2 receipts.
//!
//! This example is intentionally receipt-driven: it lists the input bag, reads
//! source-bound readiness/TF/E2E evidence, and never applies a transform or
//! silently fuses a second bag. The JSON state can be consumed by native, Web,
//! or notebook frontends; the HTML output is a zero-dependency visual proof.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use spatialrust_io::{FileReceipt, ReceiptRole};
use spatialrust_math::Vec3;
use spatialrust_ros2::{list_topics, Rosbag2Topic};
use spatialrust_viewer::{
    StudioCalibration, StudioFrameGraph, StudioLayer, StudioPerformance, StudioSource,
    StudioStageMetric, StudioState, StudioTimeline, ViewerState, ViewportSize,
};
use spatialrust_viz::{Camera, Projection};

const DEFAULT_TIME_BASIS: &str = "PointCloud2 header stamp; no clock calibration applied";
const E2E_STAGE_NAMES: [&str; 7] = [
    "preflight_wall_ns",
    "ingest_wall_ns",
    "sync_wall_ns",
    "odometry_wall_ns",
    "tsdf_wall_ns",
    "interchange_wall_ns",
    "semantic_viewer_wall_ns",
];

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    manifest: Option<PathBuf>,
    html: Option<PathBuf>,
    expected_sha256: String,
    readiness: PathBuf,
    tf_inventory: Option<PathBuf>,
    e2e_receipt: Option<PathBuf>,
    e2e_manifest: Option<PathBuf>,
}

#[derive(Debug)]
struct CalibrationSnapshot {
    registration_ready: bool,
    clock_status: String,
    frame_status: String,
    source_bound: bool,
    blockers: Vec<String>,
}

#[derive(Debug)]
struct FrameSnapshot {
    graph: StudioFrameGraph,
}

#[derive(Debug)]
struct E2eSnapshot {
    layers: Vec<StudioLayer>,
    performance: StudioPerformance,
}

#[derive(Serialize)]
struct StudioManifest {
    schema: &'static str,
    version: u32,
    mapping_admitted: bool,
    state: FileReceipt,
    dashboard: Option<FileReceipt>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    validate_output_paths(&config)?;
    if !config.input.is_file() {
        return Err(format!("input bag '{}' is not a regular file", config.input.display()).into());
    }

    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let observed_sha256 = input.sha256.clone().ok_or("input checksum was not produced")?;
    let mut blockers = Vec::new();
    if observed_sha256 != config.expected_sha256 {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            ),
        );
    }

    let topics = list_topics(&config.input)?;
    let readiness_value = read_json(&config.readiness)?;
    let calibration =
        calibration_snapshot(&readiness_value, &config.input, &observed_sha256, &mut blockers)?;
    let frame_snapshot = match &config.tf_inventory {
        Some(path) => frame_snapshot(path, &config.input, &observed_sha256, &mut blockers)?,
        None => {
            push_blocker(&mut blockers, "no source-bound TF inventory receipt supplied");
            FrameSnapshot { graph: StudioFrameGraph::try_new(Vec::new(), 0, None, false, false)? }
        }
    };

    let e2e = match &config.e2e_receipt {
        Some(path) => e2e_snapshot(
            path,
            config.e2e_manifest.as_deref(),
            &config.input,
            &observed_sha256,
            &mut blockers,
        )?,
        None => E2eSnapshot { layers: Vec::new(), performance: empty_performance()? },
    };

    let mut layers = if e2e.layers.is_empty() { inventory_layers(&topics)? } else { e2e.layers };
    layers.sort_by(|left, right| left.id.cmp(&right.id));

    let timeline = timeline_from_metadata(
        &config.input,
        topics.iter().map(|topic| topic.message_count).sum(),
        &mut blockers,
    )?;
    let viewer = default_viewer()?;
    let title = format!("Spatial Studio — {}", file_label(&config.input));
    let identity_matches = observed_sha256 == config.expected_sha256;

    let state = StudioState::try_new(
        title,
        viewer,
        StudioSource::try_new(
            "canonical rosbag2 input",
            config.input.display().to_string(),
            config.expected_sha256,
            observed_sha256,
            identity_matches,
        )?,
        layers,
        timeline,
        StudioCalibration::try_new(
            calibration.registration_ready,
            calibration.clock_status,
            calibration.frame_status,
            calibration.source_bound,
            calibration.blockers,
        )?,
        frame_snapshot.graph,
        e2e.performance,
        blockers,
    )?;
    state.validate()?;

    write_json_atomically(&config.output, &serde_json::to_string_pretty(&state)?)?;
    if let Some(path) = &config.html {
        write_json_atomically(path, &render_dashboard(&state)?)?;
    }
    let manifest_path =
        config.manifest.clone().unwrap_or_else(|| default_manifest_path(&config.output));
    let manifest = StudioManifest {
        schema: "spatialrust.rosbag2.studio",
        version: 1,
        mapping_admitted: state.mapping_admitted,
        state: FileReceipt::from_path(ReceiptRole::Output, &config.output)?,
        dashboard: config
            .html
            .as_ref()
            .map(|path| FileReceipt::from_path(ReceiptRole::Output, path))
            .transpose()?,
    };
    write_json_atomically(&manifest_path, &serde_json::to_string_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Spatial Studio state: {} (mapping_admitted={})",
        config.output.display(),
        state.mapping_admitted
    );
    if let Some(path) = &config.html {
        println!("Spatial Studio dashboard: {}", path.display());
    }
    println!("Spatial Studio manifest: {}", manifest_path.display());
    Ok(())
}

fn calibration_snapshot(
    value: &Value,
    input: &Path,
    observed_sha256: &str,
    top_blockers: &mut Vec<String>,
) -> Result<CalibrationSnapshot, Box<dyn Error>> {
    let mut blockers = string_array(value.get("blockers"));
    let input_display = input.display().to_string();
    let input_path = string_at(value, &["input", "path"]);
    let receipt_sha256 = string_at(value, &["input", "sha256"]);
    let source_matches = input_path.as_deref() == Some(input_display.as_str())
        && receipt_sha256.as_deref() == Some(observed_sha256);
    if !source_matches {
        push_blocker(&mut blockers, "calibration readiness receipt is not bound to this input");
        push_blocker(top_blockers, "calibration readiness receipt is not bound to this input");
    }

    let reported_ready = value.get("registration_ready").and_then(Value::as_bool).unwrap_or(false);
    if value.get("registration_ready").is_none() {
        push_blocker(
            &mut blockers,
            "calibration readiness receipt has no registration_ready field",
        );
    }
    let clock_status = string_at(value, &["calibration_artifacts", "clock", "status"])
        .unwrap_or_else(|| "unknown".into());
    let frame_status = string_at(value, &["calibration_artifacts", "frame", "status"])
        .unwrap_or_else(|| "unknown".into());
    let registration_ready = reported_ready && source_matches;
    let source_bound = registration_ready
        && clock_status == "registered"
        && frame_status == "registered"
        && blockers.is_empty();
    if !registration_ready && blockers.is_empty() {
        push_blocker(&mut blockers, "calibration registration is blocked without a receipt reason");
    }
    for blocker in &blockers {
        push_blocker(top_blockers, format!("calibration: {blocker}"));
    }
    Ok(CalibrationSnapshot {
        registration_ready,
        clock_status,
        frame_status,
        source_bound,
        blockers,
    })
}

fn frame_snapshot(
    path: &Path,
    input: &Path,
    observed_sha256: &str,
    top_blockers: &mut Vec<String>,
) -> Result<FrameSnapshot, Box<dyn Error>> {
    let value = read_json(path)?;
    let mut blockers = string_array(value.get("blockers"));
    let input_display = input.display().to_string();
    let receipt_input_path = string_at(&value, &["input", "path"]);
    let source_identity_matches = value
        .get("source_identity")
        .and_then(|identity| identity.get("matches"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let receipt_observed_sha = string_at(&value, &["source_identity", "observed_input_sha256"]);
    let source_bound = source_identity_matches
        && receipt_observed_sha.as_deref() == Some(observed_sha256)
        && receipt_input_path.as_deref() == Some(input_display.as_str());
    if !source_bound {
        push_blocker(&mut blockers, "TF inventory source identity does not match this input");
    }
    for blocker in &blockers {
        push_blocker(top_blockers, format!("TF inventory: {blocker}"));
    }

    let observed_frames = if source_bound {
        value
            .get("observed_frames")
            .and_then(Value::as_array)
            .map(|frames| {
                frames.iter().filter_map(Value::as_str).map(str::to_owned).collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let edge_count = if source_bound {
        value
            .get("messages")
            .and_then(Value::as_array)
            .map(|messages| {
                messages
                    .iter()
                    .filter_map(|message| message.get("transforms").and_then(Value::as_array))
                    .map(|transforms| u64::try_from(transforms.len()).unwrap_or(u64::MAX))
                    .sum()
            })
            .unwrap_or(0)
    } else {
        0
    };
    Ok(FrameSnapshot {
        graph: StudioFrameGraph::try_new(observed_frames, edge_count, None, false, source_bound)?,
    })
}

fn e2e_snapshot(
    receipt_path: &Path,
    manifest_path: Option<&Path>,
    input: &Path,
    observed_sha256: &str,
    top_blockers: &mut Vec<String>,
) -> Result<E2eSnapshot, Box<dyn Error>> {
    let value = read_json(receipt_path)?;
    let mut blockers = Vec::new();
    let input_display = input.display().to_string();
    if string_at(&value, &["input"]).as_deref() != Some(input_display.as_str()) {
        push_blocker(&mut blockers, "E2E receipt input path does not match this input");
    }
    let inferred_manifest = receipt_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| receipt_path.with_file_name(name.replace(".receipt.json", ".manifest.json")));
    let manifest = manifest_path.map(Path::to_path_buf).or(inferred_manifest);
    let manifest_matches = manifest
        .as_deref()
        .filter(|path| path.is_file())
        .map(|path| manifest_input_matches(path, input, observed_sha256))
        .transpose()?
        .unwrap_or(false);
    if !manifest_matches {
        push_blocker(&mut blockers, "E2E receipt has no manifest entry proving this input SHA-256");
    }
    if !blockers.is_empty() {
        for blocker in &blockers {
            push_blocker(top_blockers, format!("E2E receipt: {blocker}"));
        }
        return Ok(E2eSnapshot { layers: Vec::new(), performance: empty_performance()? });
    }

    let mut layers = Vec::new();
    if let Some(topics) = value.get("ingest").and_then(|ingest| ingest.get("topics")) {
        if let Some(topics) = topics.as_array() {
            for topic in topics {
                let name = string_at(topic, &["topic"]).unwrap_or_else(|| "unknown".into());
                let id = format!("ros/{}", name.trim_matches('/').replace('/', "_"));
                let frame_id = topic
                    .get("frame_ids")
                    .and_then(Value::as_array)
                    .and_then(|frames| frames.first())
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let point_count = u64_at(topic, &["retained_points"]).unwrap_or(0);
                layers.push(StudioLayer::try_new(
                    id,
                    name.clone(),
                    "point_cloud",
                    name,
                    frame_id,
                    u64_at(topic, &["bag_message_count"]).unwrap_or(0),
                    point_count,
                    point_count > 0,
                    true,
                )?);
            }
        }
    }
    if let Some(tsdf) = value.get("tsdf") {
        let point_count = u64_at(&value, &["tsdf", "mesh_vertices"]).unwrap_or(0);
        if point_count > 0 {
            layers.push(StudioLayer::try_new(
                "derived/tsdf_mesh",
                "TSDF mesh",
                "mesh",
                string_at(&value, &["interchange", "path"]).unwrap_or_else(|| "glTF".into()),
                string_at(tsdf, &["frame_id"]),
                1,
                point_count,
                true,
                true,
            )?);
        }
    }
    if let Some(semantic) = value.get("semantic") {
        let entity_count = u64_at(semantic, &["entities"]).unwrap_or(0);
        if entity_count > 0 {
            let frame_id = semantic
                .get("frame_ids")
                .and_then(Value::as_array)
                .and_then(|frames| frames.first())
                .and_then(Value::as_str)
                .map(str::to_owned);
            layers.push(StudioLayer::try_new(
                "derived/semantic",
                "Semantic entities",
                "semantic",
                string_at(semantic, &["runtime"]).unwrap_or_else(|| "semantic receipt".into()),
                frame_id,
                entity_count,
                entity_count,
                true,
                true,
            )?);
        }
    }
    Ok(E2eSnapshot { layers, performance: performance_from_e2e(&value)? })
}

fn performance_from_e2e(value: &Value) -> Result<StudioPerformance, Box<dyn Error>> {
    let performance = value.get("performance");
    let stages = E2E_STAGE_NAMES
        .iter()
        .filter_map(|name| {
            performance
                .and_then(|performance| performance.get("stages"))
                .and_then(|stages| stages.get(*name))
                .and_then(Value::as_u64)
                .map(|wall_ns| StudioStageMetric { name: (*name).into(), wall_ns })
        })
        .collect();
    Ok(StudioPerformance::try_new(
        performance
            .and_then(|performance| u64_at(performance, &["observed_pipeline_wall_ns"]))
            .unwrap_or(0),
        stages,
        performance
            .and_then(|performance| u64_at(performance, &["memory", "peak_source_bytes"]))
            .unwrap_or(0),
        performance
            .and_then(|performance| u64_at(performance, &["transfers", "host_to_device_bytes"]))
            .unwrap_or(0),
        performance
            .and_then(|performance| u64_at(performance, &["transfers", "device_to_host_bytes"]))
            .unwrap_or(0),
        performance
            .and_then(|performance| u64_at(performance, &["transfers", "hidden_device_copies"]))
            .unwrap_or(0),
    )?)
}

fn inventory_layers(topics: &[Rosbag2Topic]) -> Result<Vec<StudioLayer>, Box<dyn Error>> {
    topics
        .iter()
        .filter(|topic| topic.type_name == "sensor_msgs/msg/PointCloud2")
        .enumerate()
        .map(|(index, topic)| {
            StudioLayer::try_new(
                format!("inventory/{index}"),
                topic.name.clone(),
                "point_cloud",
                topic.name.clone(),
                None,
                topic.message_count,
                0,
                false,
                true,
            )
            .map_err(|error| error.into())
        })
        .collect()
}

fn timeline_from_metadata(
    input: &Path,
    sample_count: u64,
    blockers: &mut Vec<String>,
) -> Result<StudioTimeline, Box<dyn Error>> {
    let metadata_path = input.with_file_name("metadata.yaml");
    let bounds = if metadata_path.is_file() {
        parse_metadata_bounds(&fs::read_to_string(&metadata_path)?)
    } else {
        None
    };
    let Some((start_nanos, duration_nanos)) = bounds else {
        push_blocker(
            blockers,
            format!("rosbag2 metadata '{}' has no usable timeline bounds", metadata_path.display()),
        );
        return Ok(StudioTimeline::try_new(None, None, None, 0, "unavailable", false)?);
    };
    let end_nanos =
        start_nanos.checked_add(duration_nanos).ok_or("timeline end timestamp overflow")?;
    Ok(StudioTimeline::try_new(
        Some(start_nanos),
        Some(end_nanos),
        Some(start_nanos),
        sample_count,
        DEFAULT_TIME_BASIS,
        false,
    )?)
}

fn parse_metadata_bounds(text: &str) -> Option<(u64, u64)> {
    let mut section = "";
    let mut duration = None;
    let mut start = None;
    for line in text.lines() {
        let trimmed = line.trim();
        match trimmed {
            "duration:" => section = "duration",
            "starting_time:" => section = "starting_time",
            _ => {
                if section == "duration" {
                    duration = duration.or_else(|| scalar_u64(trimmed, "nanoseconds:"));
                } else if section == "starting_time" {
                    start = start.or_else(|| scalar_u64(trimmed, "nanoseconds_since_epoch:"));
                }
            }
        }
    }
    Some((start?, duration?))
}

fn manifest_input_matches(
    path: &Path,
    input: &Path,
    observed_sha256: &str,
) -> Result<bool, Box<dyn Error>> {
    let value = read_json(path)?;
    let input_display = input.display().to_string();
    Ok(value.get("entries").and_then(Value::as_array).into_iter().flatten().any(|entry| {
        entry.get("role").and_then(Value::as_str) == Some("input")
            && string_at(entry, &["path"]).as_deref() == Some(input_display.as_str())
            && string_at(entry, &["sha256"]).as_deref() == Some(observed_sha256)
    }))
}

fn default_viewer() -> Result<ViewerState, Box<dyn Error>> {
    let camera = Camera::try_new(
        Vec3::new(0.0, -8.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 10_000.0 },
    )?;
    Ok(ViewerState::try_new(camera, ViewportSize::try_new(1600, 900)?)?)
}

fn empty_performance() -> Result<StudioPerformance, Box<dyn Error>> {
    Ok(StudioPerformance::try_new(0, Vec::new(), 0, 0, 0, 0)?)
}

fn render_dashboard(state: &StudioState) -> Result<String, Box<dyn Error>> {
    let json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--panel:#0d1c2f;--line:#1d3856;--muted:#86a1bb;--cyan:#54d6ff;--green:#58e39b;--red:#ff6f7d;--amber:#ffc857}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#14345a 0,#07111f 42%);color:#e9f3ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1500px;margin:0 auto;padding:28px}.top{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);letter-spacing:.15em;text-transform:uppercase;font-size:11px}.title{font-size:31px;font-weight:700;margin-top:5px}.sub{color:var(--muted);font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:700;white-space:nowrap}.ok{color:var(--green);border-color:#237750}.blocked{color:var(--red);border-color:#853442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(16,37,62,.96),rgba(8,20,35,.96));border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 30px #0003}.panel h2{font-size:12px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin:0 0 12px}.metric{font-size:24px;font-weight:700;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.chips{display:flex;flex-wrap:wrap;gap:7px}.chip{background:#102b47;border:1px solid #28537b;border-radius:999px;padding:5px 9px;font-size:12px}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #17304a;padding:9px 0}.row:last-child{border-bottom:0}.danger{color:var(--red)}.warning{color:var(--amber)}.bar{height:8px;background:#07111f;border-radius:5px;overflow:hidden;margin-top:10px}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">Spatial Studio / source-bound session</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="small mono"></div></article>
<article class="panel"><h2>Calibration gate</h2><div id="calibration" class="metric"></div><div id="calibrationDetail" class="small"></div></article>
<article class="panel"><h2>Frame graph</h2><div id="frames" class="metric"></div><div id="frameDetail" class="small"></div></article>
<article class="panel"><h2>Pipeline wall time</h2><div id="wall" class="metric"></div><div id="transfer" class="small"></div></article>
<article class="panel wide"><h2>Point-cloud / derived layers</h2><div id="layers" class="chips"></div></article>
<article class="panel wide"><h2>Timeline</h2><div id="timeline" class="mono"></div><div class="bar"><div id="timelineBar" class="fill"></div></div></article>
<article class="panel wide"><h2>Stage metrics</h2><div id="stages"></div></article>
<article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:240px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="studio-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('studio-state').textContent);
const q=id=>document.getElementById(id); const fmtNs=n=>n==null?'—':(n/1e9).toFixed(3)+' s';
const fmtBytes=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB';
const esc=value=>String(value).replace(/[&<>"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[char]));
q('title').textContent=state.title; q('source').textContent=state.source.path;
q('admission').textContent=state.mapping_admitted?'MAPPING ADMITTED':'MAPPING BLOCKED'; q('admission').className='badge '+(state.mapping_admitted?'ok':'blocked');
q('identity').textContent=state.source.identity_matches?'MATCH':'MISMATCH'; q('identity').className='metric '+(state.source.identity_matches?'':'danger');
q('sha').textContent=state.source.observed_sha256; q('calibration').textContent=state.calibration.registration_ready?'READY':'BLOCKED'; q('calibration').className='metric '+(state.calibration.registration_ready?'':'danger');
q('calibrationDetail').textContent='clock: '+state.calibration.clock_status+' · frame: '+state.calibration.frame_status;
q('frames').textContent=state.frame_graph.observed_frames.length+' / '+state.frame_graph.edge_count; q('frameDetail').textContent=state.frame_graph.composed?'composed':'inventory only · no transform applied';
q('wall').textContent=fmtNs(state.performance.observed_pipeline_wall_ns); q('transfer').textContent='H→D '+fmtBytes(state.performance.host_to_device_bytes)+' · D→H '+fmtBytes(state.performance.device_to_host_bytes)+' · hidden '+state.performance.hidden_device_copies;
q('layers').innerHTML=state.layers.map(l=>'<span class="chip">'+(l.renderable?'●':'○')+' '+esc(l.label)+' · '+l.point_count.toLocaleString()+' pts</span>').join('')||'<span class="small">No renderable receipt layer</span>';
const t=state.timeline; q('timeline').textContent=t.start_nanos==null?'unavailable':fmtNs(t.start_nanos)+' → '+fmtNs(t.end_nanos)+' · '+t.sample_count.toLocaleString()+' samples · '+t.time_basis; q('timelineBar').style.width=t.start_nanos==null?'0%':(((t.cursor_nanos-t.start_nanos)/Math.max(1,t.end_nanos-t.start_nanos))*100)+'%';
q('stages').innerHTML=state.performance.stages.map(s=>'<div class="row"><span>'+s.name+'</span><span class="mono">'+fmtNs(s.wall_ns)+'</span></div>').join('')||'<div class="small">No source-bound performance receipt</div>';
q('blockers').innerHTML=state.blockers.map(b=>'<li>'+esc(b)+'</li>').join('')||'<li class="ok">All gates passed</li>'; q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &json))
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str().map(str::to_owned)
}

fn u64_at(value: &Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default()
}

fn scalar_u64(value: &str, prefix: &str) -> Option<u64> {
    value.strip_prefix(prefix)?.trim().parse().ok()
}

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
}

fn validate_output_paths(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.input.is_absolute() || !config.output.is_absolute() {
        return Err("input and --output paths must be absolute".into());
    }
    if config.html.as_ref().is_some_and(|path| !path.is_absolute()) {
        return Err("--html path must be absolute".into());
    }
    if config.manifest.as_ref().is_some_and(|path| !path.is_absolute()) {
        return Err("--manifest path must be absolute".into());
    }
    for (name, path) in [
        ("readiness", Some(&config.readiness)),
        ("TF inventory", config.tf_inventory.as_ref()),
        ("E2E receipt", config.e2e_receipt.as_ref()),
        ("E2E manifest", config.e2e_manifest.as_ref()),
    ] {
        if path.is_some_and(|path| !path.is_absolute()) {
            return Err(format!("{name} path must be absolute").into());
        }
    }
    if config.expected_sha256.len() != 64
        || !config.expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || config.expected_sha256.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err("--expected-input-sha256 must be 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut output = None;
    let mut manifest = None;
    let mut html = None;
    let mut expected_sha256 = None;
    let mut readiness = None;
    let mut tf_inventory = None;
    let mut e2e_receipt = None;
    let mut e2e_manifest = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output" => output = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--html" => html = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--readiness" => readiness = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--tf-inventory" => tf_inventory = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--e2e-receipt" => e2e_receipt = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--e2e-manifest" => e2e_manifest = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    let config = Config {
        input,
        output: output.ok_or_else(|| "--output is required".to_owned())?,
        manifest,
        html,
        expected_sha256: expected_sha256
            .ok_or_else(|| "--expected-input-sha256 is required".to_owned())?,
        readiness: readiness.ok_or_else(|| "--readiness is required".to_owned())?,
        tf_inventory,
        e2e_receipt,
        e2e_manifest,
    };
    validate_output_paths(&config)?;
    if config.e2e_manifest.is_some() && config.e2e_receipt.is_none() {
        return Err("--e2e-manifest requires --e2e-receipt".into());
    }
    let manifest = config.manifest.clone().unwrap_or_else(|| default_manifest_path(&config.output));
    for (name, path) in [("state", config.output.clone()), ("manifest", manifest.clone())] {
        if path.exists() {
            return Err(format!("Studio {name} output '{}' already exists", path.display()).into());
        }
    }
    if let Some(html) = &config.html {
        if html.exists() {
            return Err(format!("Studio HTML output '{}' already exists", html.display()).into());
        }
        if html == &config.output || html == &manifest {
            return Err("HTML path must differ from state and manifest outputs".into());
        }
    }
    Ok(config)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn write_json_atomically(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("Studio output '{}' already exists", path.display()).into());
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("studio.json")
    ));
    fs::write(&temporary, format!("{content}\n"))?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn default_manifest_path(output: &Path) -> PathBuf {
    let Some(name) = output.file_name().and_then(|name| name.to_str()) else {
        return output.join("spatial-studio.manifest.json");
    };
    let manifest_name = name
        .strip_suffix(".json")
        .map_or_else(|| format!("{name}.manifest.json"), |stem| format!("{stem}.manifest.json"));
    output.with_file_name(manifest_name)
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn file_label(path: &Path) -> String {
    path.file_name().and_then(|name| name.to_str()).unwrap_or("input").to_owned()
}

fn usage() -> String {
    "usage: rosbag2_studio INPUT_DB3 --output ABSOLUTE_STATE_JSON [--manifest ABSOLUTE_MANIFEST_JSON] \
     --expected-input-sha256 SHA256 --readiness ABSOLUTE_READINESS_JSON \
     [--html ABSOLUTE_DASHBOARD_HTML] [--tf-inventory ABSOLUTE_TF_JSON] \
     [--e2e-receipt ABSOLUTE_E2E_JSON] [--e2e-manifest ABSOLUTE_MANIFEST_JSON]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, parse_metadata_bounds};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_required_receipts_and_absolute_paths() {
        let config = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/studio.json",
                "--expected-input-sha256",
                SHA,
                "--readiness",
                "/media/results/readiness.json",
                "--html",
                "/media/results/studio.html",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.output.to_string_lossy(), "/media/results/studio.json");
        assert!(config.tf_inventory.is_none());
    }

    #[test]
    fn rejects_short_hash_and_relative_receipt_paths() {
        let short_hash = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/studio.json",
                "--expected-input-sha256",
                "bad",
                "--readiness",
                "/media/results/readiness.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(short_hash.to_string().contains("64 lowercase"));

        let relative = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/studio.json",
                "--expected-input-sha256",
                SHA,
                "--readiness",
                "readiness.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(relative.to_string().contains("absolute"));
    }

    #[test]
    fn parses_rosbag_metadata_duration_and_start() {
        let text = "duration:\n  nanoseconds: 12\nstarting_time:\n  nanoseconds_since_epoch: 100\n";
        assert_eq!(parse_metadata_bounds(text), Some((100, 12)));
    }
}

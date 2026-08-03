//! Build a source-bound TF/calibration observatory from rosbag2 receipts.
//!
//! This example records what is known, what is missing, and what is refused.
//! It never solves clock calibration, composes TF edges, or applies a transform
//! from a receipt whose input identity does not match the displayed bag.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use spatialrust_io::{FileReceipt, ReceiptRole};
use spatialrust_viewer::{
    CalibrationArtifact, CalibrationObservatoryState, ClockCalibration, FrameTransform,
    StudioSource,
};

const TIME_BASIS: &str = "PointCloud2 header stamp; no clock calibration applied";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    html: Option<PathBuf>,
    expected_sha256: String,
    readiness: PathBuf,
    tf_inventory: Option<PathBuf>,
}

#[derive(Serialize)]
struct ObservatoryManifest {
    schema: &'static str,
    version: u32,
    calibration_admitted: bool,
    state: FileReceipt,
    dashboard: Option<FileReceipt>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    let manifest_path = config.output.with_file_name(default_manifest_name(&config.output));
    ensure_outputs_absent(&config, &manifest_path)?;
    if !config.input.is_file() {
        return Err(format!("input bag '{}' is not a regular file", config.input.display()).into());
    }

    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let observed_sha256 = input.sha256.clone().ok_or("input checksum was not produced")?;
    let mut blockers = Vec::new();
    let identity_matches = observed_sha256 == config.expected_sha256;
    if !identity_matches {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            ),
        );
    }
    let input_display = config.input.display().to_string();
    let source = StudioSource::try_new(
        "calibration observatory input",
        input_display.clone(),
        config.expected_sha256,
        observed_sha256.clone(),
        identity_matches,
    )?;

    let readiness = read_json(&config.readiness)?;
    let readiness_bound = readiness_source_matches(&readiness, &input_display, &observed_sha256);
    if !readiness_bound {
        push_blocker(&mut blockers, "readiness receipt is not bound to this input");
    }
    for blocker in string_array(readiness.get("blockers")) {
        push_blocker(&mut blockers, format!("readiness: {blocker}"));
    }
    let clock_artifact = artifact_from_readiness(&readiness, "clock", readiness_bound)?;
    let frame_artifact = artifact_from_readiness(&readiness, "frame", readiness_bound)?;
    if clock_artifact.status != "registered" {
        push_blocker(
            &mut blockers,
            format!("clock calibration artifact is {}", clock_artifact.status),
        );
    }
    if frame_artifact.status != "registered" {
        push_blocker(
            &mut blockers,
            format!("frame calibration artifact is {}", frame_artifact.status),
        );
    }
    if clock_artifact.status == "registered" && !clock_artifact.source_bound {
        push_blocker(&mut blockers, "clock calibration artifact is source-unbound");
    }
    if frame_artifact.status == "registered" && !frame_artifact.source_bound {
        push_blocker(&mut blockers, "frame calibration artifact is source-unbound");
    }

    let clock = ClockCalibration::try_new(
        clock_artifact.status.clone(),
        TIME_BASIS,
        0,
        None,
        None,
        None,
        None,
        clock_artifact.source_bound,
        false,
    )?;
    push_blocker(&mut blockers, "clock model was not applied; artifact contents remain opaque");

    let (frames, edges, frame_source_bound) = match &config.tf_inventory {
        Some(path) => read_tf_inventory(path, &input_display, &observed_sha256, &mut blockers)?,
        None => {
            push_blocker(&mut blockers, "no source-bound TF inventory receipt supplied");
            (Vec::new(), Vec::new(), false)
        }
    };
    push_blocker(&mut blockers, "frame graph is inventory-only; no transform composition applied");

    let state = CalibrationObservatoryState::try_new(
        format!("TF / Calibration Observatory — {}", file_label(&config.input)),
        source,
        clock_artifact,
        frame_artifact,
        clock,
        frames,
        edges,
        0,
        frame_source_bound,
        None,
        false,
        blockers,
    )?;
    state.validate()?;

    write_atomic(&config.output, &serde_json::to_string_pretty(&state)?)?;
    if let Some(path) = &config.html {
        write_atomic(path, &render_dashboard(&state)?)?;
    }
    let manifest = ObservatoryManifest {
        schema: "spatialrust.rosbag2.calibration-observatory",
        version: 1,
        calibration_admitted: state.calibration_admitted,
        state: FileReceipt::from_path(ReceiptRole::Output, &config.output)?,
        dashboard: config
            .html
            .as_ref()
            .map(|path| FileReceipt::from_path(ReceiptRole::Output, path))
            .transpose()?,
    };
    write_atomic(&manifest_path, &serde_json::to_string_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Calibration Observatory state: {} (calibration_admitted={})",
        config.output.display(),
        state.calibration_admitted
    );
    if let Some(path) = &config.html {
        println!("Calibration Observatory dashboard: {}", path.display());
    }
    println!("Calibration Observatory manifest: {}", manifest_path.display());
    Ok(())
}

fn artifact_from_readiness(
    readiness: &Value,
    kind: &str,
    source_bound: bool,
) -> Result<CalibrationArtifact, Box<dyn Error>> {
    let status = string_at(readiness, &["calibration_artifacts", kind, "status"])
        .unwrap_or_else(|| "unknown".into());
    let path = string_at(readiness, &["calibration_artifacts", kind, "file", "path"]);
    let sha256 = string_at(readiness, &["calibration_artifacts", kind, "file", "sha256"]);
    Ok(CalibrationArtifact::try_new(
        format!("{kind}_calibration"),
        status.clone(),
        path,
        sha256,
        source_bound && status == "registered",
    )?)
}

fn read_tf_inventory(
    path: &Path,
    input_display: &str,
    observed_sha256: &str,
    blockers: &mut Vec<String>,
) -> Result<(Vec<String>, Vec<FrameTransform>, bool), Box<dyn Error>> {
    let value = read_json(path)?;
    let receipt_input = string_at(&value, &["input", "path"]);
    let source_identity_matches = value
        .get("source_identity")
        .and_then(|identity| identity.get("matches"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let receipt_observed_sha = string_at(&value, &["source_identity", "observed_input_sha256"]);
    let source_bound = source_identity_matches
        && receipt_input.as_deref() == Some(input_display)
        && receipt_observed_sha.as_deref() == Some(observed_sha256);
    for blocker in string_array(value.get("blockers")) {
        push_blocker(blockers, format!("TF inventory: {blocker}"));
    }
    if !source_bound {
        push_blocker(blockers, "TF inventory source identity does not match this input");
        return Ok((Vec::new(), Vec::new(), false));
    }

    let frames = value
        .get("observed_frames")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default();
    let mut edges = Vec::new();
    if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        for message in messages {
            let stamp_nanos = transform_stamp_nanos(message);
            if let Some(transforms) = message.get("transforms").and_then(Value::as_array) {
                for transform in transforms {
                    let parent = string_at(transform, &["frame_id"])
                        .ok_or("TF transform has no frame_id")?;
                    let child = string_at(transform, &["child_frame_id"])
                        .ok_or("TF transform has no child_frame_id")?;
                    let translation = array_f64::<3>(transform, "translation_m")?;
                    let rotation = array_f64::<4>(transform, "rotation_xyzw")?;
                    edges.push(FrameTransform::try_new(
                        parent,
                        child,
                        translation,
                        rotation,
                        stamp_nanos,
                        true,
                        true,
                    )?);
                }
            }
        }
    }
    if edges.is_empty() {
        push_blocker(blockers, "source-bound TF inventory contains no transform edges");
    }
    Ok((frames, edges, true))
}

fn transform_stamp_nanos(message: &Value) -> Option<u64> {
    let sec = message
        .get("transforms")
        .and_then(Value::as_array)
        .and_then(|transforms| transforms.first())
        .and_then(|transform| transform.get("stamp_sec"))
        .and_then(Value::as_i64)?;
    let nanosec = message
        .get("transforms")
        .and_then(Value::as_array)
        .and_then(|transforms| transforms.first())
        .and_then(|transform| transform.get("stamp_nanosec"))
        .and_then(Value::as_u64)?;
    if sec < 0 {
        return None;
    }
    u64::try_from(sec).ok()?.checked_mul(1_000_000_000)?.checked_add(nanosec)
}

fn array_f64<const N: usize>(value: &Value, key: &str) -> Result<[f64; N], Box<dyn Error>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("TF transform has no {key} array"))?;
    if values.len() != N {
        return Err(format!("TF transform {key} length {} != {N}", values.len()).into());
    }
    let mut result = [0.0; N];
    for (index, value) in values.iter().enumerate() {
        result[index] =
            value.as_f64().ok_or_else(|| format!("TF transform {key} value is not f64"))?;
    }
    Ok(result)
}

fn readiness_source_matches(value: &Value, input_display: &str, observed_sha256: &str) -> bool {
    string_at(value, &["input", "path"]).as_deref() == Some(input_display)
        && string_at(value, &["input", "sha256"]).as_deref() == Some(observed_sha256)
}

fn render_dashboard(state: &CalibrationObservatoryState) -> Result<String, Box<dyn Error>> {
    let json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#080d1b;--panel:#111b34;--line:#2a3a62;--muted:#8c9abd;--cyan:#68e1ff;--green:#62e6a5;--red:#ff7184;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#28265d 0,#080d1b 46%);color:#edf4ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}.eyebrow{color:var(--cyan);letter-spacing:.16em;text-transform:uppercase;font-size:11px}.title{font-size:29px;font-weight:750;margin-top:4px}.sub,.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:10px 15px;font-weight:750;white-space:nowrap}.good{color:var(--green);border-color:#287b59}.bad{color:var(--red);border-color:#8b3c50}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.panel{background:linear-gradient(145deg,#142341eF,#0d1529eF);border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 16px 35px #0004}.panel h2{font-size:11px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;margin:0 0 12px}.metric{font-size:23px;font-weight:750;color:var(--cyan)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:15px;padding:9px 0;border-bottom:1px solid #223253}.row:last-child{border-bottom:0}.chips{display:flex;flex-wrap:wrap;gap:7px}.chip{padding:5px 9px;border:1px solid #3c5688;border-radius:99px;background:#182b4d;font-size:12px}.edge{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:center;padding:10px;border:1px solid #30466e;border-radius:10px;background:#0b1730;margin:7px 0}.arrow{color:var(--cyan);font-size:20px;text-align:center}.empty{padding:24px;text-align:center;border:1px dashed #8b3c50;border-radius:12px;color:var(--red);letter-spacing:.12em}.blockers{margin:0;padding-left:20px}.blockers li{color:#ff9eaa;margin:7px 0}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:560px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:14px}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / TF observatory</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="mono"></div></article>
<article class="panel"><h2>Clock calibration</h2><div id="clock" class="metric"></div><div id="clockDetail" class="sub"></div></article>
<article class="panel"><h2>Frame artifact</h2><div id="frameArtifact" class="metric"></div><div id="frameDetail" class="sub"></div></article>
<article class="panel"><h2>Graph admission</h2><div id="graph" class="metric"></div><div id="graphDetail" class="sub"></div></article>
<article class="panel wide"><h2>Artifact receipts</h2><div id="artifacts"></div></article>
<article class="panel wide"><h2>Clock diagnostics</h2><div id="diagnostics"></div></article>
<article class="panel full"><h2>Accepted source-bound edges</h2><div id="edges"></div></article>
<article class="panel full"><h2>Fail-closed blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel full"><h2>Portable observatory JSON</h2><pre id="raw" class="mono" style="max-height:230px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="observatory-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('observatory-state').textContent);
const q=id=>document.getElementById(id); const esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const status=(value,ok)=>{q(value).className='metric '+(ok?'good':'bad');q(value).textContent=ok?'READY':'BLOCKED';};
q('title').textContent=state.title;q('source').textContent=state.source.path;
q('admission').textContent=state.calibration_admitted?'CALIBRATION ADMITTED':'CALIBRATION BLOCKED';q('admission').className='badge '+(state.calibration_admitted?'good':'bad');
status('identity',state.source.identity_matches);q('sha').textContent=state.source.observed_sha256;
status('clock',state.clock.applied);q('clockDetail').textContent='status: '+state.clock.status+' · samples: '+state.clock.sample_count+' · applied: '+state.clock.applied;
status('frameArtifact',state.frame_artifact.status==='registered'&&state.frame_artifact.source_bound);q('frameDetail').textContent='status: '+state.frame_artifact.status+' · source-bound: '+state.frame_artifact.source_bound;
status('graph',state.composed&&state.cycle_free);q('graphDetail').textContent=state.frames.length+' frames · '+state.edges.length+' accepted edges · '+state.rejected_edge_count+' rejected';
q('artifacts').innerHTML=[state.clock_artifact,state.frame_artifact].map(a=>'<div class="row"><span>'+esc(a.kind)+'</span><span>'+esc(a.status)+(a.source_bound?' · bound':'')+'</span></div>').join('');
q('diagnostics').innerHTML='<div class="row"><span>time basis</span><span>'+esc(state.clock.time_basis)+'</span></div><div class="row"><span>median offset</span><span>'+(state.clock.median_offset_nanos??'—')+' ns</span></div><div class="row"><span>p95 offset</span><span>'+(state.clock.p95_abs_offset_nanos??'—')+' ns</span></div><div class="row"><span>uncertainty</span><span>'+(state.clock.uncertainty_nanos??'—')+' ns</span></div>';
q('edges').innerHTML=state.edges.length?state.edges.map(e=>'<div class="edge"><span>'+esc(e.parent_frame)+'</span><span class="arrow">→</span><span>'+esc(e.child_frame)+'</span></div>').join(''):'<div class="empty">NO ACCEPTED EDGES</div>';
q('blockers').innerHTML=state.blockers.map(b=>'<li>'+esc(b)+'</li>').join('')||'<li class="good">All observatory gates passed</li>';q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &json))
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
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
        .map(|items| items.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default()
}

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
}

fn default_manifest_name(output: &Path) -> String {
    output
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".json"))
        .map_or_else(
            || "calibration-observatory.manifest.json".into(),
            |stem| format!("{stem}.manifest.json"),
        )
}

fn ensure_outputs_absent(config: &Config, manifest: &Path) -> Result<(), Box<dyn Error>> {
    let mut paths = vec![("state", config.output.clone()), ("manifest", manifest.to_path_buf())];
    if let Some(html) = &config.html {
        paths.push(("HTML", html.clone()));
    }
    for (label, path) in paths {
        if path.exists() {
            return Err(
                format!("observatory {label} output '{}' already exists", path.display()).into()
            );
        }
    }
    if config.html.as_ref().is_some_and(|html| html == &config.output || html == manifest) {
        return Err("observatory HTML path must differ from state and manifest".into());
    }
    Ok(())
}

fn write_atomic(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("observatory.json")
    ));
    fs::write(&temporary, format!("{content}\n"))?;
    fs::rename(&temporary, path)?;
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

fn file_label(path: &Path) -> String {
    path.file_name().and_then(|name| name.to_str()).unwrap_or("input").to_owned()
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut output = None;
    let mut html = None;
    let mut expected_sha256 = None;
    let mut readiness = None;
    let mut tf_inventory = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output" => output = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--html" => html = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--readiness" => readiness = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--tf-inventory" => tf_inventory = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    let config = Config {
        input,
        output: output.ok_or_else(|| "--output is required".to_owned())?,
        html,
        expected_sha256: expected_sha256
            .ok_or_else(|| "--expected-input-sha256 is required".to_owned())?,
        readiness: readiness.ok_or_else(|| "--readiness is required".to_owned())?,
        tf_inventory,
    };
    if !config.input.is_absolute()
        || !config.output.is_absolute()
        || !config.readiness.is_absolute()
        || config.html.as_ref().is_some_and(|path| !path.is_absolute())
        || config.tf_inventory.as_ref().is_some_and(|path| !path.is_absolute())
    {
        return Err("input, output, and receipt paths must be absolute".into());
    }
    if config.expected_sha256.len() != 64
        || !config.expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || config.expected_sha256.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err("--expected-input-sha256 must be 64 lowercase hexadecimal characters".into());
    }
    Ok(config)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn usage() -> String {
    "usage: rosbag2_calibration_observatory INPUT_DB3 --output ABSOLUTE_STATE_JSON \
     --expected-input-sha256 SHA256 --readiness ABSOLUTE_READINESS_JSON \
     [--html ABSOLUTE_DASHBOARD_HTML] [--tf-inventory ABSOLUTE_TF_JSON]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_absolute_observatory_paths() {
        let config = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/observatory.json",
                "--expected-input-sha256",
                SHA,
                "--readiness",
                "/media/results/readiness.json",
                "--tf-inventory",
                "/media/results/tf.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.tf_inventory.unwrap().to_string_lossy(), "/media/results/tf.json");
    }

    #[test]
    fn rejects_relative_paths_and_bad_hashes() {
        let relative = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/observatory.json",
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

        let bad_hash = parse_args(
            [
                "/media/input/bag.db3",
                "--output",
                "/media/results/observatory.json",
                "--expected-input-sha256",
                "bad",
                "--readiness",
                "/media/results/readiness.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(bad_hash.to_string().contains("64 lowercase"));
    }
}

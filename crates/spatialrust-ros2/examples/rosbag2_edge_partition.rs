//! Execute a bounded live-publish receipt across explicit edge/host partitions.
//!
//! This example consumes the JSON receipt emitted by rosbag2_live_publish. It
//! does not invent a network transport: the existing spatialrust-distribute
//! graph, named transfer, and bounded queue contracts are exercised directly,
//! and the resulting portable receipt keeps source/frame/calibration admission
//! fail-closed.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use spatialrust_distribute::{
    BackpressurePolicy, BackpressureSignal, BoundedTransferQueue,
    ExecutionPartition as DistributePartition, NamedTransfer, PartitionGraph, TransferDirection,
    TransferKind, TransferLedger, TransferPlan,
};
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_viewer::{
    EdgePartition, EdgePartitionState, EdgePartitionSummary, EdgePartitionTransfer,
    LivePublishState, ReplayArtifact, StudioSource,
};

const DEFAULT_QUEUE_CAPACITY: usize = 2;
const STATE_FILE: &str = "edge-partition.json";
const HTML_FILE: &str = "edge-partition.html";
const MANIFEST_FILE: &str = "edge-partition.manifest.json";

#[derive(Debug)]
struct Config {
    live_publish_json: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    calibration_readiness: PathBuf,
    queue_capacity: usize,
    min_output_free_bytes: u64,
}

#[derive(Debug, Default)]
struct PartitionRun {
    transfers: Vec<EdgePartitionTransfer>,
    payload_bytes: u64,
    counted_copy_bytes: u64,
    max_queue_depth: u64,
    soft_limit_trips: u64,
    hard_rejects: u64,
    deterministic_order_verified: bool,
}

#[derive(Debug)]
struct ReadinessGate {
    source_bound: bool,
    registration_ready: bool,
    blockers: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-edge-partition: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    validate_config(&config)?;
    ensure_outputs_absent(&config)?;

    let output_parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    let preflight = StoragePreflight::check(output_parent, config.min_output_free_bytes)?;

    let live_receipt = FileReceipt::from_path(ReceiptRole::Input, &config.live_publish_json)?;
    let live_state: LivePublishState = read_json(&config.live_publish_json)?;
    live_state.validate()?;
    let source_path = PathBuf::from(&live_state.source.path);
    if !source_path.is_absolute() {
        return Err("upstream live-publish source path must be absolute".into());
    }
    let source_size = fs::metadata(&source_path)?.len();
    let observed_sha256 = live_state.source.observed_sha256.clone();
    let source_identity_match = observed_sha256 == config.expected_sha256;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        live_state.source.path.clone(),
        config.expected_sha256.clone(),
        observed_sha256.clone(),
        source_identity_match,
    )?;

    let readiness_receipt =
        FileReceipt::from_path(ReceiptRole::Auxiliary, &config.calibration_readiness)?;
    let readiness: Value = read_json(&config.calibration_readiness)?;
    let readiness_gate = readiness_gate(&readiness, &source.path, &observed_sha256, source_size)?;

    let mut blockers = Vec::new();
    if !source_identity_match {
        push_blocker(
            &mut blockers,
            format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            ),
        );
    }
    if !live_state.publish_ready {
        push_blocker(&mut blockers, "upstream live-publish receipt is not publish-ready");
    }
    if !live_state.summary.frame_identity_match {
        push_blocker(&mut blockers, "upstream live-publish frame identity was not admitted");
    }
    if !live_state.summary.deterministic_order_verified {
        push_blocker(&mut blockers, "upstream live-publish order was not verified");
    }
    if live_state.packets.is_empty() {
        push_blocker(&mut blockers, "upstream live-publish receipt contains no packets");
    }
    blockers.extend(readiness_gate.blockers.iter().cloned());
    if !readiness_gate.source_bound {
        push_blocker(&mut blockers, "calibration readiness receipt is not bound to this input");
    } else if !readiness_gate.registration_ready {
        push_blocker(
            &mut blockers,
            "calibration readiness registration is incomplete; mapping remains blocked",
        );
    }
    if !live_state.summary.calibration_applied {
        push_blocker(
            &mut blockers,
            "clock/TF calibration was not applied; partition results remain inspection-only",
        );
    }

    let graph_partitions = build_graph()?;
    let can_partition = source_identity_match
        && live_state.publish_ready
        && live_state.summary.frame_identity_match
        && live_state.summary.deterministic_order_verified
        && !live_state.packets.is_empty();
    let partition_run =
        run_partition(&live_state, &graph_partitions.graph, config.queue_capacity, can_partition)?;
    if partition_run.hard_rejects > 0 {
        push_blocker(
            &mut blockers,
            format!(
                "edge-to-host transfer queue rejected {} packet(s) at its hard limit",
                partition_run.hard_rejects
            ),
        );
    }
    if can_partition && partition_run.transfers.len() != live_state.packets.len() {
        push_blocker(
            &mut blockers,
            "edge-to-host transfer count did not cover every admitted live-publish packet",
        );
    }

    let calibration_registered = readiness_gate.source_bound
        && readiness_gate.registration_ready
        && live_state.summary.calibration_registered;
    let calibration_applied = calibration_registered && live_state.summary.calibration_applied;
    let summary = EdgePartitionSummary::try_new(
        u64::try_from(live_state.packets.len())?,
        u64::try_from(partition_run.transfers.len())?,
        u64::try_from(
            partition_run.transfers.iter().filter(|transfer| transfer.completed).count(),
        )?,
        partition_run.payload_bytes,
        partition_run.counted_copy_bytes,
        partition_run.max_queue_depth,
        partition_run.soft_limit_trips,
        partition_run.hard_rejects,
        partition_run.deterministic_order_verified,
        live_state.publish_ready,
        source_identity_match,
        live_state.summary.frame_identity_match,
        calibration_registered,
        calibration_applied,
        live_state.summary.time_basis.clone(),
    )?;

    fs::create_dir_all(&config.output_dir)?;
    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    let artifacts = vec![
        replay_artifact("upstream-live-publish", &live_receipt)?,
        replay_artifact("calibration-readiness", &readiness_receipt)?,
    ];
    let state = EdgePartitionState::try_new(
        format!("Edge Partition Execution — {}", file_label(&config.live_publish_json)),
        source,
        config.live_publish_json.display().to_string(),
        config.calibration_readiness.display().to_string(),
        graph_partitions.receipt,
        partition_run.transfers,
        summary,
        artifacts,
        blockers,
    )?;
    state.validate()?;
    write_json_atomically(&state_path, &state)?;
    write_text_atomically(&html_path, &render_dashboard(&state)?)?;

    let mut manifest = DatasetManifest::new();
    manifest.entries.push(live_receipt);
    manifest.entries.push(readiness_receipt);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Edge Partition receipt: {} (partition_ready={}, mapping_admitted={})",
        state_path.display(),
        state.partition_ready,
        state.mapping_admitted
    );
    println!("Edge Partition dashboard: {}", html_path.display());
    println!(
        "Edge Partition manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.partition_ready {
        return Err("edge partition failed its source, upstream, graph, or queue gates".into());
    }
    Ok(())
}

struct GraphBuild {
    graph: PartitionGraph,
    receipt: Vec<EdgePartition>,
}

fn build_graph() -> Result<GraphBuild, Box<dyn Error>> {
    let edge =
        DistributePartition::try_new("edge", vec!["live-publish".into(), "packet-gate".into()])?;
    let host = DistributePartition::try_new("host", vec!["host-receive".into(), "mapping".into()])?;
    let mut graph = PartitionGraph::new();
    graph.insert_partition(edge)?;
    graph.insert_partition(host)?;
    graph.connect("edge", "host")?;
    if graph.topological_order()? != vec!["edge".to_owned(), "host".to_owned()] {
        return Err("edge partition graph order was not deterministic".into());
    }
    Ok(GraphBuild {
        graph,
        receipt: vec![
            EdgePartition::try_new(
                "edge",
                "edge-host-0",
                vec!["live-publish".into(), "packet-gate".into()],
            )?,
            EdgePartition::try_new(
                "host",
                "mapping-host-0",
                vec!["host-receive".into(), "mapping".into()],
            )?,
        ],
    })
}

fn run_partition(
    live_state: &LivePublishState,
    graph: &PartitionGraph,
    queue_capacity: usize,
    admitted: bool,
) -> Result<PartitionRun, Box<dyn Error>> {
    let mut plan = TransferPlan::new();
    if admitted {
        for packet in &live_state.packets {
            plan.push(NamedTransfer::try_new(
                format!("packet-{:06}", packet.sequence),
                TransferDirection::HostToNetwork,
                TransferKind::ExplicitCopy,
                "packet-gate",
                "host-receive",
                packet.payload_bytes,
            )?);
        }
    }
    plan.validate_against(graph)?;
    if !admitted {
        return Ok(PartitionRun::default());
    }

    let policy = BackpressurePolicy::try_new(queue_capacity - 1, queue_capacity)?;
    let mut queue = BoundedTransferQueue::new(policy);
    let mut ledger = TransferLedger::new();
    let mut run = PartitionRun {
        deterministic_order_verified: live_state
            .packets
            .iter()
            .enumerate()
            .all(|(index, packet)| packet.sequence == u64::try_from(index).unwrap_or(u64::MAX)),
        ..PartitionRun::default()
    };
    let mut transfer_receipts = Vec::<(NamedTransfer, String)>::new();
    for chunk in plan.transfers().chunks(queue_capacity) {
        for transfer in chunk {
            let signal = match queue.try_push(transfer.clone()) {
                Ok(signal) => signal,
                Err(_) => {
                    run.deterministic_order_verified = false;
                    break;
                }
            };
            run.max_queue_depth =
                run.max_queue_depth.max(u64::try_from(queue.depth()).unwrap_or(u64::MAX));
            transfer_receipts.push((transfer.clone(), signal_name(signal).to_owned()));
        }
        while let Some(transfer) = queue.pop() {
            let (_, queue_signal) = transfer_receipts.remove(0);
            ledger.record(transfer.clone());
            run.transfers.push(EdgePartitionTransfer::try_new(
                u64::try_from(run.transfers.len())?,
                packet_topic(live_state, run.transfers.len())?,
                transfer.from.clone(),
                transfer.to.clone(),
                transfer.bytes,
                transfer.counted_copy_bytes(),
                queue_signal,
                true,
            )?);
        }
        if queue.hard_rejects() > 0 {
            break;
        }
    }
    run.payload_bytes = run.transfers.iter().try_fold(0_u64, |total, transfer| {
        total
            .checked_add(transfer.payload_bytes)
            .ok_or("edge partition payload byte count overflow")
    })?;
    run.counted_copy_bytes = ledger.counted_copy_bytes();
    run.soft_limit_trips = queue.soft_trips();
    run.hard_rejects = queue.hard_rejects();
    if run.transfers.len() != live_state.packets.len() {
        run.deterministic_order_verified = false;
    }
    Ok(run)
}

fn packet_topic(live_state: &LivePublishState, sequence: usize) -> Result<String, Box<dyn Error>> {
    live_state
        .packets
        .get(sequence)
        .map(|packet| packet.source_topic.clone())
        .ok_or_else(|| "live-publish packet sequence is missing".into())
}

fn signal_name(signal: BackpressureSignal) -> &'static str {
    match signal {
        BackpressureSignal::Ok => "ok",
        BackpressureSignal::SoftLimit => "soft-limit",
        BackpressureSignal::HardLimit => "hard-limit",
    }
}

fn readiness_gate(
    readiness: &Value,
    input_path: &str,
    observed_sha256: &str,
    input_size: u64,
) -> Result<ReadinessGate, Box<dyn Error>> {
    let mut blockers = Vec::new();
    if readiness.get("schema").and_then(Value::as_str)
        != Some("spatialrust.rosbag2.calibration-readiness")
    {
        blockers.push("calibration readiness schema is unsupported".into());
    }
    if readiness.get("version").and_then(Value::as_u64) != Some(1) {
        blockers.push("calibration readiness version is unsupported".into());
    }
    let source_path = string_at(readiness, &["input", "path"]);
    let source_sha = string_at(readiness, &["input", "sha256"]);
    let source_size = readiness.get("input").and_then(|input| input.get("size_bytes"));
    let source_bound = source_path.as_deref() == Some(input_path)
        && source_sha.as_deref() == Some(observed_sha256)
        && source_size.and_then(Value::as_u64) == Some(input_size);
    if !source_bound {
        blockers
            .push("calibration readiness input identity does not match the canonical bag".into());
    }
    let registration_ready =
        readiness.get("registration_ready").and_then(Value::as_bool).unwrap_or(false);
    for blocker in string_array(readiness.get("blockers")) {
        push_blocker(&mut blockers, format!("readiness: {blocker}"));
    }
    Ok(ReadinessGate { source_bound, registration_ready, blockers })
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

fn replay_artifact(role: &str, receipt: &FileReceipt) -> Result<ReplayArtifact, Box<dyn Error>> {
    ReplayArtifact::try_new(
        role,
        receipt.path.display().to_string(),
        receipt.size_bytes.ok_or("artifact size is missing")?,
        receipt.sha256.clone().ok_or("artifact checksum is missing")?,
    )
    .map_err(Into::into)
}

fn render_dashboard(state: &EdgePartitionState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#06111c;--panel:#0d2134;--line:#24506f;--muted:#8ea9bd;--cyan:#5bdcff;--green:#63e7a5;--red:#ff7184;--amber:#ffd166}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#1c5271 0,#06111c 45%);color:#eef9ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.title{font-size:30px;font-weight:750;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:750;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(16,43,67,.96),rgba(7,20,34,.96));border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 14px 32px #0004}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:25px;font-weight:750;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:#10324d;border:1px solid #2d638a;border-radius:999px;padding:6px 10px;font-size:12px}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #183b55;padding:9px 0}.row:last-child{border-bottom:0}.danger{color:var(--red)}.warning{color:var(--amber)}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.trace{max-height:320px;overflow:auto}.trace .row{display:grid;grid-template-columns:42px 1.1fr 1fr 1fr auto}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}}
</style></head><body><main><section class="top"><div><div class="eyebrow">SpatialRust / edge partition execution</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section><section class="grid">
<article class="panel"><h2>Partition readiness</h2><div id="ready" class="metric"></div><div id="readyDetail" class="small"></div></article><article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article><article class="panel"><h2>Transfers</h2><div id="transfers" class="metric"></div><div id="transferDetail" class="small"></div></article><article class="panel"><h2>Queue</h2><div id="queue" class="metric"></div><div id="queueDetail" class="small"></div></article><article class="panel wide"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="small mono"></div><div id="path" class="small mono"></div></article><article class="panel wide"><h2>Partition graph</h2><div id="partitions" class="chips"></div></article><article class="panel wide"><h2>Named transfer trace</h2><div id="trace" class="trace"></div></article><article class="panel wide"><h2>Transfer receipt</h2><div id="resources"></div></article><article class="panel full"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article><article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:300px;overflow:auto;white-space:pre-wrap"></pre></article></section></main>
<script id="edge-partition-state" type="application/json">__STATE_JSON__</script><script>
const state=JSON.parse(document.getElementById('edge-partition-state').textContent),q=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString(),fmtBytes=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB',esc=v=>String(v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));q('title').textContent=state.title;q('source').textContent=state.source.path;q('admission').textContent=state.partition_ready?'PARTITION READY':'PARTITION BLOCKED';q('admission').className='badge '+(state.partition_ready?'ok':'blocked');q('ready').textContent=state.partition_ready?'READY':'BLOCKED';q('ready').className='metric '+(state.partition_ready?'':'danger');q('readyDetail').textContent=state.summary.source_packet_count+' source packets · '+state.summary.completed_transfer_count+' completed';q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';q('transfers').textContent=fmt(state.summary.completed_transfer_count)+' / '+fmt(state.summary.source_packet_count);q('transferDetail').textContent=fmtBytes(state.summary.counted_copy_bytes)+' explicit copies';q('queue').textContent='depth '+fmt(state.summary.max_queue_depth);q('queueDetail').textContent=fmt(state.summary.soft_limit_trips)+' soft · '+fmt(state.summary.hard_rejects)+' hard';q('identity').textContent=state.source.identity_matches?'MATCH':'MISMATCH';q('identity').className='metric '+(state.source.identity_matches?'':'danger');q('sha').textContent=state.source.observed_sha256;q('path').textContent=state.source.path;q('partitions').innerHTML=state.partitions.map(p=>'<span class="chip">'+esc(p.id)+' · '+esc(p.placement)+' · '+p.node_ids.map(esc).join(', ')+'</span>').join('');q('trace').innerHTML=state.transfers.map(v=>'<div class="row"><span class="mono">#'+v.sequence+'</span><span>'+esc(v.source_topic)+'</span><span>'+esc(v.from_node)+' → '+esc(v.to_node)+'</span><span>'+fmtBytes(v.payload_bytes)+' · '+esc(v.queue_signal)+'</span><span>'+(v.completed?'✓':'blocked')+'</span></div>').join('')||'<div class="small">No transfers admitted</div>';q('resources').innerHTML='<div class="row"><span>payload bytes</span><span class="mono">'+fmtBytes(state.summary.payload_bytes)+'</span></div><div class="row"><span>explicit-copy bytes</span><span class="mono">'+fmtBytes(state.summary.counted_copy_bytes)+'</span></div><div class="row"><span>time basis</span><span class="mono">'+esc(state.summary.time_basis)+'</span></div>';q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="ok">All gates passed</li>';q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
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

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.live_publish_json.is_absolute()
        || !config.output_dir.is_absolute()
        || !config.calibration_readiness.is_absolute()
    {
        return Err(
            "live-publish JSON, --output-dir, and --calibration-readiness paths must be absolute"
                .into(),
        );
    }
    if config.queue_capacity < 2 {
        return Err("--queue-capacity must be at least 2".into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let live_publish_json = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut calibration_readiness = None;
    let mut queue_capacity = DEFAULT_QUEUE_CAPACITY;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--calibration-readiness" => {
                calibration_readiness = Some(PathBuf::from(next_value(&mut args, &flag)?))
            }
            "--queue-capacity" => queue_capacity = next_value(&mut args, &flag)?.parse()?,
            "--min-output-free-bytes" => {
                min_output_free_bytes = next_value(&mut args, &flag)?.parse()?
            }
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        live_publish_json: PathBuf::from(live_publish_json),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        calibration_readiness: calibration_readiness
            .ok_or("--calibration-readiness is required")?,
        queue_capacity,
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
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
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

fn file_label(path: &Path) -> String {
    path.file_name().and_then(|name| name.to_str()).unwrap_or("receipt").to_owned()
}

fn escape_html(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn usage() -> String {
    String::from(
        "usage: rosbag2_edge_partition <absolute-live-publish.json> \\\n  --output-dir <absolute-new-dir> \\\n  --expected-input-sha256 <sha256> \\\n  --calibration-readiness <absolute-readiness.json> [options]\n\noptions:\n  --queue-capacity <count>\n  --min-output-free-bytes <bytes>",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_edge_partition_options() {
        let config = parse_args(
            [
                "/media/live-publish.json",
                "--output-dir",
                "/media/output",
                "--expected-input-sha256",
                SHA,
                "--calibration-readiness",
                "/media/readiness.json",
                "--queue-capacity",
                "4",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.queue_capacity, 4);
        assert_eq!(config.min_output_free_bytes, DEFAULT_MIN_OUTPUT_FREE_BYTES);
    }

    #[test]
    fn rejects_single_slot_queue() {
        let config = Config {
            live_publish_json: "/media/live-publish.json".into(),
            output_dir: "/media/output".into(),
            expected_sha256: SHA.into(),
            calibration_readiness: "/media/readiness.json".into(),
            queue_capacity: 1,
            min_output_free_bytes: 1,
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn graph_is_edge_to_host() {
        let graph = build_graph().unwrap();
        assert_eq!(
            graph.graph.topological_order().unwrap(),
            vec!["edge".to_owned(), "host".to_owned()]
        );
        assert_eq!(graph.graph.edges(), &[("edge".to_owned(), "host".to_owned())]);
    }
}

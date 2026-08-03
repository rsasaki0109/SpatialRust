//! Run a bounded deterministic rosbag2 replay and emit a portable dashboard.
//!
//! The command keeps the canonical input read-only, requires an exact SHA-256
//! identity, and writes all derived state to one new output directory. Replay
//! admission is independent from mapping admission: an uncalibrated episode
//! can be replayed for inspection, but it cannot silently become fused world
//! geometry.

use std::{
    collections::{BTreeSet, HashSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::Serialize;
use spatialrust_io::{
    DatasetManifest, FileReceipt, ReceiptRole, StoragePreflight, DEFAULT_MIN_OUTPUT_FREE_BYTES,
};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
};
use spatialrust_ros2::{list_topics, Rosbag2PointCloudSource, Rosbag2Topic};
use spatialrust_sync::{
    ClockDomain, DeterministicReplayer, EpisodeLimits, MemoryEpisode, MemoryEpisodeBuilder,
    StampedRecord, StampedTime, SyncWindow, TopicId,
};
use spatialrust_viewer::{
    ReplayArtifact, ReplayDemoState, ReplaySample, ReplaySummary, ReplayTopic, StudioSource,
};

const CHUNK_POINTS: usize = 65_536;
const SOURCE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
const EPISODE_MAX_POINTS: u64 = 2_000_000;
const EPISODE_MAX_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_MAX_RECORDS_PER_TOPIC: u64 = 2;
const DEFAULT_MAX_DELTA_NS: u64 = 100_000_000;
const STATE_FILE: &str = "replay-demo.json";
const HTML_FILE: &str = "replay-demo.html";
const MANIFEST_FILE: &str = "replay-demo.manifest.json";
const TIME_BASIS: &str =
    "PointCloud2 header stamp; ros2-external domain; no clock calibration applied";

#[derive(Debug)]
struct Config {
    input: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    front_topic: String,
    rear_topic: String,
    max_records_per_topic: u64,
    max_delta_ns: u64,
    min_output_free_bytes: u64,
}

#[derive(Debug)]
struct ReplayRun {
    topics: Vec<ReplayTopic>,
    summary: ReplaySummary,
    samples: Vec<ReplaySample>,
    blockers: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-replay-demo: {error}");
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
            "replay output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }
    fs::create_dir_all(&config.output_dir)?;

    let input = FileReceipt::from_path(ReceiptRole::Input, &config.input)?;
    let observed_sha256 = input.sha256.clone().ok_or("input checksum was not produced")?;
    let identity_matches = observed_sha256 == config.expected_sha256;
    let source = StudioSource::try_new(
        "canonical rosbag2 input",
        config.input.display().to_string(),
        config.expected_sha256.clone(),
        observed_sha256.clone(),
        identity_matches,
    )?;
    let available_topics = list_topics(&config.input)?;
    let replay = if identity_matches {
        bounded_replay(&config, &available_topics)?
    } else {
        ReplayRun {
            topics: requested_topic_inventory(&available_topics, &config),
            summary: ReplaySummary::try_new(
                0,
                0,
                0,
                0,
                0,
                0,
                config.max_delta_ns,
                0,
                0,
                false,
                TIME_BASIS,
                false,
            )?,
            samples: Vec::new(),
            blockers: vec![format!(
                "input SHA-256 mismatch: expected {}, observed {}",
                config.expected_sha256, observed_sha256
            )],
        }
    };

    let mut blockers = replay.blockers;
    if identity_matches {
        push_blocker(
            &mut blockers,
            "clock calibration not applied; replay remains in the header-stamp domain",
        );
        push_blocker(&mut blockers, "TF/frame composition not applied; replay is inspection-only");
        push_blocker(&mut blockers, "mapping admission requires source-bound calibration evidence");
    }
    let state_path = config.output_dir.join(STATE_FILE);
    let html_path = config.output_dir.join(HTML_FILE);
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    let input_artifact = replay_artifact("input", &input)?;
    let state = ReplayDemoState::try_new(
        format!("One-command Replay Demo — {}", file_label(&config.input)),
        source,
        replay.topics,
        replay.summary,
        replay.samples,
        vec![input_artifact],
        blockers,
    )?;
    state.validate()?;
    write_json_atomically(&state_path, &state)?;
    write_json_atomically(&html_path, &render_dashboard(&state)?)?;

    let state_receipt = FileReceipt::from_path(ReceiptRole::Output, &state_path)?;
    let html_receipt = FileReceipt::from_path(ReceiptRole::Output, &html_path)?;
    let mut manifest = DatasetManifest::new();
    manifest.entries.push(input.clone());
    manifest.entries.push(state_receipt);
    manifest.entries.push(html_receipt);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Replay demo: {} (replay_ready={}, mapping_admitted={})",
        state_path.display(),
        state.replay_ready,
        state.mapping_admitted
    );
    println!("Replay dashboard: {}", html_path.display());
    println!(
        "Replay manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.replay_ready {
        return Err("replay demo failed its source or deterministic replay admission checks".into());
    }
    Ok(())
}

fn bounded_replay(
    config: &Config,
    available_topics: &[Rosbag2Topic],
) -> Result<ReplayRun, Box<dyn Error>> {
    let mut blockers = Vec::new();
    let front_present = available_topics.iter().any(|topic| topic.name == config.front_topic);
    let rear_present = available_topics.iter().any(|topic| topic.name == config.rear_topic);
    if !front_present {
        push_blocker(&mut blockers, format!("topic '{}' is missing", config.front_topic));
    }
    if !rear_present {
        push_blocker(&mut blockers, format!("topic '{}' is missing", config.rear_topic));
    }
    if !blockers.is_empty() {
        return Ok(ReplayRun {
            topics: requested_topic_inventory(available_topics, config),
            summary: empty_summary(config.max_delta_ns)?,
            samples: Vec::new(),
            blockers,
        });
    }

    let max_records =
        config.max_records_per_topic.checked_mul(2).ok_or("episode record limit overflow")?;
    let limits = EpisodeLimits::new(max_records, EPISODE_MAX_POINTS, EPISODE_MAX_BYTES);
    let mut builder = MemoryEpisodeBuilder::try_new(limits)?;
    let front = append_topic(
        &mut builder,
        &config.input,
        &config.front_topic,
        config.max_records_per_topic,
    )?;
    let rear = append_topic(
        &mut builder,
        &config.input,
        &config.rear_topic,
        config.max_records_per_topic,
    )?;
    let episode_record_count = u64::try_from(builder.len())?;
    let episode_point_count = builder.points();
    let episode_byte_count = builder.bytes();
    let peak_source_bytes = front.peak_source_bytes().max(rear.peak_source_bytes());
    let topics = vec![front.into_topic(), rear.into_topic()];
    let episode = builder.finish();
    let replay_started = Instant::now();
    let (samples, matched_bundle_count, max_matched_delta_ns, deterministic_order_verified) =
        replay_trace(&episode, config)?;
    let replay_wall_ns = elapsed_wall_ns(replay_started)?;
    let replayed_record_count = u64::try_from(samples.len())?;
    if episode_record_count == 0 {
        push_blocker(&mut blockers, "bounded episode retained no records");
    }
    if matched_bundle_count == 0 {
        push_blocker(&mut blockers, "no front/rear bundle matched within the sync window");
    }
    if !deterministic_order_verified {
        push_blocker(&mut blockers, "deterministic replay order verification failed");
    }
    let summary = ReplaySummary::try_new(
        episode_record_count,
        episode_point_count,
        episode_byte_count,
        replayed_record_count,
        matched_bundle_count,
        max_matched_delta_ns,
        config.max_delta_ns,
        replay_wall_ns,
        peak_source_bytes,
        deterministic_order_verified,
        TIME_BASIS,
        false,
    )?;
    Ok(ReplayRun { topics, summary, samples, blockers })
}

struct TopicRun {
    topic: ReplayTopic,
    peak_source_bytes: u64,
}

impl TopicRun {
    fn peak_source_bytes(&self) -> u64 {
        self.peak_source_bytes
    }

    fn into_topic(self) -> ReplayTopic {
        self.topic
    }
}

fn append_topic(
    builder: &mut MemoryEpisodeBuilder,
    input: &Path,
    topic_name: &str,
    max_records: u64,
) -> Result<TopicRun, Box<dyn Error>> {
    let options = StreamOptions::new(CHUNK_POINTS, MemoryBudget::new(SOURCE_MEMORY_BYTES)?)?;
    let mut source =
        Rosbag2PointCloudSource::open(input, topic_name, options, CancellationToken::default())?;
    let mut retained_records = 0_u64;
    let mut retained_points = 0_u64;
    let mut frame_ids = BTreeSet::new();
    while retained_records < max_records {
        let Some(chunk) = source.next_chunk() else {
            break;
        };
        let chunk = chunk?;
        let record = chunk.record().clone();
        frame_ids.insert(record.metadata().frame_id.0.clone());
        retained_points = retained_points
            .checked_add(u64::try_from(record.cloud().len())?)
            .ok_or("topic point count overflow")?;
        let stamp = StampedTime::exact("ros2", ClockDomain::External, record.metadata().timestamp);
        builder.push(StampedRecord::new(topic_name, stamp, record))?;
        retained_records =
            retained_records.checked_add(1).ok_or("topic retained record count overflow")?;
    }
    let peak_source_bytes = source.memory_tracker().snapshot().peak_bytes;
    Ok(TopicRun {
        topic: ReplayTopic::try_new(
            topic_name,
            source.topic().message_count,
            retained_records,
            retained_points,
            frame_ids.into_iter().collect(),
        )?,
        peak_source_bytes,
    })
}

fn replay_trace(
    episode: &MemoryEpisode,
    config: &Config,
) -> Result<(Vec<ReplaySample>, u64, u64, bool), Box<dyn Error>> {
    let front_id = TopicId::new(config.front_topic.clone());
    let rear_id = TopicId::new(config.rear_topic.clone());
    let window = SyncWindow { max_delta_ns: config.max_delta_ns, max_uncertainty_ns: 0 };
    let mut pair_keys = HashSet::new();
    let mut matcher = DeterministicReplayer::new(episode);
    let mut matched_bundle_count = 0_u64;
    let mut max_matched_delta_ns = 0_u64;
    while let Some(bundle) =
        matcher.next_bundle(&front_id, std::slice::from_ref(&rear_id), window)?
    {
        let front = bundle.get(&front_id).ok_or("front bundle member missing")?;
        let rear = bundle.get(&rear_id).ok_or("rear bundle member missing")?;
        pair_keys.insert((front.stamp.as_nanos(), front.topic.as_str().to_owned()));
        pair_keys.insert((rear.stamp.as_nanos(), rear.topic.as_str().to_owned()));
        matched_bundle_count =
            matched_bundle_count.checked_add(1).ok_or("matched bundle count overflow")?;
        max_matched_delta_ns = max_matched_delta_ns.max(front.stamp.abs_delta_ns(&rear.stamp));
    }

    let expected_order: Vec<_> = episode
        .records()
        .iter()
        .map(|record| (record.stamp.as_nanos(), record.topic.as_str().to_owned()))
        .collect();
    let mut replayer = DeterministicReplayer::new(episode);
    let mut observed_order = Vec::new();
    let mut samples = Vec::new();
    while let Some(record) = replayer.next_record() {
        let topic = record.topic.as_str().to_owned();
        observed_order.push((record.stamp.as_nanos(), topic.clone()));
        let paired_topics = if pair_keys.contains(&(record.stamp.as_nanos(), topic.clone())) {
            if topic == config.front_topic {
                vec![config.rear_topic.clone()]
            } else {
                vec![config.front_topic.clone()]
            }
        } else {
            Vec::new()
        };
        samples.push(ReplaySample::try_new(
            u64::try_from(samples.len())?,
            topic,
            record.stamp.as_nanos(),
            u64::try_from(record.record.cloud().len())?,
            paired_topics,
        )?);
    }
    Ok((samples, matched_bundle_count, max_matched_delta_ns, observed_order == expected_order))
}

fn requested_topic_inventory(topics: &[Rosbag2Topic], config: &Config) -> Vec<ReplayTopic> {
    [config.front_topic.as_str(), config.rear_topic.as_str()]
        .into_iter()
        .filter_map(|name| {
            topics.iter().find(|topic| topic.name == name).and_then(|topic| {
                ReplayTopic::try_new(topic.name.clone(), topic.message_count, 0, 0, Vec::new()).ok()
            })
        })
        .collect()
}

fn empty_summary(max_delta_ns: u64) -> Result<ReplaySummary, Box<dyn Error>> {
    Ok(ReplaySummary::try_new(0, 0, 0, 0, 0, 0, max_delta_ns, 0, 0, false, TIME_BASIS, false)?)
}

fn replay_artifact(role: &str, receipt: &FileReceipt) -> Result<ReplayArtifact, Box<dyn Error>> {
    ReplayArtifact::try_new(
        role,
        receipt.path.display().to_string(),
        receipt.size_bytes.ok_or("replay artifact size is missing")?,
        receipt.sha256.clone().ok_or("replay artifact checksum is missing")?,
    )
    .map_err(Into::into)
}

fn render_dashboard(state: &ReplayDemoState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--panel:#0d1d31;--line:#21415f;--muted:#8ba6be;--cyan:#55d8ff;--green:#61e7a4;--red:#ff7180;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173d63 0,#07101d 45%);color:#edf7ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.title{font-size:30px;font-weight:750;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:750;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(16,39,64,.96),rgba(8,20,35,.96));border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 14px 32px #0004}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:25px;font-weight:750;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:#102d49;border:1px solid #2c5b83;border-radius:999px;padding:6px 10px;font-size:12px}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #183550;padding:9px 0}.row:last-child{border-bottom:0}.danger{color:var(--red)}.warning{color:var(--amber)}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.trace{max-height:270px;overflow:auto}.trace .row{display:grid;grid-template-columns:42px 1fr auto auto}.bar{height:9px;background:#07101d;border-radius:5px;overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / deterministic replay trace</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Replay readiness</h2><div id="ready" class="metric"></div><div id="readyDetail" class="small"></div></article>
<article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Episode</h2><div id="records" class="metric"></div><div id="points" class="small"></div></article>
<article class="panel"><h2>Sync bundles</h2><div id="bundles" class="metric"></div><div id="delta" class="small"></div></article>
<article class="panel wide"><h2>Source identity</h2><div id="identity" class="metric"></div><div id="sha" class="small mono"></div><div id="path" class="small mono"></div></article>
<article class="panel wide"><h2>Topic inventory</h2><div id="topics" class="chips"></div></article>
<article class="panel wide"><h2>Replay timeline</h2><div id="timeline" class="small"></div><div class="bar"><div id="timelineBar" class="fill"></div></div></article>
<article class="panel wide"><h2>Trace samples</h2><div id="trace" class="trace"></div></article>
<article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel wide"><h2>Explicit resource metrics</h2><div id="resources"></div></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:250px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="replay-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('replay-state').textContent),q=id=>document.getElementById(id),fmtNs=n=>(n/1e9).toFixed(3)+' s',fmtBytes=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB',esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
q('title').textContent=state.title;q('source').textContent=state.source.path;q('admission').textContent=state.replay_ready?'REPLAY READY':'REPLAY BLOCKED';q('admission').className='badge '+(state.replay_ready?'ok':'blocked');
q('ready').textContent=state.replay_ready?'READY':'BLOCKED';q('ready').className='metric '+(state.replay_ready?'':'danger');q('readyDetail').textContent=state.summary.deterministic_order_verified?'deterministic order verified':'order verification unavailable';
q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';
q('records').textContent=state.summary.replayed_record_count.toLocaleString()+' / '+state.summary.episode_record_count.toLocaleString();q('points').textContent=state.summary.episode_point_count.toLocaleString()+' points · '+fmtBytes(state.summary.episode_byte_count);
q('bundles').textContent=state.summary.matched_bundle_count.toLocaleString();q('delta').textContent='max delta '+(state.summary.max_matched_delta_ns/1e6).toFixed(3)+' ms / '+(state.summary.max_delta_ns/1e6).toFixed(3)+' ms window';
q('identity').textContent=state.source.identity_matches?'MATCH':'MISMATCH';q('identity').className='metric '+(state.source.identity_matches?'':'danger');q('sha').textContent=state.source.observed_sha256;q('path').textContent=state.source.path;
q('topics').innerHTML=state.topics.map(t=>'<span class="chip">'+esc(t.name)+' · '+t.retained_record_count.toLocaleString()+' records · '+t.retained_point_count.toLocaleString()+' pts</span>').join('')||'<span class="small">No admitted topic records</span>';
const s=state.samples,t=state.summary,start=s.length?s[0].stamp_nanos:0,end=s.length?s[s.length-1].stamp_nanos:start;q('timeline').textContent=s.length?((start/1e9).toFixed(6)+' → '+(end/1e9).toFixed(6)+' s · '+s.length.toLocaleString()+' ordered samples · '+state.summary.time_basis):'unavailable';q('timelineBar').style.width=s.length?((s.length/(Math.max(1,state.summary.episode_record_count)))*100)+'%':'0%';
q('trace').innerHTML=s.map(v=>'<div class="row"><span class="mono">#'+v.sequence+'</span><span>'+esc(v.topic)+'</span><span class="mono">'+(v.stamp_nanos/1e9).toFixed(6)+' s</span><span class="small">'+v.point_count.toLocaleString()+' pts'+(v.paired_topics.length?' · paired':'')+'</span></div>').join('')||'<div class="small">No replay samples</div>';
q('blockers').innerHTML=state.blockers.map(v=>'<li>'+esc(v)+'</li>').join('')||'<li class="ok">All gates passed</li>';q('resources').innerHTML='<div class="row"><span>replay wall</span><span class="mono">'+fmtNs(state.summary.replay_wall_ns)+'</span></div><div class="row"><span>peak source bytes</span><span class="mono">'+fmtBytes(state.summary.peak_source_bytes)+'</span></div><div class="row"><span>episode bytes</span><span class="mono">'+fmtBytes(state.summary.episode_byte_count)+'</span></div>';q('raw').textContent=JSON.stringify(state,null,2);
</script></body></html>
"##;
    Ok(template.replace("__TITLE__", &title).replace("__STATE_JSON__", &state_json))
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.input.is_absolute() || !config.output_dir.is_absolute() {
        return Err("input and --output-dir paths must be absolute".into());
    }
    if config.output_dir == Path::new("/") {
        return Err("--output-dir must not be the filesystem root".into());
    }
    let parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    if !parent.is_dir() {
        return Err(format!("output parent '{}' is not a directory", parent.display()).into());
    }
    if config.front_topic == config.rear_topic {
        return Err("front and rear topics must differ".into());
    }
    if config.max_records_per_topic == 0 {
        return Err("--max-records must be greater than zero".into());
    }
    if config.max_delta_ns == 0 {
        return Err("--max-delta-ns must be greater than zero".into());
    }
    if config.min_output_free_bytes == 0 {
        return Err("--min-output-free-bytes must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut front_topic = "/lidar_front/points_raw".to_owned();
    let mut rear_topic = "/lidar_rear/points_raw".to_owned();
    let mut max_records_per_topic = DEFAULT_MAX_RECORDS_PER_TOPIC;
    let mut max_delta_ns = DEFAULT_MAX_DELTA_NS;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--front-topic" => front_topic = next_value(&mut args, &flag)?,
            "--rear-topic" => rear_topic = next_value(&mut args, &flag)?,
            "--max-records" => max_records_per_topic = parse_u64(&mut args, &flag)?,
            "--max-delta-ns" => max_delta_ns = parse_u64(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        input: PathBuf::from(input),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        front_topic,
        rear_topic,
        max_records_per_topic,
        max_delta_ns,
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

fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let text = format!("{}\n", serde_json::to_string_pretty(value)?);
    write_text_atomically(path, &text)
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

fn file_label(path: &Path) -> String {
    path.file_name()
        .map_or_else(|| path.display().to_string(), |name| name.to_string_lossy().into())
}

fn elapsed_wall_ns(started: Instant) -> Result<u64, Box<dyn Error>> {
    u64::try_from(started.elapsed().as_nanos()).map_err(|_| "wall-clock duration overflow".into())
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
    "usage: rosbag2_replay_demo INPUT_DB3 --output-dir ABSOLUTE_OUTPUT_DIR \
     --expected-input-sha256 SHA256 [--front-topic TOPIC] [--rear-topic TOPIC] \
     [--max-records N] [--max-delta-ns N] [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, validate_sha256};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_one_command_replay_options() {
        let config = parse_args(
            [
                "/media/input.db3",
                "--output-dir",
                "/media/results/replay",
                "--expected-input-sha256",
                SHA,
                "--max-records",
                "3",
                "--max-delta-ns",
                "5000000",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.max_records_per_topic, 3);
        assert_eq!(config.max_delta_ns, 5_000_000);
        assert_eq!(config.front_topic, "/lidar_front/points_raw");
    }

    #[test]
    fn rejects_relative_outputs_and_bad_hashes() {
        let relative = parse_args(
            ["bag.db3", "--output-dir", "results", "--expected-input-sha256", SHA]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert!(super::validate_config(&relative).is_err());
        assert!(validate_sha256("not-a-sha256").is_err());
    }
}

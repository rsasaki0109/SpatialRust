//! Compare bounded rosbag2 E2E receipts without touching the source bag.
//!
//! The comparison is intentionally receipt/manifest based. It rejects mixed
//! inputs or run configurations, compares deterministic input/episode/glTF
//! hashes, and emits host-specific timing statistics with the bounded-smoke
//! budgets used by the v1.3 acceptance evidence.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const RECEIPT_VERSION: u32 = 2;
const PERFORMANCE_SCHEMA: &str = "spatialrust.rosbag2.e2e.performance";
const PERFORMANCE_VERSION: u32 = 1;
const COMPARISON_SCHEMA: &str = "spatialrust.rosbag2.e2e.comparison";
const COMPARISON_VERSION: u32 = 1;
const MAX_RECEIPTS: usize = 64;
const DETERMINISTIC_HASH_FILES: [&str; 3] =
    ["rosbag2_2020_09_23-15_58_07.db3", "rosbag2.e2e.episode.bin", "tsdf.mesh.gltf"];

const BUDGETS: [(&str, u64); 8] = [
    ("preflight", 100_000_000),
    ("ingest", 500_000_000),
    ("sync", 100_000_000),
    ("odometry", 70_000_000_000),
    ("tsdf", 5_000_000_000),
    ("interchange", 2_000_000_000),
    ("semantic_viewer", 100_000_000),
    ("observed_pipeline", 80_000_000_000),
];

#[derive(Debug)]
struct Config {
    receipts: Vec<PathBuf>,
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RunReceipt {
    schema: String,
    version: u32,
    input: String,
    output_dir: String,
    front_topic: String,
    rear_topic: String,
    run_mode: String,
    performance: PerformanceReceipt,
    ingest: IngestReceipt,
    sync: SyncReceipt,
    odometry: OdometryReceipt,
    tsdf: TsdfReceipt,
    semantic: SemanticReceipt,
    viewer: ViewerReceipt,
    interchange: InterchangeReceipt,
}

#[derive(Debug, Deserialize)]
struct PerformanceReceipt {
    schema: String,
    version: u32,
    observed_pipeline_wall_ns: u64,
    stages: StageTimingReceipt,
    memory: MemoryReceipt,
    transfers: TransferReceipt,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct StageTimingReceipt {
    preflight_wall_ns: u64,
    ingest_wall_ns: u64,
    sync_wall_ns: u64,
    odometry_wall_ns: u64,
    tsdf_wall_ns: u64,
    interchange_wall_ns: u64,
    semantic_viewer_wall_ns: u64,
}

#[derive(Debug, Deserialize)]
struct MemoryReceipt {
    configured_source_budget_bytes: u64,
    configured_episode_budget_bytes: u64,
    retained_episode_bytes: u64,
    peak_source_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct TransferReceipt {
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    hidden_device_copies: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct IngestReceipt {
    retained_records: u64,
    retained_points: u64,
    retained_bytes: u64,
    topics: Vec<TopicReceipt>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct TopicReceipt {
    topic: String,
    schema: String,
    bag_message_count: u64,
    retained_chunks: u64,
    retained_points: u64,
    peak_source_bytes: u64,
    frame_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct SyncReceipt {
    max_records_per_topic: u64,
    max_delta_ns: u64,
    matched_bundles: u64,
    max_matched_delta_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct OdometryReceipt {
    topic: String,
    frame_id: String,
    scans: u64,
    motions: u64,
    pose_graph_nodes: u64,
    pose_graph_edges: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct TsdfReceipt {
    integrated_records: u64,
    integrated_points: u64,
    mesh_vertices: u64,
    mesh_triangles: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct SemanticReceipt {
    entities: u64,
    visible_entities: u64,
    frame_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct ViewerReceipt {
    layers: u64,
    device_upload_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct InterchangeReceipt {
    bytes: u64,
    vertices: u64,
    indices: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct DeterministicIdentity {
    input: String,
    front_topic: String,
    rear_topic: String,
    ingest: IngestReceipt,
    sync: SyncReceipt,
    odometry: OdometryReceipt,
    tsdf: TsdfReceipt,
    semantic: SemanticReceipt,
    viewer: ViewerReceipt,
    interchange: InterchangeReceipt,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct ComparisonReport {
    schema: &'static str,
    version: u32,
    sample_count: usize,
    receipts: Vec<SampleReceipt>,
    deterministic: DeterministicReport,
    performance: PerformanceReport,
    passed: bool,
}

#[derive(Serialize)]
struct SampleReceipt {
    receipt: String,
    output_dir: String,
    run_mode: String,
    observed_pipeline_wall_ns: u64,
}

#[derive(Serialize)]
struct DeterministicReport {
    identity_match: bool,
    hashes_match: bool,
    hashes: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct PerformanceReport {
    stages: Vec<StageStats>,
    memory: MemoryComparison,
    transfers: TransferComparison,
}

#[derive(Serialize)]
struct StageStats {
    stage: &'static str,
    samples_ns: Vec<u64>,
    min_ns: u64,
    median_ns: u64,
    p95_ns: u64,
    max_ns: u64,
    mean_ns: f64,
    stddev_ns: f64,
    coefficient_of_variation: f64,
    budget_ns: u64,
    within_budget: bool,
}

#[derive(Serialize)]
struct MemoryComparison {
    configured_source_budget_bytes: u64,
    configured_episode_budget_bytes: u64,
    retained_episode_bytes: u64,
    peak_source_bytes: u64,
    stable: bool,
}

#[derive(Serialize)]
struct TransferComparison {
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    hidden_device_copies: u64,
    stable: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-e2e-compare: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let mut loaded = Vec::with_capacity(config.receipts.len());
    for path in &config.receipts {
        loaded.push((path.clone(), read_receipt(path)?));
    }
    validate_compatible(&loaded)?;

    let identities = loaded.iter().map(|(_, receipt)| identity(receipt)).collect::<Vec<_>>();
    let identity_match = identities.windows(2).all(|pair| pair[0] == pair[1]);
    if !identity_match {
        return Err("comparison receipts do not describe the same input/configuration".into());
    }

    let hash_sets = loaded
        .iter()
        .map(|(_, receipt)| deterministic_hashes(receipt))
        .collect::<Result<Vec<_>, _>>()?;
    let hashes_match = hash_sets.windows(2).all(|pair| pair[0] == pair[1]);
    let hashes = hash_sets.first().cloned().unwrap_or_default();

    let stages = vec![
        (
            "preflight",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.preflight_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "ingest",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.ingest_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "sync",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.sync_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "odometry",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.odometry_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "tsdf",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.tsdf_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "interchange",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.interchange_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "semantic_viewer",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.stages.semantic_viewer_wall_ns)
                .collect::<Vec<_>>(),
        ),
        (
            "observed_pipeline",
            loaded
                .iter()
                .map(|(_, receipt)| receipt.performance.observed_pipeline_wall_ns)
                .collect::<Vec<_>>(),
        ),
    ];
    let stage_stats = stages
        .into_iter()
        .map(|(name, samples)| {
            let budget_ns = BUDGETS
                .iter()
                .find(|(budget_name, _)| *budget_name == name)
                .map(|(_, budget)| *budget)
                .ok_or_else(|| format!("missing budget for stage '{name}'"))?;
            Ok(stage_stats(name, samples, budget_ns))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let memory = memory_comparison(&loaded);
    let transfers = transfer_comparison(&loaded);
    let budgets_passed = stage_stats.iter().all(|stage| stage.within_budget);
    let memory_stable = memory.stable;
    let transfers_stable = transfers.stable;
    let report = ComparisonReport {
        schema: COMPARISON_SCHEMA,
        version: COMPARISON_VERSION,
        sample_count: loaded.len(),
        receipts: loaded
            .iter()
            .map(|(path, receipt)| SampleReceipt {
                receipt: path.display().to_string(),
                output_dir: receipt.output_dir.clone(),
                run_mode: receipt.run_mode.clone(),
                observed_pipeline_wall_ns: receipt.performance.observed_pipeline_wall_ns,
            })
            .collect(),
        deterministic: DeterministicReport { identity_match, hashes_match, hashes },
        performance: PerformanceReport { stages: stage_stats, memory, transfers },
        passed: identity_match
            && hashes_match
            && budgets_passed
            && memory_stable
            && transfers_stable,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = config.output {
        write_json_atomically(&output, &json)?;
        eprintln!("wrote E2E comparison report {}", output.display());
    } else {
        println!("{json}");
    }
    if !report.passed {
        return Err("comparison failed deterministic or bounded-smoke budget checks".into());
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut receipts = Vec::new();
    let mut output = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => return Err(usage().into()),
            "--output" => output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
            _ if argument.starts_with('-') => {
                return Err(format!("unknown option '{argument}'\n{}", usage()).into())
            }
            _ => receipts.push(PathBuf::from(argument)),
        }
    }
    if receipts.len() < 2 {
        return Err("at least two receipt paths are required".into());
    }
    if receipts.len() > MAX_RECEIPTS {
        return Err(format!("at most {MAX_RECEIPTS} receipt paths are supported").into());
    }
    if receipts.iter().any(|path| !path.is_absolute()) {
        return Err("receipt paths must be absolute".into());
    }
    if output.as_ref().is_some_and(|path| !path.is_absolute()) {
        return Err("--output must be an absolute path".into());
    }
    Ok(Config { receipts, output })
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn read_receipt(path: &Path) -> Result<RunReceipt, Box<dyn Error>> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read receipt '{}': {error}", path.display()))?;
    let receipt = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse receipt '{}': {error}", path.display()))?;
    Ok(receipt)
}

fn validate_compatible(receipts: &[(PathBuf, RunReceipt)]) -> Result<(), Box<dyn Error>> {
    for (path, receipt) in receipts {
        if receipt.schema != RECEIPT_SCHEMA || receipt.version != RECEIPT_VERSION {
            return Err(format!(
                "receipt '{}' must use {}/{}",
                path.display(),
                RECEIPT_SCHEMA,
                RECEIPT_VERSION
            )
            .into());
        }
        if receipt.performance.schema != PERFORMANCE_SCHEMA
            || receipt.performance.version != PERFORMANCE_VERSION
        {
            return Err(format!(
                "receipt '{}' has an unsupported performance section",
                path.display()
            )
            .into());
        }
        if receipt.run_mode != "fresh-source-ingest" {
            return Err(format!(
                "receipt '{}' is '{}'; comparison requires fresh-source-ingest",
                path.display(),
                receipt.run_mode
            )
            .into());
        }
        if receipt.performance.memory.configured_source_budget_bytes == 0
            || receipt.performance.memory.configured_episode_budget_bytes == 0
        {
            return Err(format!("receipt '{}' has zero memory budget", path.display()).into());
        }
        if receipt.performance.transfers.hidden_device_copies != 0 {
            return Err(format!("receipt '{}' reports hidden device copies", path.display()).into());
        }
    }
    Ok(())
}

fn identity(receipt: &RunReceipt) -> DeterministicIdentity {
    DeterministicIdentity {
        input: receipt.input.clone(),
        front_topic: receipt.front_topic.clone(),
        rear_topic: receipt.rear_topic.clone(),
        ingest: receipt.ingest.clone(),
        sync: receipt.sync.clone(),
        odometry: receipt.odometry.clone(),
        tsdf: receipt.tsdf.clone(),
        semantic: receipt.semantic.clone(),
        viewer: receipt.viewer.clone(),
        interchange: receipt.interchange.clone(),
    }
}

fn deterministic_hashes(receipt: &RunReceipt) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let manifest_path = Path::new(&receipt.output_dir).join("rosbag2.e2e.manifest.json");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("cannot read manifest '{}': {error}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse manifest '{}': {error}", manifest_path.display()))?;
    let mut entries = BTreeMap::new();
    for entry in manifest.entries {
        let name = Path::new(&entry.path)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("manifest path '{}' has no UTF-8 filename", entry.path))?;
        if DETERMINISTIC_HASH_FILES.contains(&name) {
            entries.insert(name.to_owned(), entry.sha256);
        }
    }
    for name in DETERMINISTIC_HASH_FILES {
        if !entries.contains_key(name) {
            return Err(format!(
                "manifest '{}' is missing deterministic entry '{name}'",
                manifest_path.display()
            )
            .into());
        }
    }
    Ok(entries)
}

fn memory_comparison(receipts: &[(PathBuf, RunReceipt)]) -> MemoryComparison {
    let first = &receipts[0].1.performance.memory;
    let stable = receipts.iter().all(|(_, receipt)| {
        let memory = &receipt.performance.memory;
        memory.configured_source_budget_bytes == first.configured_source_budget_bytes
            && memory.configured_episode_budget_bytes == first.configured_episode_budget_bytes
            && memory.retained_episode_bytes == first.retained_episode_bytes
            && memory.peak_source_bytes == first.peak_source_bytes
    });
    MemoryComparison {
        configured_source_budget_bytes: first.configured_source_budget_bytes,
        configured_episode_budget_bytes: first.configured_episode_budget_bytes,
        retained_episode_bytes: first.retained_episode_bytes,
        peak_source_bytes: first.peak_source_bytes,
        stable,
    }
}

fn transfer_comparison(receipts: &[(PathBuf, RunReceipt)]) -> TransferComparison {
    let first = &receipts[0].1.performance.transfers;
    let stable = receipts.iter().all(|(_, receipt)| {
        let transfer = &receipt.performance.transfers;
        transfer.host_to_device_bytes == first.host_to_device_bytes
            && transfer.device_to_host_bytes == first.device_to_host_bytes
            && transfer.hidden_device_copies == first.hidden_device_copies
    });
    TransferComparison {
        host_to_device_bytes: first.host_to_device_bytes,
        device_to_host_bytes: first.device_to_host_bytes,
        hidden_device_copies: first.hidden_device_copies,
        stable,
    }
}

fn stage_stats(name: &'static str, mut samples: Vec<u64>, budget_ns: u64) -> StageStats {
    samples.sort_unstable();
    let min_ns = samples[0];
    let max_ns = *samples.last().expect("non-empty samples");
    let median_ns = median(&samples);
    let p95_ns = percentile_95(&samples);
    let mean_ns = samples.iter().map(|sample| *sample as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = *sample as f64 - mean_ns;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    let stddev_ns = variance.sqrt();
    let coefficient_of_variation = if mean_ns == 0.0 { 0.0 } else { stddev_ns / mean_ns };
    StageStats {
        stage: name,
        samples_ns: samples,
        min_ns,
        median_ns,
        p95_ns,
        max_ns,
        mean_ns,
        stddev_ns,
        coefficient_of_variation,
        budget_ns,
        within_budget: max_ns <= budget_ns,
    }
}

fn median(samples: &[u64]) -> u64 {
    let middle = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[middle]
    } else {
        ((samples[middle - 1] as u128 + samples[middle] as u128) / 2) as u64
    }
}

fn percentile_95(samples: &[u64]) -> u64 {
    let rank = (samples.len() * 95).div_ceil(100).max(1);
    samples[rank - 1]
}

fn write_json_atomically(path: &Path, json: &str) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("comparison output '{}' already exists", path.display()).into());
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("comparison.json")
    ));
    fs::write(&temporary, format!("{json}\n"))?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn usage() -> String {
    "usage: rosbag2_e2e_compare RECEIPT_JSON... [--output ABSOLUTE_REPORT_JSON]".into()
}

#[cfg(test)]
mod tests {
    use super::{median, parse_args, percentile_95, stage_stats};

    #[test]
    fn parses_absolute_receipts_and_output() {
        let config = parse_args(
            [
                "/media/run-1/receipt.json",
                "/media/run-2/receipt.json",
                "--output",
                "/media/compare/report.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.receipts.len(), 2);
        assert_eq!(config.output.unwrap().to_str(), Some("/media/compare/report.json"));
    }

    #[test]
    fn rejects_relative_or_insufficient_inputs() {
        assert!(parse_args(["receipt.json"].into_iter().map(str::to_owned)).is_err());
        assert!(parse_args(
            ["receipt.json", "/media/run-2/receipt.json"].into_iter().map(str::to_owned),
        )
        .is_err());
    }

    #[test]
    fn computes_median_p95_and_budget_stats() {
        assert_eq!(median(&[10, 20, 30]), 20);
        assert_eq!(median(&[10, 20, 30, 40]), 25);
        assert_eq!(percentile_95(&[10, 20, 30]), 30);
        let stats = stage_stats("test", vec![10, 20, 30], 30);
        assert_eq!(stats.min_ns, 10);
        assert_eq!(stats.max_ns, 30);
        assert!(stats.within_budget);
        assert!(stats.coefficient_of_variation > 0.0);
    }
}

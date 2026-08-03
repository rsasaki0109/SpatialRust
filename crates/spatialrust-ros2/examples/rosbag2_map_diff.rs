//! Compare two receipt-backed TSDF/glTF maps and emit a source-bound dashboard.
//!
//! The comparison reads only existing external run artifacts. It requires the
//! same canonical input identity and coordinate frame before admitting a
//! geometry diff, then reports stable-index displacement metrics and a bounded
//! spatial heatmap. Clock/TF calibration remains an explicit mapping blocker.

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
use spatialrust_scene::TriangleMesh;
use spatialrust_viewer::{
    MapDiffBounds, MapDiffCell, MapDiffMap, MapDiffState, MapDiffSummary, ReplayArtifact,
    StudioSource,
};

const RECEIPT_SCHEMA: &str = "spatialrust.rosbag2.e2e.receipt";
const DEFAULT_GRID_SIZE: u32 = 16;
const MAX_GRID_SIZE: u32 = 64;
const DEFAULT_CHANGE_THRESHOLD_UM: u64 = 1_000;
const STATE_FILE: &str = "map-diff.json";
const HTML_FILE: &str = "map-diff.html";
const MANIFEST_FILE: &str = "map-diff.manifest.json";

#[derive(Debug)]
struct Config {
    base_run_dir: PathBuf,
    candidate_run_dir: PathBuf,
    output_dir: PathBuf,
    expected_sha256: String,
    grid_size: u32,
    change_threshold_um: u64,
    min_output_free_bytes: u64,
}

struct LoadedMap {
    map: MapDiffMap,
    mesh: TriangleMesh,
    input_receipt: FileReceipt,
    map_receipt: FileReceipt,
    receipt_path: PathBuf,
    manifest_path: PathBuf,
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

#[derive(Default)]
struct CellAccumulator {
    base_vertex_count: u64,
    candidate_vertex_count: u64,
    compared_vertex_count: u64,
    changed_vertex_count: u64,
    max_displacement_um: u64,
    displacement_sum_um: u128,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rosbag2-map-diff: {error}");
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
            "Map Diff output directory '{}' already exists; choose a new run directory",
            config.output_dir.display()
        )
        .into());
    }

    let base = load_map(&config.base_run_dir, &config.expected_sha256, "base")?;
    let candidate = load_map(&config.candidate_run_dir, &config.expected_sha256, "candidate")?;
    let source_identity_match = same_source(&base.map.source, &candidate.map.source);
    let frame_identity_match = base.map.frame_id == candidate.map.frame_id;
    let comparable = source_identity_match && frame_identity_match;
    let (summary, cells) = if comparable {
        compute_diff(&base, &candidate, &config)?
    } else {
        (
            blocked_summary(
                &base,
                &candidate,
                config.change_threshold_um,
                source_identity_match,
                frame_identity_match,
            )?,
            Vec::new(),
        )
    };

    let mut blockers = Vec::new();
    if !source_identity_match {
        push_blocker(&mut blockers, "base and candidate input SHA-256 identities do not match");
    }
    if !frame_identity_match {
        push_blocker(&mut blockers, "base and candidate maps use different coordinate frames");
    }
    if !comparable {
        push_blocker(
            &mut blockers,
            "geometry comparison withheld until source and frame identity match",
        );
    } else {
        if base.time_basis != candidate.time_basis {
            push_blocker(&mut blockers, "base and candidate time bases do not match");
        }
        push_blocker(&mut blockers, format!("time basis: {}", base.time_basis));
        push_blocker(
            &mut blockers,
            "clock calibration not applied; diff remains in the header-stamp domain",
        );
        push_blocker(&mut blockers, "TF/frame composition not applied; diff is inspection-only");
        push_blocker(
            &mut blockers,
            "mapping admission requires source-bound calibrated map evidence",
        );
    }

    let state = MapDiffState::try_new(
        format!("Map Diff — {} vs {}", base.map.label, candidate.map.label),
        base.map.clone(),
        candidate.map.clone(),
        summary,
        cells,
        vec![base.map.artifact.clone(), candidate.map.artifact.clone()],
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
    manifest.entries.push(base.input_receipt.clone());
    manifest.entries.push(base.map_receipt.clone());
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, &base.receipt_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, &base.manifest_path)?);
    manifest.entries.push(candidate.map_receipt.clone());
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Auxiliary, &candidate.receipt_path)?);
    manifest
        .entries
        .push(FileReceipt::from_path(ReceiptRole::Auxiliary, &candidate.manifest_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &state_path)?);
    manifest.entries.push(FileReceipt::from_path(ReceiptRole::Output, &html_path)?);
    let validation = manifest.validate_local_files()?;
    manifest.write_json(&manifest_path)?;

    println!("{}", serde_json::to_string_pretty(&state)?);
    println!(
        "Map Diff: {} (compare_ready={}, mapping_admitted={})",
        state_path.display(),
        state.compare_ready,
        state.mapping_admitted
    );
    println!("Map Diff dashboard: {}", html_path.display());
    println!(
        "Map Diff manifest: {} (checked_files={}, total_bytes={}, free_before={})",
        manifest_path.display(),
        validation.checked_local_files,
        validation.total_bytes,
        preflight.available_bytes
    );
    if !state.compare_ready {
        return Err("Map Diff failed its source/frame admission checks".into());
    }
    Ok(())
}

fn load_map(
    run_dir: &Path,
    expected_sha256: &str,
    label: &str,
) -> Result<LoadedMap, Box<dyn Error>> {
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
        format!("{label} canonical rosbag2 input"),
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
    let map_artifact = ReplayArtifact::try_new(
        format!("{label}-map"),
        map_path.display().to_string(),
        map_receipt.size_bytes.ok_or("Map manifest entry has no size")?,
        map_receipt.sha256.clone().ok_or("Map manifest entry has no SHA-256")?,
    )?;
    let map = MapDiffMap::try_new(
        label,
        source,
        map_artifact,
        receipt.tsdf.frame_id,
        vertex_count,
        triangle_count,
        mesh_bounds(&mesh)?,
    )?;
    Ok(LoadedMap {
        map,
        mesh,
        input_receipt,
        map_receipt,
        receipt_path,
        manifest_path,
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

fn same_source(base: &StudioSource, candidate: &StudioSource) -> bool {
    base.identity_matches
        && candidate.identity_matches
        && base.expected_sha256 == candidate.expected_sha256
        && base.observed_sha256 == candidate.observed_sha256
}

fn blocked_summary(
    base: &LoadedMap,
    candidate: &LoadedMap,
    threshold_um: u64,
    source_identity_match: bool,
    frame_identity_match: bool,
) -> Result<MapDiffSummary, Box<dyn Error>> {
    let base_vertices = base.map.vertex_count;
    let candidate_vertices = candidate.map.vertex_count;
    MapDiffSummary::try_new(
        base_vertices,
        candidate_vertices,
        0,
        candidate_vertices.saturating_sub(base_vertices),
        base_vertices.saturating_sub(candidate_vertices),
        0,
        threshold_um,
        0,
        0,
        0,
        0,
        false,
        false,
        source_identity_match,
        frame_identity_match,
        false,
    )
    .map_err(Into::into)
}

fn compute_diff(
    base: &LoadedMap,
    candidate: &LoadedMap,
    config: &Config,
) -> Result<(MapDiffSummary, Vec<MapDiffCell>), Box<dyn Error>> {
    let grid_size = usize::try_from(config.grid_size)?;
    let cell_count = grid_size.checked_mul(grid_size).ok_or("Map Diff grid size overflow")?;
    let combined_bounds = combined_bounds(&base.mesh, &candidate.mesh)?;
    let mut cells = (0..cell_count).map(|_| CellAccumulator::default()).collect::<Vec<_>>();
    let base_positions = base.mesh.positions.chunks_exact(3).collect::<Vec<_>>();
    let candidate_positions = candidate.mesh.positions.chunks_exact(3).collect::<Vec<_>>();
    let mut displacements = Vec::with_capacity(base_positions.len().min(candidate_positions.len()));

    for index in 0..base_positions.len().max(candidate_positions.len()) {
        let base_position = base_positions.get(index).copied();
        let candidate_position = candidate_positions.get(index).copied();
        if let Some(position) = base_position {
            let index = cell_index(position, &combined_bounds, grid_size)?;
            cells[index].base_vertex_count = cells[index]
                .base_vertex_count
                .checked_add(1)
                .ok_or("Map Diff base cell count overflow")?;
        }
        if let Some(position) = candidate_position {
            let index = cell_index(position, &combined_bounds, grid_size)?;
            cells[index].candidate_vertex_count = cells[index]
                .candidate_vertex_count
                .checked_add(1)
                .ok_or("Map Diff candidate cell count overflow")?;
        }
        if let (Some(base_position), Some(candidate_position)) = (base_position, candidate_position)
        {
            let index = cell_index(base_position, &combined_bounds, grid_size)?;
            let displacement_um = displacement_um(base_position, candidate_position)?;
            let cell = &mut cells[index];
            cell.compared_vertex_count = cell
                .compared_vertex_count
                .checked_add(1)
                .ok_or("Map Diff compared cell count overflow")?;
            cell.displacement_sum_um = cell
                .displacement_sum_um
                .checked_add(u128::from(displacement_um))
                .ok_or("Map Diff displacement sum overflow")?;
            cell.max_displacement_um = cell.max_displacement_um.max(displacement_um);
            if displacement_um > config.change_threshold_um {
                cell.changed_vertex_count = cell
                    .changed_vertex_count
                    .checked_add(1)
                    .ok_or("Map Diff changed cell count overflow")?;
            }
            displacements.push(displacement_um);
        }
    }

    displacements.sort_unstable();
    let compared_vertex_count = u64::try_from(displacements.len())?;
    let displacement_sum_um = displacements.iter().try_fold(0_u128, |sum, value| {
        sum.checked_add(u128::from(*value)).ok_or("Map Diff displacement sum overflow")
    })?;
    let mean_displacement_um = if compared_vertex_count == 0 {
        0
    } else {
        u64::try_from(displacement_sum_um / u128::from(compared_vertex_count))?
    };
    let p95_displacement_um = if displacements.is_empty() {
        0
    } else {
        let rank = displacements
            .len()
            .checked_mul(95)
            .ok_or("Map Diff percentile overflow")?
            .div_ceil(100);
        displacements[rank.saturating_sub(1)]
    };
    let changed_vertex_count = cells.iter().try_fold(0_u64, |sum, cell| {
        sum.checked_add(cell.changed_vertex_count).ok_or("Map Diff changed count overflow")
    })?;
    let map_cells = cells
        .into_iter()
        .enumerate()
        .map(|(index, cell)| {
            let mean = if cell.compared_vertex_count == 0 {
                0
            } else {
                u64::try_from(cell.displacement_sum_um / u128::from(cell.compared_vertex_count))?
            };
            MapDiffCell::try_new(
                u32::try_from(index)?,
                cell.base_vertex_count,
                cell.candidate_vertex_count,
                cell.compared_vertex_count,
                cell.changed_vertex_count,
                cell.max_displacement_um,
                mean,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let base_vertices = u64::try_from(base_positions.len())?;
    let candidate_vertices = u64::try_from(candidate_positions.len())?;
    let summary = MapDiffSummary::try_new(
        base_vertices,
        candidate_vertices,
        compared_vertex_count,
        candidate_vertices.saturating_sub(base_vertices),
        base_vertices.saturating_sub(candidate_vertices),
        changed_vertex_count,
        config.change_threshold_um,
        displacements.last().copied().unwrap_or(0),
        mean_displacement_um,
        p95_displacement_um,
        u64::try_from(map_cells.len())?,
        base.map.artifact.sha256 == candidate.map.artifact.sha256,
        base.mesh.indices == candidate.mesh.indices,
        same_source(&base.map.source, &candidate.map.source),
        base.map.frame_id == candidate.map.frame_id,
        false,
    )?;
    Ok((summary, map_cells))
}

fn mesh_bounds(mesh: &TriangleMesh) -> Result<MapDiffBounds, Box<dyn Error>> {
    bounds_from_positions(mesh.positions.chunks_exact(3))
}

fn combined_bounds(
    base: &TriangleMesh,
    candidate: &TriangleMesh,
) -> Result<MapDiffBounds, Box<dyn Error>> {
    let mut positions = base.positions.chunks_exact(3).chain(candidate.positions.chunks_exact(3));
    let first = positions.next().ok_or("Map Diff meshes contain no positions")?;
    let mut min = quantized_position(first)?;
    let mut max = min;
    for position in positions {
        let point = quantized_position(position)?;
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    MapDiffBounds::try_new(min[0], min[1], min[2], max[0], max[1], max[2]).map_err(Into::into)
}

fn bounds_from_positions<'a>(
    positions: impl Iterator<Item = &'a [f32]>,
) -> Result<MapDiffBounds, Box<dyn Error>> {
    let mut positions = positions;
    let first = positions.next().ok_or("Map Diff mesh contains no positions")?;
    let mut min = quantized_position(first)?;
    let mut max = min;
    for position in positions {
        let point = quantized_position(position)?;
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    MapDiffBounds::try_new(min[0], min[1], min[2], max[0], max[1], max[2]).map_err(Into::into)
}

fn quantized_position(position: &[f32]) -> Result<[i64; 3], Box<dyn Error>> {
    if position.len() != 3 || position.iter().any(|value| !value.is_finite()) {
        return Err("Map Diff mesh contains a non-finite position".into());
    }
    let mut output = [0_i64; 3];
    for (axis, value) in position.iter().enumerate() {
        let scaled = f64::from(*value) * 1_000_000.0;
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err("Map Diff position exceeds integer micrometre range".into());
        }
        output[axis] = scaled.round() as i64;
    }
    Ok(output)
}

fn cell_index(
    position: &[f32],
    bounds: &MapDiffBounds,
    grid_size: usize,
) -> Result<usize, Box<dyn Error>> {
    let point = quantized_position(position)?;
    let x = axis_cell(point[0], bounds.min_x_um, bounds.max_x_um, grid_size);
    let y = axis_cell(point[1], bounds.min_y_um, bounds.max_y_um, grid_size);
    Ok(y * grid_size + x)
}

fn axis_cell(value: i64, minimum: i64, maximum: i64, grid_size: usize) -> usize {
    if maximum == minimum {
        return 0;
    }
    let numerator = value.saturating_sub(minimum) as f64;
    let denominator = maximum.saturating_sub(minimum) as f64;
    ((numerator / denominator).clamp(0.0, 1.0) * grid_size as f64)
        .floor()
        .min((grid_size - 1) as f64) as usize
}

fn displacement_um(base: &[f32], candidate: &[f32]) -> Result<u64, Box<dyn Error>> {
    let dx = f64::from(base[0]) - f64::from(candidate[0]);
    let dy = f64::from(base[1]) - f64::from(candidate[1]);
    let dz = f64::from(base[2]) - f64::from(candidate[2]);
    let displacement = (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt() * 1_000_000.0;
    if !displacement.is_finite() || displacement > u64::MAX as f64 {
        return Err("Map Diff displacement is outside the supported range".into());
    }
    Ok(displacement.round() as u64)
}

fn render_dashboard(state: &MapDiffState) -> Result<String, Box<dyn Error>> {
    let state_json = serde_json::to_string(state)?.replace("</", "<\\/");
    let title = escape_html(&state.title);
    let template = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#080d19;--panel:#111b2d;--line:#284463;--muted:#8ea8c0;--cyan:#63e6ff;--green:#64f2a3;--red:#ff7180;--amber:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#233c67 0,#080d19 48%);color:#eef7ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.title{font-size:30px;font-weight:760;margin-top:5px}.sub{color:var(--muted);font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{border:1px solid var(--line);border-radius:999px;padding:9px 14px;font-weight:760;white-space:nowrap}.ok{color:var(--green);border-color:#237850}.blocked{color:var(--red);border-color:#873442}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:linear-gradient(145deg,rgba(18,38,65,.96),rgba(9,17,31,.96));border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 14px 32px #0004}.panel h2{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 11px}.metric{font-size:25px;font-weight:760;color:var(--cyan)}.small{font-size:12px;color:var(--muted)}.wide{grid-column:span 2}.full{grid-column:1/-1}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #1c3552;padding:9px 0}.row:last-child{border-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.danger{color:var(--red)}.warning{color:var(--amber)}.blockers{margin:0;padding-left:20px}.blockers li{margin:6px 0;color:#ff9aa4}.maps{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mapcard{border:1px solid #254363;border-radius:10px;padding:11px;min-width:0}.maplabel{color:var(--cyan);font-weight:700;margin-bottom:7px}.heatmap{display:grid;grid-template-columns:repeat(16,1fr);gap:3px;aspect-ratio:1/1;max-width:600px}.cell{border-radius:3px;min-width:0;min-height:14px;border:1px solid #173452}.legend{display:flex;align-items:center;gap:8px;margin-top:12px}.legendbar{height:10px;flex:1;max-width:320px;border-radius:8px;background:linear-gradient(90deg,#132c4a,#23b6c8,#f9d65c,#ff5d73)}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:580px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}.top{display:block}.badge{display:inline-block;margin-top:15px}.maps{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<section class="top"><div><div class="eyebrow">SpatialRust / source-bound map diff</div><div class="title" id="title">__TITLE__</div><div class="sub" id="source"></div></div><div id="admission" class="badge"></div></section>
<section class="grid">
<article class="panel"><h2>Comparison</h2><div id="ready" class="metric"></div><div id="readyDetail" class="small"></div></article>
<article class="panel"><h2>Mapping gate</h2><div id="mapping" class="metric"></div><div id="mappingDetail" class="small"></div></article>
<article class="panel"><h2>Changed vertices</h2><div id="changed" class="metric"></div><div id="changedDetail" class="small"></div></article>
<article class="panel"><h2>Max displacement</h2><div id="max" class="metric"></div><div id="p95" class="small"></div></article>
<article class="panel wide"><h2>Base vs candidate</h2><div id="maps" class="maps"></div></article>
<article class="panel wide"><h2>Spatial change heatmap</h2><div id="heatmap" class="heatmap"></div><div class="legend"><span class="small">low</span><div class="legendbar"></div><span class="small">high</span></div><div id="heatmapDetail" class="small" style="margin-top:9px"></div></article>
<article class="panel wide"><h2>Geometry and topology</h2><div id="geometry"></div></article>
<article class="panel wide"><h2>Admission blockers</h2><ul id="blockers" class="blockers"></ul></article>
<article class="panel full"><h2>Portable JSON state</h2><pre id="raw" class="mono" style="max-height:280px;overflow:auto;white-space:pre-wrap"></pre></article>
</section></main>
<script id="map-diff-state" type="application/json">__STATE_JSON__</script>
<script>
const state=JSON.parse(document.getElementById('map-diff-state').textContent),q=id=>document.getElementById(id),fmtUm=n=>n>=1000000?(n/1000000).toFixed(3)+' m':n>=1000?(n/1000).toFixed(2)+' mm':n+' µm',fmtCount=n=>n.toLocaleString(),esc=v=>String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
q('title').textContent=state.title;q('source').textContent=state.base.source.path+' · '+state.base.source.observed_sha256;q('admission').textContent=state.compare_ready?'DIFF READY':'DIFF BLOCKED';q('admission').className='badge '+(state.compare_ready?'ok':'blocked');
q('ready').textContent=state.compare_ready?'READY':'BLOCKED';q('ready').className='metric '+(state.compare_ready?'':'danger');q('readyDetail').textContent=state.summary.source_identity_match?'source identity matched':'source identity mismatch';
q('mapping').textContent=state.mapping_admitted?'ADMITTED':'BLOCKED';q('mapping').className='metric '+(state.mapping_admitted?'':'danger');q('mappingDetail').textContent=state.summary.calibration_applied?'calibration applied':'inspection-only · calibration absent';
const s=state.summary,changedRatio=s.compared_vertex_count?100*s.changed_vertex_count/s.compared_vertex_count:0;q('changed').textContent=fmtCount(s.changed_vertex_count);q('changedDetail').textContent=changedRatio.toFixed(3)+'% of '+fmtCount(s.compared_vertex_count)+' compared';q('max').textContent=fmtUm(s.max_displacement_um);q('p95').textContent='p95 '+fmtUm(s.p95_displacement_um)+' · mean '+fmtUm(s.mean_displacement_um);
q('maps').innerHTML=[state.base,state.candidate].map(m=>'<div class="mapcard"><div class="maplabel">'+esc(m.label)+'</div><div class="row"><span>frame</span><span class="mono">'+esc(m.frame_id)+'</span></div><div class="row"><span>vertices</span><span class="mono">'+fmtCount(m.vertex_count)+'</span></div><div class="row"><span>triangles</span><span class="mono">'+fmtCount(m.triangle_count)+'</span></div><div class="small mono">'+esc(m.artifact.sha256)+'</div></div>').join('');
const grid=Math.max(1,Math.round(Math.sqrt(state.cells.length)));q('heatmap').style.gridTemplateColumns='repeat('+grid+',1fr)';const peak=Math.max(1,...state.cells.map(c=>c.changed_vertex_count?c.max_displacement_um:0));q('heatmap').innerHTML=state.cells.map(c=>{const ratio=c.compared_vertex_count?c.changed_vertex_count/c.compared_vertex_count:0;const intensity=Math.min(1,Math.max(ratio,c.max_displacement_um/peak));const color='hsl('+(190-170*intensity)+' '+(55+35*intensity)+'% '+(18+48*intensity)+'%)';return '<div class="cell" title="#'+c.index+' · '+fmtCount(c.changed_vertex_count)+' changed · '+fmtUm(c.max_displacement_um)+'" style="background:'+color+'"></div>';}).join('')||'<div class="small">No admitted comparison cells</div>';q('heatmapDetail').textContent=state.cells.length+' cells · threshold '+fmtUm(s.change_threshold_um)+' · changed vertices are compared by stable vertex index';
q('geometry').innerHTML='<div class="row"><span>artifact hash</span><span class="mono">'+(s.geometry_hash_equal?'IDENTICAL':'DIFFERENT')+'</span></div><div class="row"><span>topology</span><span class="mono">'+(s.topology_equal?'IDENTICAL':'DIFFERENT')+'</span></div><div class="row"><span>source identity</span><span class="mono">'+(s.source_identity_match?'MATCH':'MISMATCH')+'</span></div><div class="row"><span>frame identity</span><span class="mono">'+(s.frame_identity_match?'MATCH':'MISMATCH')+'</span></div><div class="row"><span>added / removed</span><span class="mono">'+fmtCount(s.added_vertex_count)+' / '+fmtCount(s.removed_vertex_count)+'</span></div>';
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
    if !config.base_run_dir.is_absolute()
        || !config.candidate_run_dir.is_absolute()
        || !config.output_dir.is_absolute()
    {
        return Err("run directories and --output-dir paths must be absolute".into());
    }
    if !config.base_run_dir.is_dir() || !config.candidate_run_dir.is_dir() {
        return Err("base and candidate run directories must exist".into());
    }
    if config.output_dir == Path::new("/") {
        return Err("--output-dir must not be the filesystem root".into());
    }
    let parent =
        config.output_dir.parent().ok_or("--output-dir must have an existing parent directory")?;
    if !parent.is_dir() {
        return Err(format!("output parent '{}' is not a directory", parent.display()).into());
    }
    if config.base_run_dir == config.candidate_run_dir {
        return Err("base and candidate run directories must differ".into());
    }
    if config.grid_size == 0 || config.grid_size > MAX_GRID_SIZE {
        return Err(format!("--grid-size must be between 1 and {MAX_GRID_SIZE}").into());
    }
    if config.change_threshold_um == 0 || config.min_output_free_bytes == 0 {
        return Err("numeric Map Diff limits must be greater than zero".into());
    }
    validate_sha256(&config.expected_sha256)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let base_run_dir = args.next().ok_or_else(usage)?;
    let candidate_run_dir = args.next().ok_or_else(usage)?;
    let mut output_dir = None;
    let mut expected_sha256 = None;
    let mut grid_size = DEFAULT_GRID_SIZE;
    let mut change_threshold_um = DEFAULT_CHANGE_THRESHOLD_UM;
    let mut min_output_free_bytes = DEFAULT_MIN_OUTPUT_FREE_BYTES;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--expected-input-sha256" => expected_sha256 = Some(next_value(&mut args, &flag)?),
            "--grid-size" => grid_size = parse_u64(&mut args, &flag)?.try_into()?,
            "--change-threshold-um" => change_threshold_um = parse_u64(&mut args, &flag)?,
            "--min-output-free-bytes" => min_output_free_bytes = parse_u64(&mut args, &flag)?,
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(Config {
        base_run_dir: PathBuf::from(base_run_dir),
        candidate_run_dir: PathBuf::from(candidate_run_dir),
        output_dir: output_dir.ok_or("--output-dir is required")?,
        expected_sha256: expected_sha256.ok_or("--expected-input-sha256 is required")?,
        grid_size,
        change_threshold_um,
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

fn usage() -> String {
    "usage: rosbag2_map_diff BASE_RUN_DIR CANDIDATE_RUN_DIR \
     --output-dir ABSOLUTE_OUTPUT_DIR --expected-input-sha256 SHA256 \
     [--grid-size N] [--change-threshold-um N] [--min-output-free-bytes BYTES]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_args, validate_sha256};

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_map_diff_options() {
        let config = parse_args(
            [
                "/media/base",
                "/media/candidate",
                "--output-dir",
                "/media/results/map-diff",
                "--expected-input-sha256",
                SHA,
                "--grid-size",
                "8",
                "--change-threshold-um",
                "2500",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.grid_size, 8);
        assert_eq!(config.change_threshold_um, 2_500);
    }

    #[test]
    fn rejects_relative_runs_and_bad_hashes() {
        let config = parse_args(
            ["base", "candidate", "--output-dir", "results", "--expected-input-sha256", SHA]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert!(super::validate_config(&config).is_err());
        assert!(validate_sha256("not-a-sha256").is_err());
    }
}

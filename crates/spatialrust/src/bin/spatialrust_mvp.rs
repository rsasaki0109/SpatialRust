//! MVP pipeline CLI: point cloud in → labeled cluster output.
//!
//! ```text
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- input.pcd output.pcd
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
//!   --bounds 0,0,-1,100,100,1 scan.copc.laz roi.copc.laz
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
//!   --resolution 0.5 scan.copc.laz coarse.copc.laz
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
//!   --voxel-mode approximate --leaf-size 0.2 scan.las out.las
//! ```

use std::{env, path::Path, process::ExitCode, time::Instant};

use spatialrust::{
    detect_point_cloud_format, read_copc_file_info, read_copc_file_with_query,
    read_point_cloud_file, write_point_cloud_file, CopcBounds, CopcQuery, ExecutionPolicy,
    MvpPipeline, MvpPipelineConfig, PointCloudFileFormat, VoxelAggregationMode,
    VoxelGridDownsampleConfig,
};
#[cfg(feature = "io-copc-http")]
use spatialrust::{read_copc_url_info, read_copc_url_with_query};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CopcQueryOptions {
    bounds: Option<CopcBounds>,
    resolution: Option<f64>,
}

impl CopcQueryOptions {
    const fn is_active(self) -> bool {
        self.bounds.is_some() || self.resolution.is_some()
    }
}

fn print_usage(program: &str) {
    eprintln!(
        "\
Usage: {program} [OPTIONS] <INPUT> <OUTPUT>

Run the MVP pipeline (voxel → normals → plane → cluster) and write labeled output.

Options:
  --leaf-size <METERS>       Voxel leaf size (default: 0.05)
  --voxel-mode <MODE>        centroid | approximate (default: centroid)
  --voxel-policy <POLICY>    auto | cpu | gpu (default: auto)
  --bounds <MINX,MINY,MINZ,MAXX,MAXY,MAXZ>
                             COPC spatial query bounds (requires COPC input)
  --resolution <METERS>      COPC max point spacing LOD (requires COPC input; uses root bounds
                             when --bounds is omitted)
  --repeat <N>               Run the MVP pipeline N times (default: 1); logs per-iteration timing
  -h, --help                 Show this help
"
    );
}

fn parse_voxel_policy(value: &str) -> Result<ExecutionPolicy, String> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(ExecutionPolicy::Auto),
        "cpu" => Ok(ExecutionPolicy::CpuSingle),
        "gpu" => {
            #[cfg(not(feature = "pipeline-mvp-gpu"))]
            {
                return Err(
                    "GPU voxel policy requires `--features mvp,pipeline-mvp-gpu`".to_string(),
                );
            }
            #[cfg(feature = "pipeline-mvp-gpu")]
            {
                use spatialrust::DeviceKind;
                Ok(ExecutionPolicy::Gpu(DeviceKind::Wgpu))
            }
        }
        other => Err(format!("unknown voxel policy `{other}` (expected auto, cpu, or gpu)")),
    }
}

fn parse_voxel_mode(value: &str) -> Result<VoxelAggregationMode, String> {
    match value.to_ascii_lowercase().as_str() {
        "centroid" => Ok(VoxelAggregationMode::Centroid),
        "approximate" | "approximate-first" | "approx" => {
            Ok(VoxelAggregationMode::ApproximateFirst)
        }
        other => Err(format!(
            "unknown voxel mode `{other}` (expected centroid or approximate)"
        )),
    }
}

fn build_voxel_config(leaf_size: f32, mode: VoxelAggregationMode) -> VoxelGridDownsampleConfig {
    match mode {
        VoxelAggregationMode::Centroid => VoxelGridDownsampleConfig::centroid(leaf_size),
        VoxelAggregationMode::ApproximateFirst => {
            VoxelGridDownsampleConfig::approximate(leaf_size)
        }
    }
}

fn voxel_mode_label(mode: VoxelAggregationMode) -> &'static str {
    match mode {
        VoxelAggregationMode::Centroid => "centroid",
        VoxelAggregationMode::ApproximateFirst => "approximate",
    }
}

fn parse_bounds(value: &str) -> Result<CopcBounds, String> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 6 {
        return Err(format!(
            "expected 6 comma-separated values (minx,miny,minz,maxx,maxy,maxz), got {}",
            parts.len()
        ));
    }

    let mut coords = [0.0_f64; 6];
    for (index, part) in parts.iter().enumerate() {
        coords[index] = part
            .trim()
            .parse()
            .map_err(|_| format!("invalid bounds coordinate `{part}`"))?;
    }

    let bounds = CopcBounds::from_ranges(
        (coords[0], coords[3]),
        (coords[1], coords[4]),
        (coords[2], coords[5]),
    );
    bounds
        .validate()
        .map_err(|error| format!("invalid COPC bounds: {error}"))?;
    Ok(bounds)
}

fn parse_resolution(value: &str) -> Result<f64, String> {
    let resolution = value
        .trim()
        .parse()
        .map_err(|_| format!("invalid resolution `{value}`"))?;
    CopcQuery::with_resolution(CopcBounds::new([0.0; 3], [1.0; 3]), resolution)
        .validate()
        .map_err(|error| format!("invalid COPC resolution: {error}"))?;
    Ok(resolution)
}

fn parse_repeat(value: &str) -> Result<usize, String> {
    let repeat: usize = value
        .parse()
        .map_err(|_| format!("invalid repeat count `{value}`"))?;
    if repeat == 0 {
        return Err("--repeat requires a positive integer".to_string());
    }
    Ok(repeat)
}

fn log_repeat_summary(timings: &[std::time::Duration]) {
    let min = timings.iter().min().copied().expect("repeat timings");
    let max = timings.iter().max().copied().expect("repeat timings");
    let total: std::time::Duration = timings.iter().sum();
    let avg = total / timings.len() as u32;
    eprintln!(
        "repeat summary: min={min:.3?} max={max:.3?} avg={avg:.3?} (n={})",
        timings.len()
    );
}

fn is_http_copc_input(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

fn detect_input_format(input: &str) -> Option<PointCloudFileFormat> {
    if is_http_copc_input(input) {
        let file_name = input.rsplit('/').next()?.to_ascii_lowercase();
        if file_name.ends_with(".copc.laz") || file_name.ends_with(".copc.las") {
            return Some(PointCloudFileFormat::Copc);
        }
        return None;
    }
    detect_point_cloud_format(input)
}

fn ensure_copc_input(input_path: &str, option: &str) -> Result<(), String> {
    let format = detect_input_format(input_path)
        .ok_or_else(|| format!("cannot detect input format for {option}: {input_path}"))?;
    if format != PointCloudFileFormat::Copc {
        return Err(format!("{option} requires a COPC input (.copc.laz/.copc.las or http URL)"));
    }
    Ok(())
}

fn read_copc_header_info(
    input_path: &str,
) -> Result<spatialrust::CopcFileInfo, Box<dyn std::error::Error>> {
    if is_http_copc_input(input_path) {
        #[cfg(feature = "io-copc-http")]
        {
            return Ok(read_copc_url_info(input_path)?);
        }
        #[cfg(not(feature = "io-copc-http"))]
        {
            return Err(
                "HTTP COPC input requires `--features mvp,mvp-http` when building spatialrust-mvp"
                    .into(),
            );
        }
    }
    Ok(read_copc_file_info(input_path)?)
}

fn read_copc_points(
    input_path: &str,
    query: Option<&CopcQuery>,
) -> Result<spatialrust::PointCloud, Box<dyn std::error::Error>> {
    if is_http_copc_input(input_path) {
        #[cfg(feature = "io-copc-http")]
        {
            return Ok(read_copc_url_with_query(input_path, query)?);
        }
        #[cfg(not(feature = "io-copc-http"))]
        {
            return Err(
                "HTTP COPC input requires `--features mvp,mvp-http` when building spatialrust-mvp"
                    .into(),
            );
        }
    }
    match query {
        Some(query) => Ok(read_copc_file_with_query(input_path, query)?),
        None => Ok(read_point_cloud_file(input_path)?),
    }
}

fn build_copc_query(
    input_path: &str,
    options: CopcQueryOptions,
) -> Result<CopcQuery, Box<dyn std::error::Error>> {
    let bounds = match options.bounds {
        Some(bounds) => bounds,
        None => read_copc_header_info(input_path)?.root_bounds,
    };

    let query = match options.resolution {
        Some(resolution) => CopcQuery::with_resolution(bounds, resolution),
        None => CopcQuery::bounds(bounds),
    };
    query.validate()?;
    Ok(query)
}

fn log_copc_query(query: &CopcQuery) {
    eprintln!(
        "COPC query: x=[{}, {}] y=[{}, {}] z=[{}, {}]",
        query.bounds.min[0],
        query.bounds.max[0],
        query.bounds.min[1],
        query.bounds.max[1],
        query.bounds.min[2],
        query.bounds.max[2],
    );
    if let Some(resolution) = query.max_resolution {
        eprintln!("COPC max resolution: {resolution} m");
    }
}

fn load_input(
    input_path: &str,
    copc: CopcQueryOptions,
) -> Result<spatialrust::PointCloud, Box<dyn std::error::Error>> {
    if !copc.is_active() {
        if is_http_copc_input(input_path) {
            ensure_copc_input(input_path, "HTTP COPC input")?;
            return read_copc_points(input_path, None);
        }
        return Ok(read_point_cloud_file(input_path)?);
    }

    let option = if copc.bounds.is_some() && copc.resolution.is_some() {
        "--bounds/--resolution"
    } else if copc.bounds.is_some() {
        "--bounds"
    } else {
        "--resolution"
    };
    ensure_copc_input(input_path, option)?;

    let query = build_copc_query(input_path, copc)?;
    log_copc_query(&query);
    read_copc_points(input_path, Some(&query))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let program = env::args()
        .next()
        .unwrap_or_else(|| "spatialrust-mvp".to_string());

    let mut leaf_size = 0.05_f32;
    let mut voxel_mode = VoxelAggregationMode::Centroid;
    let mut voxel_policy = ExecutionPolicy::Auto;
    let mut copc = CopcQueryOptions::default();
    let mut repeat = 1_usize;
    let mut input_path = None;
    let mut output_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage(&program);
                return Ok(());
            }
            "--leaf-size" => {
                let value = args
                    .next()
                    .ok_or("--leaf-size requires a numeric value")?;
                leaf_size = value
                    .parse()
                    .map_err(|_| format!("invalid leaf size `{value}`"))?;
            }
            "--voxel-mode" => {
                let value = args
                    .next()
                    .ok_or("--voxel-mode requires centroid or approximate")?;
                voxel_mode = parse_voxel_mode(&value)?;
            }
            "--voxel-policy" => {
                let value = args
                    .next()
                    .ok_or("--voxel-policy requires auto, cpu, or gpu")?;
                voxel_policy = parse_voxel_policy(&value)?;
            }
            "--bounds" => {
                let value = args.next().ok_or("--bounds requires 6 comma-separated values")?;
                copc.bounds = Some(parse_bounds(&value)?);
            }
            "--resolution" => {
                let value = args.next().ok_or("--resolution requires a numeric value")?;
                copc.resolution = Some(parse_resolution(&value)?);
            }
            "--repeat" => {
                let value = args.next().ok_or("--repeat requires a positive integer")?;
                repeat = parse_repeat(&value)?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option `{value}`").into());
            }
            value => {
                if input_path.is_none() {
                    input_path = Some(value.to_string());
                } else if output_path.is_none() {
                    output_path = Some(value.to_string());
                } else {
                    return Err("unexpected extra argument".into());
                }
            }
        }
    }

    let input_path = input_path.ok_or("missing INPUT path")?;
    let output_path = output_path.ok_or("missing OUTPUT path")?;

    if !is_http_copc_input(&input_path) && !Path::new(&input_path).exists() {
        return Err(format!("input file not found: {input_path}").into());
    }

    eprintln!("loading {input_path}");
    let input = load_input(&input_path, copc)?;
    let input_points = input.len();
    eprintln!("input points: {input_points}");

    let config = MvpPipelineConfig {
        voxel: build_voxel_config(leaf_size, voxel_mode),
        voxel_policy,
        ..MvpPipelineConfig::default()
    };

    eprintln!(
        "running MVP pipeline (leaf_size={leaf_size}, voxel_mode={}, voxel_policy={voxel_policy:?}, repeat={repeat})",
        voxel_mode_label(voxel_mode),
    );
    let pipeline = MvpPipeline::new(config);
    let mut timings = Vec::with_capacity(repeat);
    let mut result = None;
    for index in 0..repeat {
        let started = Instant::now();
        let run_result = pipeline.run(&input)?;
        let elapsed = started.elapsed();
        timings.push(elapsed);
        if repeat > 1 {
            eprintln!("repeat {}/{} elapsed: {elapsed:.3?}", index + 1, repeat);
        }
        result = Some(run_result);
    }
    if repeat > 1 {
        log_repeat_summary(&timings);
    }
    let result = result.expect("pipeline produced no result");
    let elapsed = *timings.last().expect("repeat timings");

    write_point_cloud_file(&output_path, &result.output)?;

    eprintln!("output points: {}", result.output.len());
    eprintln!("plane inliers: {}", result.plane.inlier_count);
    eprintln!("clusters: {}", result.clusters.cluster_count);
    eprintln!("elapsed: {:.3?}", elapsed);
    eprintln!("wrote {output_path}");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            print_usage(
                &env::args()
                    .next()
                    .unwrap_or_else(|| "spatialrust-mvp".to_string()),
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_voxel_config, detect_input_format, parse_bounds, parse_repeat, parse_resolution,
        parse_voxel_mode, CopcQueryOptions,
    };
    use spatialrust::{CopcQuery, PointCloudFileFormat, VoxelAggregationMode};

    #[test]
    fn parse_bounds_accepts_six_values() {
        let bounds = parse_bounds("0,0,-1,100,100,1").unwrap();
        assert_eq!(bounds.min, [0.0, 0.0, -1.0]);
        assert_eq!(bounds.max, [100.0, 100.0, 1.0]);
    }

    #[test]
    fn parse_bounds_rejects_wrong_count() {
        assert!(parse_bounds("0,0,0").is_err());
    }

    #[test]
    fn parse_bounds_rejects_inverted_axis() {
        assert!(parse_bounds("10,0,0,0,1,1").is_err());
    }

    #[test]
    fn parse_resolution_accepts_positive_value() {
        assert_eq!(parse_resolution("0.5").unwrap(), 0.5);
    }

    #[test]
    fn parse_resolution_rejects_non_positive() {
        assert!(parse_resolution("0").is_err());
        assert!(parse_resolution("-1").is_err());
    }

    #[test]
    fn parse_repeat_accepts_positive_integers() {
        assert_eq!(parse_repeat("1").unwrap(), 1);
        assert_eq!(parse_repeat("3").unwrap(), 3);
    }

    #[test]
    fn parse_repeat_rejects_zero_and_invalid() {
        assert!(parse_repeat("0").is_err());
        assert!(parse_repeat("-1").is_err());
        assert!(parse_repeat("abc").is_err());
    }

    #[test]
    fn copc_query_options_active_when_bounds_or_resolution_set() {
        let bounds = parse_bounds("0,0,0,1,1,1").unwrap();
        assert!(!CopcQueryOptions::default().is_active());
        assert!(CopcQueryOptions {
            bounds: Some(bounds),
            resolution: None,
        }
        .is_active());
        assert!(CopcQueryOptions {
            bounds: None,
            resolution: Some(0.5),
        }
        .is_active());
    }

    #[test]
    fn resolution_query_sets_max_resolution_field() {
        let bounds = parse_bounds("0,0,0,10,10,10").unwrap();
        let query = CopcQuery::with_resolution(bounds, 0.25);
        assert_eq!(query.max_resolution, Some(0.25));
    }

    #[test]
    fn parse_voxel_mode_accepts_centroid_and_approximate() {
        assert_eq!(
            parse_voxel_mode("centroid").unwrap(),
            VoxelAggregationMode::Centroid
        );
        assert_eq!(
            parse_voxel_mode("approximate").unwrap(),
            VoxelAggregationMode::ApproximateFirst
        );
        assert_eq!(
            parse_voxel_mode("approximate-first").unwrap(),
            VoxelAggregationMode::ApproximateFirst
        );
    }

    #[test]
    fn parse_voxel_mode_rejects_unknown_value() {
        assert!(parse_voxel_mode("fast").is_err());
    }

    #[test]
    fn build_voxel_config_selects_mode_specific_defaults() {
        let centroid = build_voxel_config(0.2, VoxelAggregationMode::Centroid);
        assert_eq!(centroid.mode, VoxelAggregationMode::Centroid);
        assert_eq!(centroid.leaf_size, 0.2);

        let approximate = build_voxel_config(0.2, VoxelAggregationMode::ApproximateFirst);
        assert_eq!(approximate.mode, VoxelAggregationMode::ApproximateFirst);
        assert_eq!(approximate.leaf_size, 0.2);
    }

    #[test]
    fn detect_input_format_accepts_http_copc_urls() {
        assert_eq!(
            detect_input_format("https://example.com/path/scan.copc.laz"),
            Some(PointCloudFileFormat::Copc)
        );
        assert_eq!(
            detect_input_format("http://127.0.0.1:8080/data.copc.las"),
            Some(PointCloudFileFormat::Copc)
        );
        assert!(detect_input_format("https://example.com/cloud.laz").is_none());
    }
}

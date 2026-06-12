//! MVP pipeline CLI: point cloud in → labeled cluster output.
//!
//! ```text
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- input.pcd output.pcd
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
//!   --bounds 0,0,-1,100,100,1 scan.copc.laz roi.copc.laz
//! ```

use std::{env, path::Path, process::ExitCode, time::Instant};

use spatialrust::{
    detect_point_cloud_format, read_copc_file_with_query, read_point_cloud_file,
    write_point_cloud_file, CopcBounds, CopcQuery, ExecutionPolicy, MvpPipeline,
    MvpPipelineConfig, PointCloudFileFormat,
};

fn print_usage(program: &str) {
    eprintln!(
        "\
Usage: {program} [OPTIONS] <INPUT> <OUTPUT>

Run the MVP pipeline (voxel → normals → plane → cluster) and write labeled output.

Options:
  --leaf-size <METERS>       Voxel leaf size (default: 0.05)
  --voxel-policy <POLICY>    auto | cpu | gpu (default: auto)
  --bounds <MINX,MINY,MINZ,MAXX,MAXY,MAXZ>
                             COPC spatial query bounds (requires .copc.laz/.copc.las input)
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

fn load_input(
    input_path: &str,
    bounds: Option<CopcBounds>,
) -> Result<spatialrust::PointCloud, Box<dyn std::error::Error>> {
    match bounds {
        Some(bounds) => {
            let format = detect_point_cloud_format(input_path).ok_or_else(|| {
                format!("cannot detect input format for --bounds: {input_path}")
            })?;
            if format != PointCloudFileFormat::Copc {
                return Err(
                    "--bounds requires a COPC input (.copc.laz or .copc.las)".into(),
                );
            }

            eprintln!(
                "COPC bounds query: x=[{}, {}] y=[{}, {}] z=[{}, {}]",
                bounds.min[0],
                bounds.max[0],
                bounds.min[1],
                bounds.max[1],
                bounds.min[2],
                bounds.max[2],
            );
            Ok(read_copc_file_with_query(input_path, &CopcQuery::bounds(bounds))?)
        }
        None => Ok(read_point_cloud_file(input_path)?),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let program = env::args()
        .next()
        .unwrap_or_else(|| "spatialrust-mvp".to_string());

    let mut leaf_size = 0.05_f32;
    let mut voxel_policy = ExecutionPolicy::Auto;
    let mut copc_bounds = None;
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
            "--voxel-policy" => {
                let value = args
                    .next()
                    .ok_or("--voxel-policy requires auto, cpu, or gpu")?;
                voxel_policy = parse_voxel_policy(&value)?;
            }
            "--bounds" => {
                let value = args.next().ok_or("--bounds requires 6 comma-separated values")?;
                copc_bounds = Some(parse_bounds(&value)?);
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

    if !Path::new(&input_path).exists() {
        return Err(format!("input file not found: {input_path}").into());
    }

    eprintln!("loading {input_path}");
    let input = load_input(&input_path, copc_bounds)?;
    let input_points = input.len();
    eprintln!("input points: {input_points}");

    let config = MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(leaf_size),
        voxel_policy,
        ..MvpPipelineConfig::default()
    };

    eprintln!(
        "running MVP pipeline (leaf_size={leaf_size}, voxel_policy={voxel_policy:?})"
    );
    let started = Instant::now();
    let result = MvpPipeline::new(config).run(&input)?;
    let elapsed = started.elapsed();

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
    use super::parse_bounds;

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
}

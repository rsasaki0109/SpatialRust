//! MVP pipeline CLI: point cloud in → labeled cluster output.
//!
//! ```text
//! cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- input.pcd output.pcd
//! ```

use std::{env, path::Path, process::ExitCode, time::Instant};

use spatialrust::{
    read_point_cloud_file, write_point_cloud_file, ExecutionPolicy, MvpPipeline, MvpPipelineConfig,
};

fn print_usage(program: &str) {
    eprintln!(
        "\
Usage: {program} [OPTIONS] <INPUT> <OUTPUT>

Run the MVP pipeline (voxel → normals → plane → cluster) and write labeled output.

Options:
  --leaf-size <METERS>       Voxel leaf size (default: 0.05)
  --voxel-policy <POLICY>    auto | cpu | gpu (default: auto)
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let program = env::args()
        .next()
        .unwrap_or_else(|| "spatialrust-mvp".to_string());

    let mut leaf_size = 0.05_f32;
    let mut voxel_policy = ExecutionPolicy::Auto;
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
    let input = read_point_cloud_file(&input_path)?;
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

//! Bounded-memory point-cloud streaming command line workflow.

use std::error::Error;
use std::path::{Path, PathBuf};

use spatialrust::io::{
    CopcChunkSource, LasChunkSink, LasChunkSource, LasWriteFormat, PcdChunkSource, PlyChunkSource,
    SpoolOptions,
};
use spatialrust::math::{Mat3, Mat4, Vec3};
use spatialrust::pipeline::{StreamingPipeline, StreamingVoxelConfig};
use spatialrust::records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, StreamOptions,
    DEFAULT_STREAM_CHUNK_POINTS, DEFAULT_STREAM_MEMORY_BUDGET_BYTES,
};

#[derive(Debug)]
struct Config {
    input: String,
    output: PathBuf,
    chunk_points: usize,
    memory_bytes: u64,
    crop: Option<([f32; 3], [f32; 3])>,
    translation: Option<[f32; 3]>,
    voxel_leaf: Option<f32>,
    run_points: usize,
    max_runs: usize,
    spool_dir: PathBuf,
    spool_bytes: u64,
    receipt: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spatialrust-stream: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    let options = StreamOptions::new(config.chunk_points, MemoryBudget::new(config.memory_bytes)?)?;
    let cancellation = CancellationToken::default();
    let interrupt = cancellation.clone();
    ctrlc::set_handler(move || interrupt.cancel())?;

    let source = open_source(&config.input, options, cancellation)?;
    let mut pipeline = StreamingPipeline::new(source, config.input.clone())?;
    if let Some((min, max)) = config.crop {
        pipeline = pipeline.crop(min, max, false)?;
    }
    if let Some(translation) = config.translation {
        pipeline = pipeline.transform(Mat4::<f32>::from_rotation_translation(
            Mat3::<f32>::identity(),
            Vec3::new(translation[0], translation[1], translation[2]),
        ))?;
    }
    if let Some(leaf) = config.voxel_leaf {
        let spool = SpoolOptions::new(&config.spool_dir, config.spool_bytes)?;
        pipeline = pipeline.voxel(StreamingVoxelConfig::new(
            leaf,
            config.run_points,
            config.max_runs,
            spool,
        )?)?;
    }

    let format = output_format(&config.output)?;
    let mut sink =
        LasChunkSink::create_open_ended(&config.output, pipeline.schema().clone(), format)?;
    let receipt = pipeline.run_to_sink(&mut sink)?;
    let json = receipt.to_json()?;
    if let Some(path) = config.receipt {
        std::fs::write(path, format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn open_source(
    input: &str,
    options: StreamOptions,
    cancellation: CancellationToken,
) -> Result<Box<dyn BoundedSpatialRecordSource>, Box<dyn Error>> {
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(Box::new(CopcChunkSource::open_url(input, None, options, cancellation)?));
    }
    let lower = input.to_ascii_lowercase();
    if lower.ends_with(".copc.laz") {
        Ok(Box::new(CopcChunkSource::open(input, None, options, cancellation)?))
    } else if lower.ends_with(".pcd") {
        Ok(Box::new(PcdChunkSource::open(input, options, cancellation)?))
    } else if lower.ends_with(".ply") {
        Ok(Box::new(PlyChunkSource::open(input, options, cancellation)?))
    } else if lower.ends_with(".las") || lower.ends_with(".laz") {
        Ok(Box::new(LasChunkSource::open(input, options, cancellation)?))
    } else {
        Err(format!(
            "unsupported input '{input}'; expected PCD, PLY, LAS, LAZ, COPC, or an HTTP(S) COPC URL"
        )
        .into())
    }
}

fn output_format(path: &Path) -> Result<LasWriteFormat, Box<dyn Error>> {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".las") {
        Ok(LasWriteFormat::Las)
    } else if lower.ends_with(".laz") {
        Ok(LasWriteFormat::Laz)
    } else {
        Err("output must end in .las or .laz".into())
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, Box<dyn Error>> {
    let mut args = args.into_iter();
    let input = args.next().ok_or_else(usage)?;
    if input == "-h" || input == "--help" {
        return Err(usage().into());
    }
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut config = Config {
        input,
        output,
        chunk_points: DEFAULT_STREAM_CHUNK_POINTS,
        memory_bytes: DEFAULT_STREAM_MEMORY_BUDGET_BYTES,
        crop: None,
        translation: None,
        voxel_leaf: None,
        run_points: DEFAULT_STREAM_CHUNK_POINTS,
        max_runs: 1024,
        spool_dir: std::env::temp_dir(),
        spool_bytes: DEFAULT_STREAM_MEMORY_BUDGET_BYTES.saturating_mul(8),
        receipt: None,
    };
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--chunk-points" => config.chunk_points = parse_one(&mut args, &flag)?,
            "--memory-budget" => config.memory_bytes = parse_one(&mut args, &flag)?,
            "--voxel" => config.voxel_leaf = Some(parse_one(&mut args, &flag)?),
            "--run-points" => config.run_points = parse_one(&mut args, &flag)?,
            "--max-runs" => config.max_runs = parse_one(&mut args, &flag)?,
            "--spool-limit" => config.spool_bytes = parse_one(&mut args, &flag)?,
            "--spool-dir" => config.spool_dir = PathBuf::from(next_value(&mut args, &flag)?),
            "--receipt" => config.receipt = Some(PathBuf::from(next_value(&mut args, &flag)?)),
            "--translate" => {
                config.translation = Some([
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                ]);
            }
            "--crop" => {
                let values = [
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                    parse_one(&mut args, &flag)?,
                ];
                config.crop =
                    Some(([values[0], values[1], values[2]], [values[3], values[4], values[5]]));
            }
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        }
    }
    Ok(config)
}

fn parse_one<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(next_value(args, flag)?.parse()?)
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} requires another value").into())
}

fn usage() -> String {
    "usage: spatialrust-stream INPUT OUTPUT [--chunk-points N] [--memory-budget BYTES] \
     [--crop MINX MINY MINZ MAXX MAXY MAXZ] [--translate X Y Z] [--voxel LEAF] \
     [--run-points N] [--max-runs N] [--spool-dir DIR] [--spool-limit BYTES] \
     [--receipt PATH]"
        .into()
}

#[cfg(test)]
mod tests {
    use super::{output_format, parse_args};
    use spatialrust::io::LasWriteFormat;

    #[test]
    fn parses_workflow_options() {
        let config = parse_args(
            ["in.pcd", "out.laz", "--chunk-points", "10", "--voxel", "0.2", "--crop"]
                .into_iter()
                .chain(["0", "1", "2", "3", "4", "5"])
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.chunk_points, 10);
        assert_eq!(config.voxel_leaf, Some(0.2));
        assert_eq!(config.crop, Some(([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])));
        assert_eq!(output_format(&config.output).unwrap(), LasWriteFormat::Laz);
    }
}

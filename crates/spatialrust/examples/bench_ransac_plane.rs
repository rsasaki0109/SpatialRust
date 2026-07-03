//! Times CPU vs GPU RANSAC plane segmentation on a point cloud file.
//!
//! ```text
//! cargo run --release -p spatialrust --example bench_ransac_plane \
//!   --features segment-ransac-plane,segment-ransac-plane-gpu,io-pcd,gpu-wgpu,filter-voxel,feature-normal \
//!   -- cloud.pcd
//! ```
//!
//! Prints CSV lines: `backend,seconds,inlier_count,iterations`

use std::time::{Duration, Instant};

use spatialrust::{
    read_point_cloud_file, FeatureEstimator, GpuRansacPlaneSegmenter, NormalEstimationConfig,
    NormalEstimator, PointCloudFilter, RansacPlaneConfig, RansacPlaneSegmenter,
    VoxelGridDownsample, VoxelGridDownsampleConfig,
};

fn parse_usize(flag: &str, default: usize) -> usize {
    std::env::args()
        .position(|arg| arg == flag)
        .and_then(|index| std::env::args().nth(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_f32(flag: &str, default: f32) -> f32 {
    std::env::args()
        .position(|arg| arg == flag)
        .and_then(|index| std::env::args().nth(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_u64(flag: &str, default: u64) -> u64 {
    std::env::args()
        .position(|arg| arg == flag)
        .and_then(|index| std::env::args().nth(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}

fn config_from_args() -> RansacPlaneConfig {
    RansacPlaneConfig {
        distance_threshold: parse_f32("--distance-threshold", 0.025),
        max_iterations: parse_usize("--iterations", 1_000),
        min_inliers: parse_usize("--min-inliers", 3),
        seed: parse_u64("--seed", 42),
        ..Default::default()
    }
}

fn prepare_mvp_plane_input(
    cloud: spatialrust::PointCloud,
    leaf_size: f32,
) -> spatialrust::PointCloud {
    let downsampled = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(leaf_size))
        .filter(&cloud)
        .expect("voxel downsample");
    NormalEstimator::new(NormalEstimationConfig::default())
        .estimate(&downsampled)
        .expect("normal estimation")
}

fn bench_cpu(cloud: &spatialrust::PointCloud, config: RansacPlaneConfig) -> (Duration, usize) {
    let started = Instant::now();
    let result = RansacPlaneSegmenter::new(config).segment(cloud).expect("cpu ransac plane");
    (started.elapsed(), result.inlier_count)
}

fn bench_gpu(cloud: &spatialrust::PointCloud, config: RansacPlaneConfig) -> (Duration, usize) {
    let started = Instant::now();
    let result = GpuRansacPlaneSegmenter::new(config).segment(cloud).expect("gpu ransac plane");
    (started.elapsed(), result.inlier_count)
}

fn summarize(label: &str, samples: &[(Duration, usize)], iterations: usize) {
    let total: Duration = samples.iter().map(|(elapsed, _)| *elapsed).sum();
    let avg = total / samples.len().max(1) as u32;
    let inliers = samples.last().map(|(_, count)| *count).unwrap_or(0);
    println!("{label},{:.4},{inliers},{iterations}", avg.as_secs_f64());
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .filter(|arg| !arg.starts_with("--"))
        .expect("usage: bench_ransac_plane <cloud.pcd> [--mvp-leaf SIZE] [--iterations N]");

    let warmup = parse_usize("--warmup", 1);
    let repeat = parse_usize("--repeat", 3);
    let config = config_from_args();
    let mvp_leaf = if has_flag("--mvp-leaf") { Some(parse_f32("--mvp-leaf", 0.05)) } else { None };

    let mut cloud = read_point_cloud_file(&path).expect("failed to read cloud");
    if let Some(leaf_size) = mvp_leaf {
        cloud = prepare_mvp_plane_input(cloud, leaf_size);
        eprintln!("mvp preprocess (voxel leaf={leaf_size} + normals): {} points", cloud.len());
    }
    eprintln!(
        "benchmarking {} points (iterations={}, threshold={}, seed={})",
        cloud.len(),
        config.max_iterations,
        config.distance_threshold,
        config.seed
    );

    for _ in 0..warmup {
        let _ = RansacPlaneSegmenter::new(config).segment(&cloud);
    }

    let mut cpu_samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        cpu_samples.push(bench_cpu(&cloud, config));
    }
    summarize("cpu", &cpu_samples, config.max_iterations);

    if spatialrust::gpu::WgpuRuntime::shared().is_err() {
        eprintln!("gpu backend unavailable; skipping gpu row");
        return;
    }

    for _ in 0..warmup {
        let _ = GpuRansacPlaneSegmenter::new(config).segment(&cloud);
    }

    let mut gpu_samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        gpu_samples.push(bench_gpu(&cloud, config));
    }
    summarize("gpu", &gpu_samples, config.max_iterations);

    let cpu_avg = cpu_samples.iter().map(|(elapsed, _)| elapsed.as_secs_f64()).sum::<f64>()
        / cpu_samples.len() as f64;
    let gpu_avg = gpu_samples.iter().map(|(elapsed, _)| elapsed.as_secs_f64()).sum::<f64>()
        / gpu_samples.len() as f64;
    if gpu_avg > 0.0 {
        eprintln!("speedup (cpu/gpu): {:.2}x", cpu_avg / gpu_avg);
    }
}

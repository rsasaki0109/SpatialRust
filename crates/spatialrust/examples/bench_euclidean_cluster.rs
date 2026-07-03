//! Times CPU vs GPU Euclidean clustering on a point cloud file.
//!
//! ```text
//! cargo run --release -p spatialrust --example bench_euclidean_cluster \
//!   --features segment-euclidean,segment-euclidean-gpu,segment-ransac-plane,io-pcd,gpu-wgpu,filter-voxel,feature-normal \
//!   -- cloud.pcd
//! ```
//!
//! Prints CSV lines: `backend,seconds,cluster_count,point_count`

use std::time::{Duration, Instant};

use spatialrust::{
    read_point_cloud_file, EuclideanClusterConfig, EuclideanClusterExtractor, FeatureEstimator,
    GpuEuclideanClusterExtractor, NormalEstimationConfig, NormalEstimator, PointCloudFilter,
    RansacPlaneConfig, RansacPlaneSegmenter, VoxelGridDownsample, VoxelGridDownsampleConfig,
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

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}

fn config_from_args() -> EuclideanClusterConfig {
    EuclideanClusterConfig {
        cluster_tolerance: parse_f32("--tolerance", 0.05),
        min_cluster_size: parse_usize("--min-cluster-size", 1),
        max_cluster_size: parse_usize("--max-cluster-size", usize::MAX),
        ..Default::default()
    }
}

fn prepare_mvp_cluster_input(
    cloud: spatialrust::PointCloud,
    leaf_size: f32,
) -> spatialrust::PointCloud {
    let downsampled = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(leaf_size))
        .filter(&cloud)
        .expect("voxel downsample");
    let with_normals = NormalEstimator::new(NormalEstimationConfig::default())
        .estimate(&downsampled)
        .expect("normal estimation");
    let plane = RansacPlaneSegmenter::new(RansacPlaneConfig {
        distance_threshold: 0.025,
        max_iterations: 1_000,
        min_inliers: 10,
        seed: 42,
        ..Default::default()
    })
    .segment(&with_normals)
    .expect("plane segmentation");
    plane.outliers
}

fn bench_cpu(cloud: &spatialrust::PointCloud, config: EuclideanClusterConfig) -> (Duration, usize) {
    let started = Instant::now();
    let result = EuclideanClusterExtractor::new(config).extract(cloud).expect("cpu cluster");
    (started.elapsed(), result.cluster_count)
}

fn bench_gpu(cloud: &spatialrust::PointCloud, config: EuclideanClusterConfig) -> (Duration, usize) {
    let started = Instant::now();
    let result = GpuEuclideanClusterExtractor::new(config).extract(cloud).expect("gpu cluster");
    (started.elapsed(), result.cluster_count)
}

fn summarize(label: &str, samples: &[(Duration, usize)], point_count: usize) {
    let total: Duration = samples.iter().map(|(elapsed, _)| *elapsed).sum();
    let avg = total / samples.len().max(1) as u32;
    let clusters = samples.last().map(|(_, count)| *count).unwrap_or(0);
    let avg_secs = avg.as_secs_f64();
    println!("{label},{avg_secs:.4},{clusters},{point_count}");
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .filter(|arg| !arg.starts_with("--"))
        .expect("usage: bench_euclidean_cluster <cloud.pcd> [--mvp-leaf SIZE] [--tolerance R]");

    let warmup = parse_usize("--warmup", 1);
    let repeat = parse_usize("--repeat", 3);
    let config = config_from_args();
    let mvp_leaf = if has_flag("--mvp-leaf") { Some(parse_f32("--mvp-leaf", 0.05)) } else { None };

    let mut cloud = read_point_cloud_file(&path).expect("failed to read cloud");
    if let Some(leaf_size) = mvp_leaf {
        cloud = prepare_mvp_cluster_input(cloud, leaf_size);
        eprintln!(
            "mvp cluster input (voxel leaf={leaf_size} + plane outliers): {} points",
            cloud.len()
        );
    }
    let point_count = cloud.len();
    eprintln!(
        "benchmarking {point_count} points (tolerance={}, min_cluster_size={})",
        config.cluster_tolerance, config.min_cluster_size
    );

    for _ in 0..warmup {
        let _ = EuclideanClusterExtractor::new(config).extract(&cloud);
    }

    let mut cpu_samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        cpu_samples.push(bench_cpu(&cloud, config));
    }
    summarize("cpu", &cpu_samples, point_count);

    if spatialrust::gpu::WgpuRuntime::shared().is_err() {
        eprintln!("gpu backend unavailable; skipping gpu row");
        return;
    }

    for _ in 0..warmup {
        let _ = GpuEuclideanClusterExtractor::new(config).extract(&cloud);
    }

    let mut gpu_samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        gpu_samples.push(bench_gpu(&cloud, config));
    }
    summarize("gpu", &gpu_samples, point_count);

    let cpu_avg = cpu_samples.iter().map(|(elapsed, _)| elapsed.as_secs_f64()).sum::<f64>()
        / cpu_samples.len() as f64;
    let gpu_avg = gpu_samples.iter().map(|(elapsed, _)| elapsed.as_secs_f64()).sum::<f64>()
        / gpu_samples.len() as f64;
    if gpu_avg > 0.0 {
        eprintln!("speedup (cpu/gpu): {:.2}x", cpu_avg / gpu_avg);
    }
}

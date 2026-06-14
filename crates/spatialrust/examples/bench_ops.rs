//! Times core point-cloud operations on a PCD file, for the PCL comparison.
//!
//! Run with the `mvp` + `filter-outlier` features:
//! ```text
//! cargo run --release --example bench_ops --features mvp,filter-outlier -- cloud.pcd
//! ```
//! Prints `operation,seconds,output_points` lines on stdout.

use std::time::Instant;

use spatialrust::{
    read_point_cloud_file, FeatureEstimator, NormalEstimationConfig, NormalEstimator,
    PointCloudFilter, RadiusOutlierConfig, RadiusOutlierRemoval, StatisticalOutlierConfig,
    StatisticalOutlierRemoval, VoxelGridDownsample, VoxelGridDownsampleConfig,
};

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench_ops <cloud.pcd>");
    let cloud = read_point_cloud_file(&path).expect("failed to read cloud");
    eprintln!("loaded {} points from {path}", cloud.len());

    // Voxel-grid downsample (leaf 0.05).
    let vg = VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(0.05));
    let t = Instant::now();
    let downsampled = vg.filter(&cloud).expect("voxel downsample");
    println!("voxel_downsample,{:.4},{}", t.elapsed().as_secs_f64(), downsampled.len());

    // Normal estimation (k = 10).
    let ne = NormalEstimator::new(NormalEstimationConfig::k_neighbors(10));
    let t = Instant::now();
    let normals = ne.estimate(&cloud).expect("normal estimation");
    println!("normal_estimation,{:.4},{}", t.elapsed().as_secs_f64(), normals.len());

    // Statistical Outlier Removal (k = 16, std = 1.0).
    let sor = StatisticalOutlierRemoval::new(StatisticalOutlierConfig::new(16, 1.0));
    let t = Instant::now();
    let cleaned = sor.filter(&cloud).expect("statistical outlier removal");
    println!("statistical_outlier_removal,{:.4},{}", t.elapsed().as_secs_f64(), cleaned.len());

    // Radius Outlier Removal (radius 0.1, min 4).
    let ror = RadiusOutlierRemoval::new(RadiusOutlierConfig::new(0.1, 4));
    let t = Instant::now();
    let radius_cleaned = ror.filter(&cloud).expect("radius outlier removal");
    println!("radius_outlier_removal,{:.4},{}", t.elapsed().as_secs_f64(), radius_cleaned.len());
}

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spatialrust_core::{ExecutionPolicy, PointCloud, PointCloudBuilder, StandardSchemas};
use spatialrust_features::NormalEstimationConfig;
use spatialrust_filtering::VoxelGridDownsampleConfig;
use spatialrust_math::Vec3;
use spatialrust_pipeline::{MvpPipeline, MvpPipelineConfig};
use spatialrust_segmentation::{EuclideanClusterConfig, RansacPlaneConfig};

fn synthetic_scan_xyzi(point_count: usize) -> PointCloud {
    let side = (point_count as f64).sqrt().ceil() as usize;
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
    for index in 0..point_count {
        let x = (index % side) as f32 * 0.1;
        let y = (index / side) as f32 * 0.1;
        let intensity = (index % 256) as f32;
        builder
            .push_point([x, y, 0.0, intensity])
            .expect("push point");
    }
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([
                    side as f32 * 0.1 + 10.0 + x as f32 * 0.01,
                    y as f32 * 0.01,
                    0.5,
                    128.0,
                ])
                .expect("push bump point");
        }
    }
    builder.build().expect("point cloud")
}

fn mvp_config(voxel_policy: ExecutionPolicy) -> MvpPipelineConfig {
    MvpPipelineConfig {
        voxel: VoxelGridDownsampleConfig::centroid(4.0).without_gpu_min_points(),
        voxel_policy,
        normals: NormalEstimationConfig {
            k_neighbors: 16,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 100.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.2,
            max_iterations: 200,
            min_inliers: 100,
            seed: 42,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 1.0,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    }
}

fn bench_mvp_large_scale(c: &mut Criterion) {
    let cpu_pipeline = MvpPipeline::new(mvp_config(ExecutionPolicy::CpuSingle));

    for point_count in [500_000_usize, 1_000_000, 2_000_000] {
        let input = synthetic_scan_xyzi(point_count);
        let mut group = c.benchmark_group(format!("mvp_large_scale/{point_count}"));
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_function("cpu_full_pipeline", |bencher| {
            bencher.iter(|| {
                black_box(cpu_pipeline.run(&input).expect("mvp cpu"));
            });
        });
        group.finish();
    }

    #[cfg(feature = "pipeline-mvp-gpu")]
    {
        use spatialrust_core::DeviceKind;

        let gpu_pipeline = MvpPipeline::new(mvp_config(ExecutionPolicy::Gpu(DeviceKind::Wgpu)));
        for point_count in [500_000_usize, 1_000_000, 2_000_000] {
            let input = synthetic_scan_xyzi(point_count);
            let mut group = c.benchmark_group(format!("mvp_large_scale/{point_count}"));
            group.throughput(Throughput::Elements(point_count as u64));
            group.bench_function("gpu_full_pipeline", |bencher| {
                bencher.iter(|| {
                    black_box(gpu_pipeline.run(&input).expect("mvp gpu"));
                });
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_mvp_large_scale);
criterion_main!(benches);

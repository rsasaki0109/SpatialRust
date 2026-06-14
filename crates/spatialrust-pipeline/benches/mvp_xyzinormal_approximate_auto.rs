use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spatialrust_core::{ExecutionPolicy, PointCloud, PointCloudBuilder, StandardSchemas};
use spatialrust_features::NormalEstimationConfig;
use spatialrust_filtering::VoxelGridDownsampleConfig;
use spatialrust_math::Vec3;
use spatialrust_pipeline::{MvpPipeline, MvpPipelineConfig};
use spatialrust_segmentation::{EuclideanClusterConfig, RansacPlaneConfig};

fn synthetic_xyzinormal_plane_with_bump(point_count: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
    for index in 0..point_count {
        let x = (index % 256) as f32 * 0.1;
        let y = ((index / 256) % 256) as f32 * 0.1;
        let intensity = (index % 256) as f32;
        builder
            .push_point([x, y, 0.0, intensity, 0.0, 0.0, 1.0])
            .expect("push point");
    }
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([90.0 + x as f32 * 0.02, y as f32 * 0.02, 2.5, 200.0, 0.0, 0.0, 1.0])
                .expect("push bump point");
        }
    }
    builder.build().expect("point cloud")
}

fn mvp_config(voxel_policy: ExecutionPolicy) -> MvpPipelineConfig {
    MvpPipelineConfig {
        voxel: VoxelGridDownsampleConfig::approximate(4.0),
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
            min_inliers: 10,
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

fn bench_mvp_xyzinormal_approximate_auto(c: &mut Criterion) {
    let auto_pipeline = MvpPipeline::new(mvp_config(ExecutionPolicy::Auto));
    let cpu_pipeline = MvpPipeline::new(mvp_config(ExecutionPolicy::CpuSingle));

    for point_count in [500_000_usize, 1_000_000, 2_000_000] {
        let input = synthetic_xyzinormal_plane_with_bump(point_count);
        let throughput = (input.len()) as u64;
        let mut group = c.benchmark_group(format!("mvp_xyzinormal_approximate_auto/{point_count}"));
        group.throughput(Throughput::Elements(throughput));
        group.bench_function("auto_full_pipeline", |bencher| {
            bencher.iter(|| {
                black_box(auto_pipeline.run(&input).expect("mvp auto"));
            });
        });
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
            let input = synthetic_xyzinormal_plane_with_bump(point_count);
            let throughput = (input.len()) as u64;
            let mut group =
                c.benchmark_group(format!("mvp_xyzinormal_approximate_auto/{point_count}"));
            group.throughput(Throughput::Elements(throughput));
            group.bench_function("gpu_full_pipeline", |bencher| {
                bencher.iter(|| {
                    black_box(gpu_pipeline.run(&input).expect("mvp gpu"));
                });
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_mvp_xyzinormal_approximate_auto);
criterion_main!(benches);

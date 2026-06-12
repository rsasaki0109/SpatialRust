use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spatialrust_core::{ExecutionPolicy, PointCloudBuilder, StandardSchemas};
use spatialrust_filtering::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig};

fn synthetic_point_cloud(point_count: usize) -> spatialrust_core::PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
    for index in 0..point_count {
        let t = index as f32;
        builder
            .push_point([
                (t * 0.013).fract() * 100.0,
                ((index % 97) as f32) * 0.017,
                ((index % 53) as f32) * 0.019,
                (index % 255) as f32,
            ])
            .expect("push point");
    }
    builder.build().expect("point cloud")
}

fn bench_voxel_downsample(c: &mut Criterion) {
    let centroid = VoxelGridDownsample::new(
        VoxelGridDownsampleConfig::centroid(4.0).without_gpu_min_points(),
    );
    let approximate = VoxelGridDownsample::new(
        VoxelGridDownsampleConfig::approximate(4.0).without_gpu_min_points(),
    );

    for point_count in [10_000_usize, 65_536, 100_000, 200_000, 500_000] {
        let input = synthetic_point_cloud(point_count);
        let mut group = c.benchmark_group(format!("voxel_downsample/{point_count}"));
        group.throughput(Throughput::Elements(point_count as u64));

        group.bench_function("cpu_centroid", |bencher| {
            bencher.iter(|| {
                black_box(centroid.filter(&input).expect("cpu centroid"));
            });
        });

        group.bench_function("gpu_centroid", |bencher| {
            bencher.iter(|| {
                black_box(
                    centroid
                        .filter_with_policy(
                            &input,
                            ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu),
                        )
                        .expect("gpu centroid"),
                );
            });
        });

        group.bench_function("cpu_approximate_first", |bencher| {
            bencher.iter(|| {
                black_box(approximate.filter(&input).expect("cpu approximate-first"));
            });
        });

        group.bench_function("gpu_approximate_first", |bencher| {
            bencher.iter(|| {
                black_box(
                    approximate
                        .filter_with_policy(
                            &input,
                            ExecutionPolicy::Gpu(spatialrust_core::DeviceKind::Wgpu),
                        )
                        .expect("gpu approximate-first"),
                );
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_voxel_downsample);
criterion_main!(benches);

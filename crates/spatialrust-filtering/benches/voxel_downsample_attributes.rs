use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spatialrust_core::{
    ExecutionPolicy, PointCloud, PointCloudBuilder, PointSchema, StandardSchemas,
};
use spatialrust_filtering::{PointCloudFilter, VoxelGridDownsample, VoxelGridDownsampleConfig};

fn synthetic_values(schema: &PointSchema, index: usize) -> Vec<f32> {
    let t = index as f32;
    let x = (t * 0.013).fract() * 100.0;
    let y = ((index % 97) as f32) * 0.017;
    let z = ((index % 53) as f32) * 0.019;
    let intensity = (index % 255) as f32;
    let r = (index % 256) as f32;
    let g = ((index / 3) % 256) as f32;
    let b = ((index / 7) % 256) as f32;
    let nx = (x * 0.01).sin();
    let ny = (y * 0.01).cos();
    let nz = 1.0;

    schema
        .fields()
        .iter()
        .map(|field| {
            use spatialrust_core::FieldSemantic;
            match field.semantic {
                FieldSemantic::PositionX => x,
                FieldSemantic::PositionY => y,
                FieldSemantic::PositionZ => z,
                FieldSemantic::Intensity => intensity,
                FieldSemantic::ColorR => r,
                FieldSemantic::ColorG => g,
                FieldSemantic::ColorB => b,
                FieldSemantic::NormalX => nx,
                FieldSemantic::NormalY => ny,
                FieldSemantic::NormalZ => nz,
                _ => 0.0,
            }
        })
        .collect()
}

fn synthetic_point_cloud(schema: PointSchema, point_count: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::new(schema.clone());
    for index in 0..point_count {
        builder.push_point(synthetic_values(&schema, index)).expect("push point");
    }
    builder.build().expect("point cloud")
}

fn bench_voxel_downsample_attributes(c: &mut Criterion) {
    let centroid =
        VoxelGridDownsample::new(VoxelGridDownsampleConfig::centroid(4.0).without_gpu_min_points());
    let approximate = VoxelGridDownsample::new(
        VoxelGridDownsampleConfig::approximate(4.0).without_gpu_min_points(),
    );

    let schemas = [
        ("xyz", StandardSchemas::point_xyz()),
        ("xyzi", StandardSchemas::point_xyzi()),
        ("xyzrgb", StandardSchemas::point_xyzrgb()),
        ("xyzinormal", StandardSchemas::point_xyzinormal()),
    ];

    for point_count in [500_000_usize, 1_000_000, 2_000_000] {
        for (label, schema) in &schemas {
            let input = synthetic_point_cloud(schema.clone(), point_count);
            let mut group =
                c.benchmark_group(format!("voxel_downsample_attributes/{point_count}/{label}"));
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
}

criterion_group!(benches, bench_voxel_downsample_attributes);
criterion_main!(benches);

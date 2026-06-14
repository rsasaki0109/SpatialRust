use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spatialrust_core::PointCloud;
use spatialrust_features::{
    FeatureEstimator, GpuNormalEstimator, NormalEstimationConfig, NormalEstimator,
};
use spatialrust_core::{PointCloudBuilder, StandardSchemas};

/// A gently undulating height field so normals vary across the surface.
fn synthetic_surface(point_count: usize) -> PointCloud {
    let side = (point_count as f64).sqrt() as usize;
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    let mut count = 0;
    'outer: for i in 0..side {
        for j in 0..side {
            let x = i as f32 * 0.05;
            let y = j as f32 * 0.05;
            let z = (x * 0.7).sin() * 0.1 + (y * 0.5).cos() * 0.1;
            builder.push_point([x, y, z]).expect("push point");
            count += 1;
            if count >= point_count {
                break 'outer;
            }
        }
    }
    builder.build().expect("point cloud")
}

fn bench_normals(c: &mut Criterion) {
    let config = NormalEstimationConfig::k_neighbors(20);
    let cpu = NormalEstimator::new(config);
    let gpu = GpuNormalEstimator::new(config);

    for point_count in [10_000_usize, 50_000, 100_000, 200_000, 500_000] {
        let input = synthetic_surface(point_count);
        let mut group = c.benchmark_group(format!("normals/{point_count}"));
        group.throughput(Throughput::Elements(point_count as u64));

        group.bench_function("cpu", |bencher| {
            bencher.iter(|| {
                black_box(cpu.estimate(&input).expect("cpu normals"));
            });
        });

        group.bench_function("gpu", |bencher| {
            bencher.iter(|| {
                black_box(gpu.estimate(&input).expect("gpu normals"));
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_normals);
criterion_main!(benches);

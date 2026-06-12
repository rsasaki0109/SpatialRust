use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::{
    downsample_voxel_approximate_first_gpu, downsample_voxel_centroid_gpu, WgpuRuntime,
};

const ORIGIN: [f32; 3] = [0.0, 0.0, 0.0];
const INV_LEAF: f32 = 0.25;

fn synthetic_positions(point_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(point_count);
    let mut y = Vec::with_capacity(point_count);
    let mut z = Vec::with_capacity(point_count);

    for index in 0..point_count {
        let t = index as f32;
        x.push((t * 0.013).fract() * 100.0);
        y.push(((index % 97) as f32) * 0.017);
        z.push(((index % 53) as f32) * 0.019);
    }

    (x, y, z)
}

fn bench_centroid_shared_runtime(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_centroid/shared_runtime");
    let runtime = WgpuRuntime::shared().expect("shared wgpu runtime");
    let _ = runtime.pipelines();

    for point_count in [10_000_usize, 65_536, 100_000] {
        let (x, y, z) = synthetic_positions(point_count);
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(point_count),
            &(x, y, z),
            |bencher, (x, y, z)| {
                bencher.iter(|| {
                    black_box(
                        downsample_voxel_centroid_gpu(
                            &runtime,
                            x,
                            y,
                            z,
                            ORIGIN,
                            INV_LEAF,
                        )
                        .expect("centroid pipeline"),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_approximate_first_shared_runtime(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_approximate_first/shared_runtime");
    let runtime = WgpuRuntime::shared().expect("shared wgpu runtime");
    let _ = runtime.pipelines();

    for point_count in [10_000_usize, 65_536, 100_000] {
        let (x, y, z) = synthetic_positions(point_count);
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(point_count),
            &(x, y, z),
            |bencher, (x, y, z)| {
                bencher.iter(|| {
                    black_box(
                        downsample_voxel_approximate_first_gpu(
                            &runtime,
                            x,
                            y,
                            z,
                            ORIGIN,
                            INV_LEAF,
                        )
                        .expect("approximate-first pipeline"),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_pipeline_cache(c: &mut Criterion) {
    let point_count = 65_536_usize;
    let (x, y, z) = synthetic_positions(point_count);

    let mut group = c.benchmark_group("gpu_pipeline_cache");
    group.throughput(Throughput::Elements(point_count as u64));

    group.bench_function("shared_runtime_cached_pipelines", |bencher| {
        let runtime = WgpuRuntime::shared().expect("shared wgpu runtime");
        let _ = runtime.pipelines();
        bencher.iter(|| {
            black_box(
                downsample_voxel_centroid_gpu(&runtime, &x, &y, &z, ORIGIN, INV_LEAF)
                    .expect("centroid pipeline"),
            );
        });
    });

    group.bench_function("fresh_runtime_per_iteration", |bencher| {
        bencher.iter(|| {
            let runtime = WgpuRuntime::new_headless().expect("headless wgpu runtime");
            black_box(
                downsample_voxel_centroid_gpu(&runtime, &x, &y, &z, ORIGIN, INV_LEAF)
                    .expect("centroid pipeline"),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_centroid_shared_runtime,
    bench_approximate_first_shared_runtime,
    bench_pipeline_cache
);
criterion_main!(benches);

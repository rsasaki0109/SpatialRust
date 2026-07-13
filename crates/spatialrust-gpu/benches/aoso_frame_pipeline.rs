use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_core::{PointCloudBuilder, SpatialTensor};
use spatialrust_gpu::{run_aoso_voxel_normal_frame, WgpuRuntime};

fn synthetic_cloud(point_count: usize) -> spatialrust_core::PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    let side = (point_count as f32).sqrt().ceil() as usize;
    for index in 0..point_count {
        let x = (index % side) as f32 * 0.03;
        let y = (index / side) as f32 * 0.03;
        let z = ((index % 17) as f32 * 0.001).sin() * 0.01;
        builder.push_point([x, y, z]).expect("synthetic point");
    }
    builder.build().expect("synthetic cloud")
}

fn bench_aoso_frame_pipeline(c: &mut Criterion) {
    let runtime = WgpuRuntime::shared().expect("shared wgpu runtime");
    let mut group = c.benchmark_group("aoso_frame/upload_voxel_radius_normals");

    for point_count in [10_000_usize, 65_536] {
        let cloud = synthetic_cloud(point_count);
        let tensor = SpatialTensor::new(&cloud, 16_384).expect("spatial tensor");
        group.throughput(Throughput::Elements(point_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(point_count),
            &tensor,
            |bencher, tensor| {
                bencher.iter(|| {
                    let frame = run_aoso_voxel_normal_frame(
                        &runtime,
                        black_box(tensor),
                        [0.0; 3],
                        20.0,
                        0.08,
                    )
                    .expect("AoSoA frame pipeline");
                    black_box(frame.receipt());
                    frame.recycle(&runtime).expect("frame recycle");
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_aoso_frame_pipeline);
criterion_main!(benches);

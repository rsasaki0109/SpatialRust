use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::{build_voxel_segments_gpu_from_keys_buffer, WgpuRuntime};
use wgpu::util::DeviceExt;

fn bench_segments(c: &mut Criterion) {
    let runtime = WgpuRuntime::shared().unwrap();
    eprintln!("Segment benchmark adapter: {:?}", runtime.adapter_info());
    let mut group = c.benchmark_group("gpu_segments/resident_keys");
    for n in [65_536_u32, 1_048_576] {
        let keys: Vec<[i32; 4]> = (0..n)
            .map(|i| {
                [
                    (i * 37 % 4093) as i32 - 2046,
                    (i * 13 % 257) as i32 - 128,
                    (i * 7 % 17) as i32 - 8,
                    0,
                ]
            })
            .collect();
        let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("segment-benchmark-keys"),
            contents: bytemuck::cast_slice(&keys),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let _ = build_voxel_segments_gpu_from_keys_buffer(&runtime, &input, n, n).unwrap();
        runtime.wait_idle();
        group.throughput(Throughput::Elements(u64::from(n)));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let segments =
                    build_voxel_segments_gpu_from_keys_buffer(&runtime, black_box(&input), n, n)
                        .unwrap();
                runtime.wait_idle();
                black_box(segments);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_segments);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_vision::{
    distance_transform_edt, distance_transform_edt_into, BinaryMask, DistanceTransformWorkspace,
};

fn benchmark_exact_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform_edt");
    group.sample_size(10);
    for &(name, width, height) in &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height)
            .map(|index| u8::from(index % 97 != 0 && index % (width * 11 + 1) != 0))
            .collect();
        let mask = BinaryMask::try_new(width, height, data).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(distance_transform_edt(black_box(&mask)).unwrap()));
        });
        let mut output = vec![0.0_f32; width * height];
        let mut workspace = DistanceTransformWorkspace::new();
        distance_transform_edt_into(&mask, &mut output, &mut workspace).unwrap();
        group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| {
                distance_transform_edt_into(
                    black_box(&mask),
                    black_box(&mut output),
                    black_box(&mut workspace),
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_exact_distance_transform);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{detect_and_describe_orb, detect_fast, FastOptions, OrbOptions};

fn benchmark_feature2d(c: &mut Criterion) {
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                ((x * 37 + y * 19) ^ (x * y * 3) ^ ((x / 8 + y / 8) * 127)) as u8
            })
            .collect();
        let image = Image::<u8, 1>::try_new(width, height, data).unwrap();

        let mut fast = c.benchmark_group("fast_9_16");
        fast.sample_size(10);
        fast.throughput(Throughput::Elements((width * height) as u64));
        fast.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(detect_fast(image.view(), FastOptions::default()).unwrap());
            });
        });
        fast.finish();

        let mut orb = c.benchmark_group("orb_500");
        orb.sample_size(10);
        orb.throughput(Throughput::Elements((width * height) as u64));
        orb.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(detect_and_describe_orb(image.view(), OrbOptions::default()).unwrap());
            });
        });
        orb.finish();
    }
}

criterion_group!(benches, benchmark_feature2d);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{canny, CannyOptions};

fn benchmark_canny(c: &mut Criterion) {
    let mut group = c.benchmark_group("canny");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height)
            .map(|index| ((index * 37 + index / width * 17) & 255) as u8)
            .collect();
        let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(canny(image.view(), CannyOptions::default()).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_canny);
criterion_main!(benches);

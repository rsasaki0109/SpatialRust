use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{clahe, equalize_histogram, integral_image, threshold, ThresholdType};

fn benchmark_analysis(c: &mut Criterion) {
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height).map(|index| (index & 255) as u8).collect();
        let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
        for (operation, mut run) in [
            (
                "threshold",
                Box::new(|| {
                    black_box(
                        threshold(image.view(), 127.0, 255.0, ThresholdType::Binary).unwrap(),
                    );
                }) as Box<dyn FnMut()>,
            ),
            (
                "equalize_histogram",
                Box::new(|| {
                    black_box(equalize_histogram(image.view()).unwrap());
                }),
            ),
            (
                "clahe_8x8",
                Box::new(|| {
                    black_box(clahe(image.view(), 2.0, 8, 8).unwrap());
                }),
            ),
            (
                "integral",
                Box::new(|| {
                    black_box(integral_image(image.view(), 0).unwrap());
                }),
            ),
        ] {
            let mut group = c.benchmark_group(operation);
            group.sample_size(10);
            group.throughput(Throughput::Elements((width * height) as u64));
            group.bench_function(BenchmarkId::from_parameter(name), |b| b.iter(&mut run));
            group.finish();
        }
    }
}

criterion_group!(benches, benchmark_analysis);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    morphology_ex, BorderMode, MorphologyOperation, MorphologyShape, StructuringElement,
};

fn benchmark_morphology(c: &mut Criterion) {
    let element = StructuringElement::try_new(MorphologyShape::Ellipse, 5, 5).unwrap();
    let mut group = c.benchmark_group("morphology_open_gray8_5x5");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let image = Image::<u8, 1>::try_new(width, height, vec![127; width * height]).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                morphology_ex(
                    black_box(image.view()),
                    MorphologyOperation::Open,
                    &element,
                    1,
                    BorderMode::Replicate,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_morphology);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    morphology_ex, morphology_rect_u8, morphology_rect_u8_into, BorderMode, MorphologyOperation,
    MorphologyShape, RectMorphologyWorkspace, StructuringElement,
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

    for kernel_size in [5, 511] {
        let element =
            StructuringElement::try_new(MorphologyShape::Rect, kernel_size, kernel_size).unwrap();
        let mut group =
            c.benchmark_group(format!("morphology_open_gray8_rect_{kernel_size}x{kernel_size}"));
        group.sample_size(10);
        for &(name, width, height) in
            &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)]
        {
            let pixels = (0..width * height)
                .map(|index| ((index * 73 + index / width * 29) & 255) as u8)
                .collect();
            let image = Image::<u8, 1>::try_new(width, height, pixels).unwrap();
            group.throughput(Throughput::Elements((width * height) as u64));
            group.bench_with_input(BenchmarkId::new("allocate", name), &image, |b, image| {
                b.iter(|| {
                    morphology_rect_u8(
                        black_box(image.view()),
                        MorphologyOperation::Open,
                        &element,
                        1,
                        BorderMode::Replicate,
                    )
                    .unwrap()
                });
            });
            let mut output = vec![0; width * height];
            let mut workspace = RectMorphologyWorkspace::new();
            morphology_rect_u8_into(
                image.view(),
                MorphologyOperation::Open,
                &element,
                1,
                BorderMode::Replicate,
                &mut output,
                &mut workspace,
            )
            .unwrap();
            group.bench_with_input(BenchmarkId::new("reuse", name), &image, |b, image| {
                b.iter(|| {
                    morphology_rect_u8_into(
                        black_box(image.view()),
                        MorphologyOperation::Open,
                        &element,
                        1,
                        BorderMode::Replicate,
                        &mut output,
                        &mut workspace,
                    )
                    .unwrap();
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_morphology);
criterion_main!(benches);

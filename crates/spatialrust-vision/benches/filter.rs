use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    bilateral_filter, gaussian_blur, median_blur, pyr_down, sobel, BorderMode,
};

fn benchmark_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur_rgb8_5x5");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let image = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                gaussian_blur(black_box(image.view()), 5, 5, 1.2, 1.2, BorderMode::Reflect101)
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn benchmark_advanced_filters(c: &mut Criterion) {
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let rgb = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        let gray = Image::<u8, 1>::try_new(width, height, vec![127; width * height]).unwrap();
        let throughput = Throughput::Elements((width * height) as u64);
        for (operation, mut run) in [
            (
                "median_3x3",
                Box::new(|| {
                    black_box(median_blur(rgb.view(), 3, BorderMode::Replicate).unwrap());
                }) as Box<dyn FnMut()>,
            ),
            (
                "bilateral_d5",
                Box::new(|| {
                    black_box(
                        bilateral_filter(rgb.view(), 5, 40.0, 3.0, BorderMode::Reflect101).unwrap(),
                    );
                }),
            ),
            (
                "sobel_3x3",
                Box::new(|| {
                    black_box(
                        sobel(gray.view(), 1, 0, 3, 1.0, 0.0, BorderMode::Reflect101).unwrap(),
                    );
                }),
            ),
            (
                "pyr_down",
                Box::new(|| {
                    black_box(pyr_down(rgb.view(), BorderMode::Reflect101).unwrap());
                }),
            ),
        ] {
            let mut group = c.benchmark_group(operation);
            group.sample_size(10);
            group.throughput(throughput.clone());
            group.bench_function(name, |b| b.iter(&mut run));
            group.finish();
        }
    }
}

criterion_group!(benches, benchmark_gaussian, benchmark_advanced_filters);
criterion_main!(benches);

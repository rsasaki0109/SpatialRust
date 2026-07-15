use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    bilateral_filter, gaussian_blur, gaussian_blur_u8, gaussian_blur_u8_into, median_blur,
    pyr_down, sobel, sobel_l1_magnitude_u8, sobel_l1_magnitude_u8_into, spatial_gradient_u8,
    BorderMode, GaussianBlurU8Workspace,
};

fn benchmark_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur_rgb8_5x5");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let image = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        let mut output = vec![0_u8; width * height * 3];
        let mut workspace = GaussianBlurU8Workspace::new();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_with_input(BenchmarkId::new("legacy_allocate", name), &image, |b, image| {
            b.iter(|| {
                gaussian_blur(black_box(image.view()), 5, 5, 1.2, 1.2, BorderMode::Reflect101)
                    .unwrap()
            });
        });
        group.bench_with_input(
            BenchmarkId::new("specialized_allocate", name),
            &image,
            |b, image| {
                b.iter(|| {
                    gaussian_blur_u8(
                        black_box(image.view()),
                        5,
                        5,
                        1.2,
                        1.2,
                        BorderMode::Reflect101,
                    )
                    .unwrap()
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("specialized_reuse", name), &image, |b, image| {
            b.iter(|| {
                gaussian_blur_u8_into(
                    black_box(image.view()),
                    5,
                    5,
                    1.2,
                    1.2,
                    BorderMode::Reflect101,
                    black_box(&mut output),
                    black_box(&mut workspace),
                )
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

fn benchmark_paired_sobel(c: &mut Criterion) {
    for &(name, width, height) in &[("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let image = Image::<u8, 1>::try_new(
            width,
            height,
            (0..width * height).map(|index| ((index * 37 + 11) & 255) as u8).collect(),
        )
        .unwrap();
        let mut magnitude = vec![0_i16; width * height];
        let mut group = c.benchmark_group("paired_sobel_3x3");
        group.sample_size(10);
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::new("xy_allocate", name), |b| {
            b.iter(|| spatial_gradient_u8(black_box(image.view()), BorderMode::Reflect101).unwrap())
        });
        group.bench_function(BenchmarkId::new("l1_allocate", name), |b| {
            b.iter(|| {
                sobel_l1_magnitude_u8(black_box(image.view()), BorderMode::Reflect101).unwrap()
            })
        });
        group.bench_function(BenchmarkId::new("l1_reuse", name), |b| {
            b.iter(|| {
                sobel_l1_magnitude_u8_into(
                    black_box(image.view()),
                    BorderMode::Reflect101,
                    black_box(&mut magnitude),
                )
                .unwrap()
            })
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark_gaussian, benchmark_advanced_filters, benchmark_paired_sobel);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    canny, canny_into, canny_with_intermediates, CannyOptions, CannyWorkspace,
};

fn benchmark_canny(c: &mut Criterion) {
    let mut group = c.benchmark_group("canny");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height)
            .map(|index| ((index * 37 + index / width * 17) & 255) as u8)
            .collect();
        let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::new("allocate", name), |b| {
            b.iter(|| {
                black_box(canny(image.view(), CannyOptions::default()).unwrap());
            });
        });
        group.bench_function(BenchmarkId::new("inspectable", name), |b| {
            b.iter(|| {
                black_box(canny_with_intermediates(image.view(), CannyOptions::default()).unwrap());
            });
        });
        let mut output = Image::<u8, 1>::from_pixel(width, height, [0]).unwrap();
        let mut workspace = CannyWorkspace::new();
        canny_into(image.view(), CannyOptions::default(), output.view_mut(), &mut workspace)
            .unwrap();
        group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| {
                canny_into(
                    image.view(),
                    CannyOptions::default(),
                    output.view_mut(),
                    &mut workspace,
                )
                .unwrap();
                black_box(output.as_slice());
            });
        });
    }
    group.finish();
}

fn benchmark_canny_document_lines(c: &mut Criterion) {
    let mut group = c.benchmark_group("canny_document_lines");
    group.sample_size(10);
    for &(name, width, height) in &[("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let mut data = vec![0_u8; width * height];
        for y in (20..height).step_by(80) {
            for row in y..(y + 3).min(height) {
                data[row * width + 10..row * width + width - 10].fill(255);
            }
        }
        let image = Image::<u8, 1>::try_new(width, height, data).unwrap();
        let options = CannyOptions {
            low_threshold: 80.0,
            high_threshold: 160.0,
            l2_gradient: true,
            ..Default::default()
        };
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::new("inspectable", name), |b| {
            b.iter(|| black_box(canny_with_intermediates(image.view(), options).unwrap()));
        });
        let mut output = Image::<u8, 1>::from_pixel(width, height, [0]).unwrap();
        let mut workspace = CannyWorkspace::new();
        canny_into(image.view(), options, output.view_mut(), &mut workspace).unwrap();
        group.bench_function(BenchmarkId::new("ring_reuse", name), |b| {
            b.iter(|| {
                canny_into(image.view(), options, output.view_mut(), &mut workspace).unwrap();
                black_box(output.as_slice());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_canny, benchmark_canny_document_lines);
criterion_main!(benches);

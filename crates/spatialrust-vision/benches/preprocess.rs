use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    letterbox, pack_chw, pack_chw_into, rgb_to_gray, rgb_to_gray_into, Interpolation,
};

fn benchmark_preprocess(c: &mut Criterion) {
    let input = Image::<u8, 3>::try_new(1280, 720, vec![127; 1280 * 720 * 3]).unwrap();
    c.bench_function("letterbox_normalize_chw_1280x720_to_640", |b| {
        b.iter(|| {
            let (resized, _) =
                letterbox(black_box(input.view()), 640, 640, Interpolation::Bilinear, [114; 3])
                    .unwrap();
            pack_chw(resized.view(), 1.0 / 255.0, [0.0; 3], [1.0; 3]).unwrap()
        });
    });
}

fn benchmark_reusable_preprocess(c: &mut Criterion) {
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let input = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        let mut gray = Image::<u8, 1>::try_new(width, height, vec![0; width * height]).unwrap();
        let mut chw = vec![0.0_f32; width * height * 3];
        let throughput = Throughput::Elements((width * height) as u64);

        let mut gray_group = c.benchmark_group("rgb_to_gray_rgb8");
        gray_group.sample_size(10);
        gray_group.throughput(throughput.clone());
        gray_group.bench_function(BenchmarkId::new("allocate", name), |b| {
            b.iter(|| rgb_to_gray(black_box(input.view())).unwrap());
        });
        gray_group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| rgb_to_gray_into(black_box(input.view()), gray.view_mut()).unwrap());
        });
        gray_group.finish();

        let mut chw_group = c.benchmark_group("pack_chw_rgb8");
        chw_group.sample_size(10);
        chw_group.throughput(throughput);
        chw_group.bench_function(BenchmarkId::new("allocate", name), |b| {
            b.iter(|| pack_chw(black_box(input.view()), 1.0 / 255.0, [0.0; 3], [1.0; 3]).unwrap());
        });
        chw_group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| {
                pack_chw_into(black_box(input.view()), 1.0 / 255.0, [0.0; 3], [1.0; 3], &mut chw)
                    .unwrap()
            });
        });
        chw_group.finish();
    }
}

criterion_group!(benches, benchmark_preprocess, benchmark_reusable_preprocess);
criterion_main!(benches);

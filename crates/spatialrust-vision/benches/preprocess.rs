use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_image::Image;
use spatialrust_vision::{letterbox, pack_chw, Interpolation};

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

criterion_group!(benches, benchmark_preprocess);
criterion_main!(benches);

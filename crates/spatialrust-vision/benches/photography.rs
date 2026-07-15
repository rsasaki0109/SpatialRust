use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_image::Image;
use spatialrust_vision::{fuse_exposures, ExposureFusionOptions};

fn benchmark_photography(c: &mut Criterion) {
    let images = [48_u8, 128, 220].map(|value| Image::from_pixel(640, 480, [value; 3]).unwrap());
    c.bench_function("exposure_fusion_vga_3", |b| {
        b.iter(|| {
            black_box(
                fuse_exposures(
                    &[images[0].view(), images[1].view(), images[2].view()],
                    ExposureFusionOptions::default(),
                )
                .unwrap(),
            )
        })
    });
}

criterion_group!(benches, benchmark_photography);
criterion_main!(benches);

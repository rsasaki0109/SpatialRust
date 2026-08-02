use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_image_io::{
    decode_bytes, encode_bytes, DecodedPixels, EncodeOptions, ImageFileFormat,
};

fn encoded_rgb(width: usize, height: usize, format: ImageFileFormat) -> Vec<u8> {
    let data =
        (0..width * height * 3).map(|index| ((index * 31 + index / 7) & 0xff) as u8).collect();
    let image = Image::try_new(width, height, data).expect("valid benchmark image");
    encode_bytes(&DecodedPixels::Rgb8(image), EncodeOptions::new(format))
        .expect("benchmark fixture encodes")
}

fn benchmark_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_rgb8");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        for format in [ImageFileFormat::Png, ImageFileFormat::Jpeg] {
            let encoded = encoded_rgb(width, height, format);
            group.throughput(Throughput::Elements((width * height) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{format}/{name}"), encoded.len()),
                &encoded,
                |b, bytes| {
                    b.iter(|| decode_bytes(black_box(bytes), Default::default()).unwrap());
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_decode);
criterion_main!(benches);

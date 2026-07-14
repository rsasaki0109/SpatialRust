use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::{Image, ImageView};
use spatialrust_tensor::{interleaved_image_view, pack_interleaved_image};

fn benchmark_image_bridge(c: &mut Criterion) {
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let packed = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        let row_stride = width * 3 + 16;
        let padded = vec![127_u8; row_stride * height];
        let strided = ImageView::<u8, 3>::new(width, height, row_stride, &padded).unwrap();

        let mut zero_copy = c.benchmark_group("tensor_image_zero_copy");
        zero_copy.throughput(Throughput::Elements((width * height) as u64));
        zero_copy.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(interleaved_image_view(black_box(&packed)).unwrap()));
        });
        zero_copy.finish();

        let mut packing = c.benchmark_group("tensor_image_pack_strided");
        packing.sample_size(10);
        packing.throughput(Throughput::Bytes((width * height * 3) as u64));
        packing.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(pack_interleaved_image(black_box(strided)).unwrap()));
        });
        packing.finish();
    }
}

criterion_group!(benches, benchmark_image_bridge);
criterion_main!(benches);

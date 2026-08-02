use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{dense_flow_block_match, DenseFlowOptions};

fn benchmark_video(c: &mut Criterion) {
    for &(name, width, height) in &[("qqvga", 160, 120), ("qvga", 320, 240)] {
        let previous = Image::<u8, 1>::try_new(
            width,
            height,
            (0..width * height).map(|index| ((index * 37) % 251) as u8).collect(),
        )
        .unwrap();
        let next = previous.clone();
        let mut group = c.benchmark_group("dense_flow_block_match");
        group.sample_size(10);
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                dense_flow_block_match(
                    black_box(previous.view()),
                    black_box(next.view()),
                    DenseFlowOptions::default(),
                )
                .unwrap()
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark_video);
criterion_main!(benches);

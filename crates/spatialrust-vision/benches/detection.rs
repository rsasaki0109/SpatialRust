use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_vision::{nms, BoundingBox2};

fn benchmark_nms(c: &mut Criterion) {
    let mut group = c.benchmark_group("nms_xyxy_f32");
    group.sample_size(10);
    for &count in &[100_usize, 1_000, 8_400] {
        let (boxes, scores) = detections(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(BenchmarkId::from_parameter(count), |b| {
            b.iter(|| {
                black_box(
                    nms(black_box(&boxes), black_box(&scores), black_box(0.25), black_box(0.5))
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn detections(count: usize) -> (Vec<BoundingBox2>, Vec<f32>) {
    let mut state = 115_u64;
    let mut boxes = Vec::with_capacity(count);
    let mut scores = Vec::with_capacity(count);
    for _ in 0..count {
        let center_x = sample(&mut state) * 640.0;
        let center_y = sample(&mut state) * 640.0;
        let width = 5.0 + sample(&mut state) * 115.0;
        let height = 5.0 + sample(&mut state) * 115.0;
        boxes.push(
            BoundingBox2::try_new(
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            )
            .unwrap(),
        );
        scores.push(sample(&mut state));
    }
    (boxes, scores)
}

fn sample(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 40) as f32 / (1_u32 << 24) as f32
}

criterion_group!(benches, benchmark_nms);
criterion_main!(benches);

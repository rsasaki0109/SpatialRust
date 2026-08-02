use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_vision::{batched_nms, nms, soft_nms, BoundingBox2, Detection, SoftNmsMethod};
use std::cmp::Ordering;

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

    let mut group = c.benchmark_group("batched_nms_xyxy_f32_80_classes");
    group.sample_size(10);
    for &count in &[1_000_usize, 8_400] {
        let (boxes, scores) = detections(count);
        let detections = boxes
            .into_iter()
            .zip(scores)
            .enumerate()
            .map(|(index, (bbox, score))| Detection { bbox, score, class_id: (index % 80) as i64 })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(BenchmarkId::from_parameter(count), |b| {
            b.iter(|| {
                black_box(
                    batched_nms(black_box(&detections), black_box(0.25), black_box(0.5)).unwrap(),
                )
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("soft_nms_linear_xyxy_f32");
    group.sample_size(10);
    for &count in &[100_usize, 1_000, 8_400] {
        let (boxes, scores) = detections(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(BenchmarkId::from_parameter(count), |b| {
            b.iter(|| {
                black_box(
                    soft_nms(
                        black_box(&boxes),
                        black_box(&scores),
                        black_box(0.25),
                        black_box(0.5),
                        black_box(SoftNmsMethod::Linear),
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_function(BenchmarkId::new("sorting_baseline", count), |b| {
            b.iter(|| {
                black_box(soft_nms_sorting_baseline(
                    black_box(&boxes),
                    black_box(&scores),
                    black_box(0.25),
                    black_box(0.5),
                ))
            });
        });
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct BaselineCandidate {
    index: usize,
    score: f32,
}

fn soft_nms_sorting_baseline(
    boxes: &[BoundingBox2],
    scores: &[f32],
    score_threshold: f32,
    iou_threshold: f32,
) -> Vec<BaselineCandidate> {
    let mut candidates = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(index, score)| BaselineCandidate { index, score })
        .collect::<Vec<_>>();
    let areas = boxes.iter().copied().map(BoundingBox2::area).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(candidates.len());
    while !candidates.is_empty() {
        candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.index.cmp(&right.index))
        });
        let selected = candidates.remove(0);
        if selected.score < score_threshold {
            break;
        }
        output.push(selected);
        for candidate in &mut candidates {
            let overlap = cached_iou(
                boxes[selected.index],
                areas[selected.index],
                boxes[candidate.index],
                areas[candidate.index],
            );
            if overlap > iou_threshold {
                candidate.score *= 1.0 - overlap;
            }
        }
        candidates.retain(|candidate| candidate.score >= score_threshold);
    }
    output
}

fn cached_iou(left: BoundingBox2, left_area: f32, right: BoundingBox2, right_area: f32) -> f32 {
    let width = left.x_max.min(right.x_max) - left.x_min.max(right.x_min);
    if width <= 0.0 {
        return 0.0;
    }
    let height = left.y_max.min(right.y_max) - left.y_min.max(right.y_min);
    if height <= 0.0 {
        return 0.0;
    }
    let intersection = width * height;
    intersection / (left_area + right_area - intersection)
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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_vision::{select_keypoints_grid, GridSelectionOptions, Keypoint2};

fn benchmark_odometry_frontend(c: &mut Criterion) {
    let keypoints = (0..4096)
        .map(|index| {
            Keypoint2::try_new(
                (index % 128) as f32 * 15.0,
                (index / 128) as f32 * 15.0,
                (index % 97) as f32,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    c.bench_function("grid_select_4096", |b| {
        b.iter(|| {
            black_box(
                select_keypoints_grid(&keypoints, 1920, 1080, GridSelectionOptions::default())
                    .unwrap(),
            )
        });
    });
}

criterion_group!(benches, benchmark_odometry_frontend);
criterion_main!(benches);

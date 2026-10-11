#[path = "../examples/graph_common/mod.rs"]
mod common;
use common::{fixture, run, Kind, Mode};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn neighbor_graphs(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_graphs");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    for points in [4096usize, 65536] {
        for duplicates in [false, true] {
            let cloud = fixture(points, duplicates);
            group.throughput(Throughput::Elements(points as u64));
            for kind in [Kind::Knn8, Kind::Knn32, Kind::Radius] {
                let expected = run(&cloud, kind, Mode::Previous);
                for mode in [Mode::Previous, Mode::Serial, Mode::Parallel] {
                    assert_eq!(run(&cloud, kind, mode), expected);
                    group.bench_function(
                        BenchmarkId::new(
                            format!("{kind:?}/{mode:?}"),
                            format!("{points}_duplicates{duplicates}"),
                        ),
                        |b| {
                            b.iter_custom(|iterations| {
                                let mut elapsed = Duration::ZERO;
                                for _ in 0..iterations {
                                    let start = Instant::now();
                                    let actual = run(&cloud, kind, mode);
                                    elapsed += start.elapsed();
                                    black_box(&actual);
                                }
                                elapsed
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}
criterion_group!(benches, neighbor_graphs);
criterion_main!(benches);

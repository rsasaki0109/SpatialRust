#[path = "../examples/common/mod.rs"]
mod common;

use common::{Fixture, Kind, Mode};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn chunked_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_queries");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    for (points, chunk_size) in [(16_384usize, 256usize), (65_536, 1024), (65_536, 1)] {
        let fixture = Fixture::new(points, chunk_size);
        let query = fixture.query();
        group.throughput(Throughput::Elements(points as u64));
        for kind in [Kind::Radius, Kind::Knn] {
            let expected = query.run(kind, Mode::Serial);
            let modes: &[Mode] = if points.div_ceil(chunk_size) <= 256 {
                &[Mode::Serial, Mode::Bounded, Mode::Legacy]
            } else {
                &[Mode::Serial, Mode::Bounded]
            };
            for &mode in modes {
                assert!(query.run(kind, mode) == expected, "result/order mismatch");
                group.bench_function(
                    BenchmarkId::new(
                        format!("{kind:?}/{mode:?}"),
                        format!("{points}_chunk{chunk_size}"),
                    ),
                    |b| {
                        b.iter_custom(|iterations| {
                            let mut elapsed = Duration::ZERO;
                            for _ in 0..iterations {
                                let start = Instant::now();
                                let out = query.run(kind, mode);
                                elapsed += start.elapsed();
                                black_box(&out);
                            }
                            elapsed
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, chunked_queries);
criterion_main!(benches);

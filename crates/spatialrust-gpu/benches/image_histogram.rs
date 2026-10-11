#[path = "../examples/histogram_common/mod.rs"]
mod common;

use common::{Fixture, MODES};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::WgpuRuntime;
use std::{hint::black_box, time::Duration};

fn histograms(c: &mut Criterion) {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!(
        "adapter={:?}, CPU histogram pool={} workers",
        runtime.adapter_info(),
        rayon::current_num_threads()
    );
    let mut group = c.benchmark_group("image_histogram");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    for (width, height) in [(640, 480), (1920, 1080), (3840, 2160)] {
        for hot in [false, true] {
            let mut fixture = Fixture::new(&runtime, width, height, hot);
            group.throughput(Throughput::Elements((width * height) as u64));
            for mode in MODES {
                group.bench_function(
                    BenchmarkId::new(
                        format!("{mode:?}"),
                        format!("{width}x{height}_{}", if hot { "hot" } else { "mixed" }),
                    ),
                    |b| {
                        b.iter_custom(|iterations| {
                            let mut elapsed = Duration::ZERO;
                            for _ in 0..iterations {
                                let sample = fixture.sample(&runtime, mode);
                                elapsed += sample.elapsed;
                                black_box((
                                    sample.host_to_device_bytes,
                                    sample.device_to_host_bytes,
                                ));
                            }
                            elapsed
                        });
                    },
                );
            }
            fixture.recycle(&runtime);
        }
    }
    group.finish();
}

criterion_group!(benches, histograms);
criterion_main!(benches);

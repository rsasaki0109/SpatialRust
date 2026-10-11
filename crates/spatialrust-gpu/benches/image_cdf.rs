#[path = "../examples/cdf_common/mod.rs"]
mod common;

use common::{Fixture, MODES};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::WgpuRuntime;
use std::{hint::black_box, time::Duration};

fn cdfs(c: &mut Criterion) {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!("adapter={:?}", runtime.adapter_info());
    let mut group = c.benchmark_group("image_cdf");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(256));
    for hot in [false, true] {
        let mut fixture = Fixture::new(&runtime, 3840, 2160, hot);
        // CDF has exactly 256 input bins regardless of original image size.
        // Image-chain placement is separately covered by the rotating example.
        for mode in MODES.into_iter().take(4) {
            group.bench_function(
                BenchmarkId::new(format!("{mode:?}"), if hot { "hot" } else { "mixed" }),
                |b| {
                    b.iter_custom(|iterations| {
                        let mut elapsed = Duration::ZERO;
                        for _ in 0..iterations {
                            let sample = fixture.sample(&runtime, mode);
                            elapsed += sample.elapsed;
                            black_box((sample.host_to_device_bytes, sample.device_to_host_bytes));
                        }
                        elapsed
                    })
                },
            );
        }
        fixture.recycle(&runtime);
    }
    group.finish();
}

criterion_group!(benches, cdfs);
criterion_main!(benches);

#[path = "../examples/clahe_common/mod.rs"]
mod common;

use common::{Fixture, Mode};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::WgpuRuntime;
use std::{hint::black_box, time::Duration};

fn adaptive_contrast(c: &mut Criterion) {
    let runtime = WgpuRuntime::new_headless().unwrap();
    eprintln!(
        "adapter={:?}, CPU pool={} workers",
        runtime.adapter_info(),
        rayon::current_num_threads()
    );
    let mut group = c.benchmark_group("image_clahe");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    // All Criterion modes end on the CPU, so byte verification adds no untimed
    // GPU transfer. The rotating example separately measures resident outputs.
    black_box(common::MODES);
    for (width, height) in [(640, 480), (3840, 2160)] {
        for hot in [false, true] {
            let mut fixture = Fixture::new(&runtime, width, height, hot);
            group.throughput(Throughput::Elements((width * height) as u64));
            for mode in
                [Mode::CpuHost, Mode::GpuReadback, Mode::UploadRoundTrip, Mode::GpuInputCpuOutput]
            {
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
                        })
                    },
                );
            }
            fixture.recycle(&runtime);
        }
    }
    group.finish();
}
criterion_group!(benches, adaptive_contrast);
criterion_main!(benches);

use std::{hint::black_box, sync::Arc, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_ai::{
    CopyPolicy, InferenceBackend, IoBinding, ModelSource, NamedTensors, OnnxRuntimeBackend,
    OutputBinding, RunOptions, SessionOptions,
};
use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

const DOUBLE_DYNAMIC: &[u8] = &[
    8, 8, 18, 16, 115, 112, 97, 116, 105, 97, 108, 114, 117, 115, 116, 45, 116, 101, 115, 116, 58,
    106, 10, 27, 10, 5, 105, 110, 112, 117, 116, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111, 117,
    116, 112, 117, 116, 34, 3, 65, 100, 100, 18, 14, 100, 111, 117, 98, 108, 101, 95, 100, 121,
    110, 97, 109, 105, 99, 90, 28, 10, 5, 105, 110, 112, 117, 116, 18, 19, 10, 17, 8, 1, 18, 13,
    10, 7, 18, 5, 98, 97, 116, 99, 104, 10, 2, 8, 3, 98, 29, 10, 6, 111, 117, 116, 112, 117, 116,
    18, 19, 10, 17, 8, 1, 18, 13, 10, 7, 18, 5, 98, 97, 116, 99, 104, 10, 2, 8, 3, 66, 4, 10, 0,
    16, 13,
];

fn benchmark_onnxruntime(c: &mut Criterion) {
    let backend = OnnxRuntimeBackend;
    let mut session = backend
        .create_session(&ModelSource::Bytes(Arc::from(DOUBLE_DYNAMIC)), &SessionOptions::default())
        .expect("embedded model");
    let mut group = c.benchmark_group("onnxruntime_cpu_dynamic_rgb_f32");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for (label, width, height) in
        [("640p", 640_usize, 480_usize), ("1080p", 1920, 1080), ("4k", 3840, 2160)]
    {
        let pixels = width * height;
        let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![pixels, 3], Device::CPU);
        let input = TensorBuffer::try_from_f32(vec![0.5; pixels * 3], descriptor).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("input", input).unwrap();
        group.throughput(Throughput::Bytes((pixels * 3 * 4 * 2) as u64));

        group.bench_with_input(BenchmarkId::new("copy_run", label), &inputs, |b, inputs| {
            b.iter(|| {
                black_box(
                    session
                        .run_with_options(
                            inputs.clone(),
                            RunOptions {
                                input_copy: CopyPolicy::Allow,
                                output_copy: CopyPolicy::Allow,
                            },
                        )
                        .unwrap(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("io_binding", label), &inputs, |b, inputs| {
            b.iter(|| {
                let mut binding = IoBinding::try_new(
                    inputs.clone(),
                    vec![OutputBinding::Allocate { name: "output".into(), device: Device::CPU }],
                )
                .unwrap();
                session.run_with_binding(&mut binding).unwrap();
                black_box(binding.into_results())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_onnxruntime);
criterion_main!(benches);

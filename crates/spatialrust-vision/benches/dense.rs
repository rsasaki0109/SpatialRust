use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_vision::{
    connected_components, distance_transform_edt, distance_transform_edt_into, BinaryMask,
    Connectivity, DistanceTransformWorkspace,
};

fn benchmark_exact_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform_edt");
    group.sample_size(10);
    for &(name, width, height) in &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height)
            .map(|index| u8::from(index % 97 != 0 && index % (width * 11 + 1) != 0))
            .collect();
        let mask = BinaryMask::try_new(width, height, data).unwrap();
        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(distance_transform_edt(black_box(&mask)).unwrap()));
        });
        let mut output = vec![0.0_f32; width * height];
        let mut workspace = DistanceTransformWorkspace::new();
        distance_transform_edt_into(&mask, &mut output, &mut workspace).unwrap();
        group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| {
                distance_transform_edt_into(
                    black_box(&mask),
                    black_box(&mut output),
                    black_box(&mut workspace),
                )
                .unwrap()
            });
        });

        let mut state = 103_u64;
        let random_data = (0..width * height)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut sample = state;
                sample = (sample ^ (sample >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                sample = (sample ^ (sample >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                sample ^= sample >> 31;
                // The OpenCV harness's seeded BGR-to-gray mask is about 29.4%
                // background at its `gray > 96` threshold.
                u8::from((sample >> 56) > 74)
            })
            .collect();
        let random_mask = BinaryMask::try_new(width, height, random_data).unwrap();
        let mut random_output = vec![0.0_f32; width * height];
        let mut random_workspace = DistanceTransformWorkspace::new();
        distance_transform_edt_into(&random_mask, &mut random_output, &mut random_workspace)
            .unwrap();
        group.bench_function(BenchmarkId::new("opencv-density-reuse", name), |b| {
            b.iter(|| {
                distance_transform_edt_into(
                    black_box(&random_mask),
                    black_box(&mut random_output),
                    black_box(&mut random_workspace),
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

fn benchmark_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components_8_structured");
    group.sample_size(10);
    for &(profile, width, height) in &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)]
    {
        for pattern in ["segmentation-blobs", "document-lines"] {
            let data = (0..width * height)
                .map(|index| {
                    let (x, y) = (index % width, index / width);
                    match pattern {
                        "segmentation-blobs" => u8::from(x % 97 < 23 && y % 83 < 19),
                        "document-lines" => u8::from(y % 32 < 3 && x % 211 > 8 && x % 211 < 190),
                        _ => unreachable!(),
                    }
                })
                .collect();
            let mask = BinaryMask::try_new(width, height, data).unwrap();
            group.throughput(Throughput::Elements((width * height) as u64));
            group.bench_function(BenchmarkId::new(pattern, profile), |b| {
                b.iter(|| {
                    black_box(connected_components(black_box(&mask), Connectivity::Eight).unwrap())
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_exact_distance_transform, benchmark_connected_components);
criterion_main!(benches);

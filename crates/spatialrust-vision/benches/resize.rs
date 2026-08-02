use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_image::Image;
use spatialrust_vision::{
    resize, resize_into, AreaResizeU8Plan, BilinearResizeU8Plan, Interpolation, NearestResizeU8Plan,
};

fn benchmark_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_bilinear_rgb8");
    group.sample_size(10);
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let input = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
        let output_width = width / 2;
        let output_height = height / 2;
        let mut output = Image::<u8, 3>::try_new(
            output_width,
            output_height,
            vec![0; output_width * output_height * 3],
        )
        .unwrap();
        let plan = BilinearResizeU8Plan::new(width, height, output_width, output_height).unwrap();
        group.throughput(Throughput::Elements((output_width * output_height) as u64));
        group.bench_function(BenchmarkId::new("allocate", name), |b| {
            b.iter(|| {
                resize(
                    black_box(input.view()),
                    output_width,
                    output_height,
                    Interpolation::Bilinear,
                )
                .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("reuse", name), |b| {
            b.iter(|| {
                resize_into(black_box(input.view()), output.view_mut(), Interpolation::Bilinear)
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("planned_allocate", name), |b| {
            b.iter(|| plan.resize(black_box(input.view())).unwrap());
        });
        group.bench_function(BenchmarkId::new("planned_reuse", name), |b| {
            b.iter(|| plan.resize_into(black_box(input.view()), output.view_mut()).unwrap());
        });
    }
    group.finish();
}

fn benchmark_nearest_and_area(c: &mut Criterion) {
    for &(filter_name, interpolation) in
        &[("nearest", Interpolation::Nearest), ("area", Interpolation::Area)]
    {
        let mut group = c.benchmark_group(format!("resize_{filter_name}_rgb8"));
        group.sample_size(10);
        for &(name, width, height) in
            &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)]
        {
            let input =
                Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
            let output_width = width / 2;
            let output_height = height / 2;
            let mut output = Image::<u8, 3>::try_new(
                output_width,
                output_height,
                vec![0; output_width * output_height * 3],
            )
            .unwrap();
            group.throughput(Throughput::Elements((output_width * output_height) as u64));
            group.bench_function(BenchmarkId::new("generic_reuse", name), |b| {
                b.iter(|| {
                    resize_into(
                        black_box(input.view()),
                        output.view_mut(),
                        black_box(interpolation),
                    )
                    .unwrap()
                });
            });
            if interpolation == Interpolation::Nearest {
                let plan =
                    NearestResizeU8Plan::new(width, height, output_width, output_height).unwrap();
                group.bench_function(BenchmarkId::new("planned_reuse", name), |b| {
                    b.iter(|| {
                        plan.resize_into(black_box(input.view()), output.view_mut()).unwrap()
                    });
                });
            } else {
                let plan =
                    AreaResizeU8Plan::new(width, height, output_width, output_height).unwrap();
                group.bench_function(BenchmarkId::new("planned_reuse", name), |b| {
                    b.iter(|| {
                        plan.resize_into(black_box(input.view()), output.view_mut()).unwrap()
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_resize, benchmark_nearest_and_area);
criterion_main!(benches);

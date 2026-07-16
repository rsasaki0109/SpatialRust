use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::{run_gpu_vision_chain, GpuImage, GpuVisionChainOptions, WgpuRuntime};
use spatialrust_image::Image;

const PROFILES: &[(&str, usize, usize)] =
    &[("vga", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)];

fn clamp(value: isize, upper: usize) -> usize {
    value.clamp(0, upper.saturating_sub(1) as isize) as usize
}

fn cpu_reference_chain(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let out_width = width / 2;
    let out_height = height / 2;
    let mut gray = vec![0_u8; out_width * out_height];
    for y in 0..out_height {
        for x in 0..out_width {
            let source = ((y * 2) * width + x * 2) * 3;
            let r = u32::from(rgb[source]);
            let g = u32::from(rgb[source + 1]);
            let b = u32::from(rgb[source + 2]);
            gray[y * out_width + x] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
        }
    }

    let mut blurred = vec![0_u8; gray.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let mut sum = 0_u32;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    sum += u32::from(
                        gray[clamp(y as isize + ky, out_height) * out_width
                            + clamp(x as isize + kx, out_width)],
                    );
                }
            }
            blurred[y * out_width + x] = ((sum + 4) / 9) as u8;
        }
    }

    let mut edges = vec![0_u8; gray.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let sample = |dx: isize, dy: isize| {
                i32::from(
                    blurred[clamp(y as isize + dy, out_height) * out_width
                        + clamp(x as isize + dx, out_width)],
                )
            };
            let gx = -sample(-1, -1) + sample(1, -1) - 2 * sample(-1, 0) + 2 * sample(1, 0)
                - sample(-1, 1)
                + sample(1, 1);
            let gy = -sample(-1, -1) - 2 * sample(0, -1) - sample(1, -1)
                + sample(-1, 1)
                + 2 * sample(0, 1)
                + sample(1, 1);
            edges[y * out_width + x] = (gx.abs() + gy.abs()).min(255) as u8;
        }
    }

    let mut dilated = vec![0_u8; gray.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let mut maximum = 0_u8;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    maximum = maximum.max(
                        edges[clamp(y as isize + ky, out_height) * out_width
                            + clamp(x as isize + kx, out_width)],
                    );
                }
            }
            dilated[y * out_width + x] = maximum;
        }
    }
    dilated.into_iter().map(|value| f32::from(value) / 255.0).collect()
}

fn options(width: usize, height: usize) -> GpuVisionChainOptions {
    GpuVisionChainOptions {
        width: (width / 2) as u32,
        height: (height / 2) as u32,
        ..Default::default()
    }
}

fn benchmark_gpu_image(c: &mut Criterion) {
    let Ok(runtime) = WgpuRuntime::new_headless() else {
        return;
    };
    eprintln!("wgpu adapter: {:?}", runtime.adapter_info());
    for &(name, width, height) in PROFILES {
        let data: Vec<u8> =
            (0..width * height * 3).map(|index| ((index * 37) % 200 + 20) as u8).collect();
        let image = Image::<u8, 3>::try_new(width, height, data).unwrap();
        let throughput = Throughput::Elements((width * height) as u64);

        let mut cpu = c.benchmark_group("vision2_cpu_chain");
        cpu.sample_size(10);
        cpu.throughput(throughput.clone());
        cpu.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(cpu_reference_chain(image.as_slice(), width, height)));
        });
        cpu.finish();

        let mut round_trip = c.benchmark_group("vision2_gpu_round_trip");
        round_trip.sample_size(10);
        round_trip.throughput(throughput.clone());
        round_trip.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
                let mut tensor =
                    run_gpu_vision_chain(&runtime, &uploaded, options(width, height)).unwrap();
                let values = tensor.readback_f32(&runtime).unwrap();
                black_box(values);
                tensor.recycle(&runtime);
                uploaded.recycle(&runtime);
            });
        });
        round_trip.finish();

        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        runtime.wait_idle();
        let mut resident = c.benchmark_group("vision2_gpu_resident");
        resident.sample_size(10);
        resident.throughput(throughput);
        resident.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let tensor =
                    run_gpu_vision_chain(&runtime, &uploaded, options(width, height)).unwrap();
                runtime.wait_idle();
                black_box(tensor.shape());
                tensor.recycle(&runtime);
            });
        });
        resident.finish();
        uploaded.recycle(&runtime);
    }
}

criterion_group!(benches, benchmark_gpu_image);
criterion_main!(benches);

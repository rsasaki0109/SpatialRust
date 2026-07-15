use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_gpu::{box_blur_gpu, rgb_to_gray_gpu, GpuImage, GpuImageBorder, WgpuRuntime};
use spatialrust_image::Image;

fn benchmark_gpu_image(c: &mut Criterion) {
    let Ok(runtime) = WgpuRuntime::new_headless() else {
        return;
    };
    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let data = (0..width * height * 3).map(|index| ((index * 37) % 200 + 20) as u8).collect();
        let image = Image::<u8, 3>::try_new(width, height, data).unwrap();

        let mut upload = c.benchmark_group("gpu_image_upload_rgb");
        upload.sample_size(10);
        upload.throughput(Throughput::Bytes((width * height * 4) as u64));
        upload.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let gpu = GpuImage::upload_u8(&runtime, image.view()).unwrap();
                runtime.wait_idle();
                black_box(gpu.width());
                gpu.recycle(&runtime);
            });
        });
        upload.finish();

        let mut chain = c.benchmark_group("gpu_image_gray_box_blur");
        chain.sample_size(10);
        chain.throughput(Throughput::Elements((width * height) as u64));
        chain.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
                let gray = rgb_to_gray_gpu(&runtime, &uploaded).unwrap();
                let blurred =
                    box_blur_gpu(&runtime, &gray, 5, 5, GpuImageBorder::Replicate).unwrap();
                runtime.wait_idle();
                black_box(blurred.width());
                uploaded.recycle(&runtime);
                gray.recycle(&runtime);
                blurred.recycle(&runtime);
            });
        });
        chain.finish();

        let uploaded = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let mut resident = c.benchmark_group("gpu_image_texture_resident_chain");
        resident.sample_size(10);
        resident.throughput(Throughput::Elements((width * height) as u64));
        resident.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let resized = spatialrust_gpu::resize_nearest_gpu(
                    &runtime,
                    &uploaded,
                    width as u32 / 2,
                    height as u32 / 2,
                )
                .unwrap();
                let gray = rgb_to_gray_gpu(&runtime, &resized).unwrap();
                let blurred =
                    box_blur_gpu(&runtime, &gray, 5, 5, GpuImageBorder::Replicate).unwrap();
                let edges = spatialrust_gpu::sobel_gpu(&runtime, &blurred).unwrap();
                let dilated = spatialrust_gpu::morphology_gpu(
                    &runtime,
                    &edges,
                    3,
                    3,
                    spatialrust_gpu::GpuMorphology::Dilate,
                )
                .unwrap();
                runtime.wait_idle();
                black_box(dilated.width());
                resized.recycle(&runtime);
                gray.recycle(&runtime);
                blurred.recycle(&runtime);
                edges.recycle(&runtime);
                dilated.recycle(&runtime);
            });
        });
        resident.finish();
    }
}

criterion_group!(benches, benchmark_gpu_image);
criterion_main!(benches);

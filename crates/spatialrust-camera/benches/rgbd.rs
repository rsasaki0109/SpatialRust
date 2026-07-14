use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_camera::{rgbd_to_point_cloud, CameraIntrinsics, PinholeCamera};
use spatialrust_image::Image;

fn benchmark_rgbd(c: &mut Criterion) {
    let width = 640;
    let height = 480;
    let depth = Image::<f32, 1>::try_new(width, height, vec![2.0; width * height]).unwrap();
    let color = Image::<u8, 3>::try_new(width, height, vec![127; width * height * 3]).unwrap();
    let camera = PinholeCamera::new(
        CameraIntrinsics::try_new(525.0, 525.0, 319.5, 239.5, width, height).unwrap(),
    );

    c.bench_function("rgbd_to_point_cloud_640x480", |b| {
        b.iter(|| {
            rgbd_to_point_cloud(
                black_box(depth.view()),
                black_box(color.view()),
                black_box(&camera),
                Default::default(),
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, benchmark_rgbd);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_camera::CameraIntrinsics;
use spatialrust_image::Image;
use spatialrust_math::{Mat3, Vec2, Vec3};
use spatialrust_vision::{
    estimate_homography_ransac, project_object_point, solve_pnp, stereo_block_match,
    track_points_lucas_kanade, AbsolutePose, CameraMatrix3, ObjectImageCorrespondence,
    PointCorrespondence2, RobustEstimationOptions, StereoBmOptions,
};

fn benchmark_geometry(c: &mut Criterion) {
    let camera = CameraMatrix3::from_intrinsics(
        CameraIntrinsics::try_new(500.0, 500.0, 320.0, 240.0, 640, 480).unwrap(),
    );
    let pose = AbsolutePose::try_new(
        Mat3::from_rows([0.98, -0.1, 0.17], [0.12, 0.98, -0.1], [-0.16, 0.12, 0.98]),
        Vec3::new(0.1, -0.05, 2.0),
    )
    .unwrap();

    let mut pnp = c.benchmark_group("solve_pnp_correspondences");
    for &count in &[64usize, 256, 1024] {
        let pairs = (0..count)
            .map(|index| {
                let object = Vec3::new(
                    (index % 16) as f64 * 0.05 - 0.4,
                    (index / 16) as f64 * 0.05 - 0.4,
                    (index % 5) as f64 * 0.02,
                );
                let image = project_object_point(pose, camera, object).unwrap();
                ObjectImageCorrespondence::try_new(object, image).unwrap()
            })
            .collect::<Vec<_>>();
        pnp.throughput(Throughput::Elements(count as u64));
        pnp.bench_with_input(BenchmarkId::from_parameter(count), &pairs, |b, pairs| {
            b.iter(|| black_box(solve_pnp(pairs, camera).unwrap()));
        });
    }
    pnp.finish();

    let mut homography = c.benchmark_group("homography_ransac_correspondences");
    for &count in &[64usize, 256, 1024] {
        let pairs = (0..count)
            .map(|index| {
                let source = Vec2 { x: (index % 32) as f64 * 10.0, y: (index / 32) as f64 * 8.0 };
                PointCorrespondence2::try_new(
                    source,
                    Vec2 { x: source.x * 1.05 + 2.0, y: source.y * 0.98 - 1.0 },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        homography.throughput(Throughput::Elements(count as u64));
        homography.bench_with_input(BenchmarkId::from_parameter(count), &pairs, |b, pairs| {
            b.iter(|| {
                black_box(
                    estimate_homography_ransac(pairs, RobustEstimationOptions::default()).unwrap(),
                )
            });
        });
    }
    homography.finish();

    for &(name, width, height) in &[("640p", 640, 480), ("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let left_data = (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                ((x * 17 + y * 29) % 200 + 20) as u8
            })
            .collect::<Vec<_>>();
        let mut right_data = vec![0u8; width * height];
        let disparity = 20usize;
        for y in 0..height {
            for x in disparity..width {
                right_data[y * width + (x - disparity)] = left_data[y * width + x];
            }
        }
        let left = Image::<u8, 1>::try_new(width, height, left_data.clone()).unwrap();
        let right = Image::<u8, 1>::try_new(width, height, right_data).unwrap();

        let mut bm = c.benchmark_group("stereo_bm");
        bm.sample_size(10);
        bm.throughput(Throughput::Elements((width * height) as u64));
        bm.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(
                    stereo_block_match(
                        left.view(),
                        right.view(),
                        StereoBmOptions {
                            window_size: 15,
                            min_disparity: 1,
                            num_disparities: 64,
                            uniqueness_ratio: 10.0,
                        },
                    )
                    .unwrap(),
                )
            });
        });
        bm.finish();

        let points = (0..200)
            .map(|index| Vec2 {
                x: 40.0 + (index % 20) as f64 * 20.0,
                y: 40.0 + (index / 20) as f64 * 20.0,
            })
            .collect::<Vec<_>>();
        let next = Image::<u8, 1>::try_new(width, height, left_data).unwrap();
        let mut lk = c.benchmark_group("lucas_kanade_200");
        lk.sample_size(10);
        lk.throughput(Throughput::Elements(points.len() as u64));
        lk.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(
                    track_points_lucas_kanade(
                        left.view(),
                        next.view(),
                        &points,
                        Default::default(),
                    )
                    .unwrap(),
                )
            });
        });
        lk.finish();
    }
}

criterion_group!(benches, benchmark_geometry);
criterion_main!(benches);

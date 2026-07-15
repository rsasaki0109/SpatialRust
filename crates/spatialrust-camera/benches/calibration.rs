use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use spatialrust_camera::{
    bundle_adjust_points, calibrate_pinhole, BundleObservation, BundleProblem, BundleView,
    CalibrationOptions, CameraIntrinsics, PinholeCamera, PinholeObservation, RigidTransform3,
};
use spatialrust_math::{Mat3, Vec3};

fn benchmark_calibration(c: &mut Criterion) {
    let camera = PinholeCamera::new(
        CameraIntrinsics::try_new(800.0, 805.0, 640.0, 360.0, 1280, 720).unwrap(),
    );
    for &count in &[100_usize, 1_000] {
        let observations = (0..count)
            .map(|index| {
                let x = index % 31;
                let y = (index / 31) % 23;
                let point = Vec3::new(
                    x as f64 * 0.02 - 0.3,
                    y as f64 * 0.02 - 0.2,
                    2.0 + index as f64 * 1e-4,
                );
                PinholeObservation { camera_point: point, pixel: camera.project(point).unwrap() }
            })
            .collect::<Vec<_>>();
        let mut group = c.benchmark_group("calibrate_pinhole");
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &observations, |b, input| {
            b.iter(|| {
                calibrate_pinhole(black_box(input), 1280, 720, CalibrationOptions::default())
                    .unwrap()
            });
        });
        group.finish();
    }

    let views = [0.0, -0.3, 0.3]
        .into_iter()
        .map(|x| BundleView {
            camera,
            camera_from_world: RigidTransform3 {
                rotation: Mat3::<f64>::identity(),
                translation: Vec3::new(x, 0.0, 0.0),
            },
        })
        .collect::<Vec<_>>();
    let truth = (0..100)
        .map(|index| Vec3::new(index as f64 * 0.005 - 0.25, (index % 13) as f64 * 0.01 - 0.06, 2.5))
        .collect::<Vec<_>>();
    let observations = truth
        .iter()
        .enumerate()
        .flat_map(|(point_index, &point)| {
            views.iter().enumerate().map(move |(view_index, view)| {
                let camera_point = view.camera_from_world.transform_point(point);
                BundleObservation {
                    view_index,
                    point_index,
                    pixel: view.camera.project(camera_point).unwrap(),
                }
            })
        })
        .collect::<Vec<_>>();
    c.bench_function("bundle_adjust_fixed_cameras/100_points_3_views", |b| {
        b.iter_batched(
            || BundleProblem {
                views: views.clone(),
                points: truth.iter().map(|point| *point + Vec3::new(0.02, -0.01, 0.05)).collect(),
                observations: observations.clone(),
            },
            |mut problem| {
                bundle_adjust_points(black_box(&mut problem), CalibrationOptions::default())
                    .unwrap()
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, benchmark_calibration);
criterion_main!(benches);

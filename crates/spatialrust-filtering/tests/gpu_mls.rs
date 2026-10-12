#![cfg(all(feature = "filter-mls-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{
    DType, FieldSemantic, HasPositions3, PointBuffer, PointBufferSet, PointCloud, PointField,
    PointSchema, SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_filtering::{GpuMlsPositionResult, MlsConfig, MlsSmoothing};
use spatialrust_gpu::{GpuMlsInput, GpuMlsMode, WgpuRuntime};
use spatialrust_math::Vec3;

pub fn fixture(n: usize, kind: usize) -> PointCloud {
    fixture_with_width(n, kind, if kind == 4 { 128 } else { 32 })
}
pub fn fixture_with_width(n: usize, kind: usize, width: usize) -> PointCloud {
    let mut columns: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % width) as f32 * 0.05 - 0.8;
        let y = (j / width) as f32 * 0.05 - 0.4;
        let noise = ((j * 37 % 101) as f32 / 100.0 - 0.5) * 0.02;
        let z = match kind {
            0 => noise,
            1 | 2 | 4 => 0.015 * (x * x + y * y) + noise,
            _ => 0.0,
        };
        for (column, value) in columns.iter_mut().zip([x, y, z]) {
            column.push(value);
        }
    }
    from_columns(columns)
}
pub fn from_columns(columns: [Vec<f32>; 3]) -> PointCloud {
    let mut buffers = PointBufferSet::new();
    for (name, values) in ["x", "y", "z"].into_iter().zip(columns) {
        buffers.insert(name, PointBuffer::F32(values));
    }
    PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, Default::default()).unwrap()
}
pub fn compare(actual: &[Vec3<f32>], expected: &[Vec3<f32>], radius: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        for (axis, (a, b)) in [a.x, a.y, a.z].into_iter().zip([b.x, b.y, b.z]).enumerate() {
            assert!(a.is_finite());
            let error = (a - b).abs();
            maximum = maximum.max(error);
            let ulp = f32::from_bits(b.abs().to_bits() + 1) - b.abs();
            let bound = 5e-5 + radius * 1e-3 + 8.0 * ulp;
            assert!(
                error <= bound,
                "point={i} axis={axis} gpu={a} cpu={b} error={error} bound={bound}"
            );
        }
    }
    maximum
}

// Validate the coordinates of each reported outcome against independent CPU
// math for that outcome; retain error against the requested CPU order separately.
pub fn audit(
    cloud: &PointCloud,
    config: MlsConfig,
    actual: &GpuMlsPositionResult,
    requested: &[Vec3<f32>],
) -> (f32, f32) {
    let plane = MlsSmoothing::new(MlsConfig { polynomial_order: 0, ..config })
        .smoothed_positions(cloud)
        .unwrap();
    let (x, y, z) = cloud.positions3().unwrap();
    let mut reference = Vec::with_capacity(cloud.len());
    let mut requested_error = 0.0_f32;
    for (i, mode) in actual.modes.iter().enumerate() {
        reference.push(match mode {
            GpuMlsMode::Unchanged => Vec3::new(x[i], y[i], z[i]),
            GpuMlsMode::PlaneFallback => plane[i],
            GpuMlsMode::Polynomial => requested[i],
        });
        if *mode == GpuMlsMode::Unchanged {
            let p = actual.positions[i];
            assert_eq!(
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                [x[i].to_bits(), y[i].to_bits(), z[i].to_bits()]
            );
        }
        let a = actual.positions[i];
        let b = requested[i];
        requested_error =
            requested_error.max((a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs()));
    }
    (compare(&actual.positions, &reference, config.search_radius), requested_error)
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_mls_compares_all_orders_coordinates_modes_and_receipts_with_cpu() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut coordinates = 0;
    let mut maximum = 0.0_f32;
    let mut requested_maximum = 0.0_f32;
    for n in [0, 1, 2, 17, 63, 64, 65, 257, 4097] {
        for kind in 0..5 {
            let cloud = fixture(n, kind);
            for order in 0..=2 {
                let config =
                    MlsConfig { search_radius: 0.16, polynomial_order: order, min_neighbors: 6 };
                let smoother = MlsSmoothing::new(config);
                let expected = smoother.smoothed_positions(&cloud).unwrap();
                let actual = smoother.smoothed_positions_with_gpu(&runtime, &cloud, 128).unwrap();
                let (difference, requested_error) = audit(&cloud, config, &actual, &expected);
                requested_maximum = requested_maximum.max(requested_error);
                if n == 4097 && matches!(kind, 0..=2) && order == 2 {
                    assert!(
                        actual.diagnostics.polynomial as usize >= n * 95 / 100,
                        "well-supported surfaces must use quadratic fits: {:?}",
                        actual.diagnostics
                    );
                }
                maximum = maximum.max(difference);
                coordinates += n * 3;
                assert_eq!(actual.modes.len(), n);
                for (mode, count) in [
                    (GpuMlsMode::Unchanged, actual.diagnostics.unchanged),
                    (GpuMlsMode::PlaneFallback, actual.diagnostics.plane_fallback),
                    (GpuMlsMode::Polynomial, actual.diagnostics.polynomial),
                ] {
                    assert_eq!(actual.modes.iter().filter(|&&m| m == mode).count(), count as usize);
                }
                let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
                let nodes = tree.layout().nodes().len();
                assert_eq!(
                    actual.transfers.host_to_device_bytes(),
                    if n == 0 { 0 } else { (n * 32 + nodes * 32) as u64 }
                );
                assert_eq!(
                    actual.transfers.device_to_host_bytes(),
                    if n == 0 { 0 } else { (n * 16 + 32) as u64 }
                );
                eprintln!("GPU_MLS_ACCURACY n={n} kind={kind} order={order} max_mode_reference_error={difference} max_requested_cpu_error={requested_error} unchanged={} plane={} polynomial={}",actual.diagnostics.unchanged,actual.diagnostics.plane_fallback,actual.diagnostics.polynomial);
            }
        }
    }
    eprintln!("GPU_MLS compared {coordinates} coordinates across all orders; maximum mode-reference difference {maximum}; maximum requested-order CPU difference {requested_maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_mls_preserves_no_support_and_validates_capacity_configuration_and_extent() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(65, 1);
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(MlsSmoothing::new(MlsConfig::with_radius(radius))
            .smoothed_positions_with_gpu(&runtime, &cloud, 128)
            .is_err());
    }
    assert!(MlsSmoothing::new(MlsConfig { polynomial_order: 3, ..Default::default() })
        .smoothed_positions_with_gpu(&runtime, &cloud, 128)
        .is_err());
    let smoother = MlsSmoothing::new(MlsConfig::with_radius(0.16));
    for capacity in [0, 1, 3] {
        assert!(smoother.smoothed_positions_with_gpu(&runtime, &cloud, capacity).is_err());
    }
    for columns in [
        [vec![f32::NAN], vec![0.0], vec![0.0]],
        [vec![f32::MAX, -f32::MAX], vec![0.0; 2], vec![0.0; 2]],
    ] {
        assert!(smoother
            .smoothed_positions_with_gpu(&runtime, &from_columns(columns), 128)
            .is_err());
    }
    let input = from_columns([vec![-0.0, 1.0, 2.0], vec![0.0; 3], vec![-0.0; 3]]);
    for order in 0..=2 {
        let result = MlsSmoothing::new(MlsConfig {
            search_radius: 3.0,
            polynomial_order: order,
            min_neighbors: usize::MAX,
        })
        .smoothed_positions_with_gpu(&runtime, &input, 3)
        .unwrap();
        assert_eq!(result.diagnostics.unchanged, 3);
        let (x, y, z) = input.positions3().unwrap();
        for (i, p) in result.positions.iter().enumerate() {
            assert_eq!(
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                [x[i].to_bits(), y[i].to_bits(), z[i].to_bits()]
            );
        }
    }
    let translated = from_columns([
        (0..25).map(|i| 1e6 + (i % 5) as f32 * 0.25).collect(),
        (0..25).map(|i| -1e6 + (i / 5) as f32 * 0.25).collect(),
        vec![7.0; 25],
    ]);
    let s = MlsSmoothing::new(MlsConfig::with_radius(1.0));
    compare(
        &s.smoothed_positions_with_gpu(&runtime, &translated, 25).unwrap().positions,
        &s.smoothed_positions(&translated).unwrap(),
        1.0,
    );
    let tiny = from_columns([vec![0.0, 1e-22, 2e-22], vec![0.0; 3], vec![0.0; 3]]);
    assert!(MlsSmoothing::new(MlsConfig {
        search_radius: 1.0,
        min_neighbors: 0,
        ..Default::default()
    })
    .smoothed_positions_with_gpu(&runtime, &tiny, 3)
    .is_err());
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_mls_reduces_plane_noise_and_preserves_all_attribute_bits() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut points = Vec::new();
    let mut seed = 12345_u64;
    for x in 0..25 {
        for y in 0..25 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = (((seed >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.02;
            points.push([x as f32 * 0.05, y as f32 * 0.05, noise]);
        }
    }
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for (axis, (name, semantic)) in [
        ("px", FieldSemantic::PositionX),
        ("py", FieldSemantic::PositionY),
        ("pz", FieldSemantic::PositionZ),
    ]
    .into_iter()
    .enumerate()
    {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
        buffers.insert(name, PointBuffer::F32(points.iter().map(|p| p[axis]).collect()));
    }
    for (name, dtype, buffer) in [
        (
            "float",
            DType::F32,
            PointBuffer::F32(
                (0..625)
                    .map(|i| f32::from_bits(if i % 2 == 0 { 0x7fc00037 } else { 0x80000000 }))
                    .collect(),
            ),
        ),
        (
            "double",
            DType::F64,
            PointBuffer::F64(
                (0..625)
                    .map(|i| {
                        f64::from_bits(if i % 2 == 0 {
                            0x7ff8000000000037
                        } else {
                            0x8000000000000000
                        })
                    })
                    .collect(),
            ),
        ),
        ("byte", DType::U8, PointBuffer::U8((0..625).map(|i| i as u8).collect())),
        ("short", DType::U16, PointBuffer::U16((0..625).map(|i| i as u16).collect())),
        ("index", DType::U32, PointBuffer::U32((0..625).map(|i| i as u32).collect())),
        ("signed", DType::I32, PointBuffer::I32((0..625).map(|i| -i).collect())),
    ] {
        schema = schema.with_field(PointField::scalar(name, FieldSemantic::Unknown, dtype));
        buffers.insert(name, buffer);
    }
    let cloud = PointCloud::try_from_parts(
        schema,
        buffers,
        SpatialMetadata::new("map", Timestamp::from_nanos(37)),
    )
    .unwrap();
    let smoother =
        MlsSmoothing::new(MlsConfig { search_radius: 0.2, polynomial_order: 1, min_neighbors: 6 });
    let output = smoother.filter_with_gpu(&runtime, &cloud, 128).unwrap();
    assert_eq!(output.cloud.schema(), cloud.schema());
    assert_eq!(output.cloud.metadata(), cloud.metadata());
    let (_, _, z) = output.cloud.positions3().unwrap();
    let interior: Vec<_> =
        (0..625).filter(|&i| (4..21).contains(&(i / 25)) && (4..21).contains(&(i % 25))).collect();
    let before = (interior.iter().map(|&i| points[i][2].powi(2)).sum::<f32>()
        / interior.len() as f32)
        .sqrt();
    let after =
        (interior.iter().map(|&i| z[i].powi(2)).sum::<f32>() / interior.len() as f32).sqrt();
    assert!(after < before * 0.6, "noise RMS before={before} after={after}");
    for field in cloud.schema().fields().iter().filter(|f| f.semantic == FieldSemantic::Unknown) {
        match (cloud.field(&field.name).unwrap(), output.cloud.field(&field.name).unwrap()) {
            (PointBuffer::F32(a), PointBuffer::F32(b)) => assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            ),
            (PointBuffer::F64(a), PointBuffer::F64(b)) => assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            ),
            (a, b) => assert_eq!(a, b),
        }
    }
    eprintln!("GPU_MLS plane RMS before={before} after={after}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_mls_retains_outputs_and_reuses_scratch_with_concurrent_runtime_guards() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(257, 1);
    let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
    let input = GpuMlsInput::upload(&runtime, &tree).unwrap();
    let different = GpuMlsInput::upload(&runtime, &tree).unwrap();
    let rows = input.neighborhoods_resident(&runtime, 0.16, 128).unwrap();
    assert!(input.neighborhoods_resident(&other, 0.16, 128).is_err());
    assert!(input.smooth_resident(&other, &rows, 2, 6).is_err());
    assert!(different.smooth_resident(&runtime, &rows, 2, 6).is_err());
    let retained = input.smooth_resident(&runtime, &rows, 2, 6).unwrap();
    let expected = retained.download(&runtime).unwrap();
    std::thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        let output = input
                            .smooth_resident(&runtime, &rows, 2, 6)
                            .unwrap()
                            .download(&runtime)
                            .unwrap();
                        assert_eq!(output, expected);
                    }
                })
            })
            .collect();
        for call in calls {
            call.join().unwrap();
        }
    });
    drop(rows);
    drop(input);
    drop(tree);
    assert_eq!(retained.download(&runtime).unwrap(), expected);
    assert!(retained.download(&other).is_err());
    assert!(runtime.scratch_buffer_pool_stats().cached_resources >= 2);
}

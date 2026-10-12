#![cfg(all(feature = "feature-iss-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointCloud, PointField, PointSchema,
    SpatialMetadata, StandardSchemas, Timestamp,
};
use spatialrust_features::{IssKeypointConfig, IssKeypointDetector, IssKeypointResult};
use spatialrust_gpu::{GpuIssInput, WgpuRuntime};

#[path = "fixtures/iss_gpu_reference.rs"]
mod math_reference;
pub use math_reference::audit;

pub fn fixture(n: usize, kind: usize) -> PointCloud {
    let mut columns: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 128) as f32 * 0.05 - 3.2;
        let y = (j / 128) as f32 * 0.05 - 1.6;
        let z = if kind == 0 {
            0.0
        } else {
            0.015 * (x * x + y * y) + ((j * 37 % 101) as f32 / 100.0 - 0.5) * 0.08
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
pub fn compare(actual: &IssKeypointResult, expected: &IssKeypointResult) {
    assert_eq!(actual.indices, expected.indices);
    assert_eq!(actual.keypoints, expected.keypoints);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_iss_compares_selection_and_receipts_with_current_cpu() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut decisions = 0;
    let mut differences = 0;
    let mut max_error = 0.0_f64;
    for n in [0, 1, 2, 17, 63, 64, 65, 257, 4097] {
        for kind in 0..3 {
            let cloud = fixture(n, kind);
            for radius in [0.075, 0.16] {
                let detector =
                    IssKeypointDetector::new(IssKeypointConfig::with_radii(radius, radius * 0.75));
                let expected = detector.detect(&cloud).unwrap();
                let actual = detector.detect_with_gpu(&runtime, &cloud, 128, 128).unwrap();
                let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
                let input = GpuIssInput::upload(&runtime, &tree).unwrap();
                let rows = input
                    .neighborhoods_resident(&runtime, radius, radius * 0.75, 128, 128)
                    .unwrap();
                let checked = audit(&runtime, &cloud, detector.config(), &input, &rows, &expected);
                assert_eq!(
                    actual.result.indices,
                    checked.indices.iter().map(|&i| i as usize).collect::<Vec<_>>()
                );
                differences += checked.differences;
                max_error = max_error.max(checked.max_absolute_eigen_error);
                eprintln!("GPU_ISS_ACCURACY n={n} kind={kind} radius={radius} cpu={} gpu={} differences={} max_abs_eigen_error={}",expected.indices.len(),actual.result.indices.len(),checked.differences,checked.max_absolute_eigen_error);
                decisions += n;
                let nodes = spatialrust_search::KdTree::from_point_cloud(&cloud)
                    .unwrap()
                    .layout()
                    .nodes()
                    .len();
                assert_eq!(
                    actual.transfers.host_to_device_bytes(),
                    if n == 0 { 0 } else { (n * 32 + nodes * 32) as u64 }
                );
                assert_eq!(
                    actual.transfers.device_to_host_bytes(),
                    if n == 0 { 0 } else { (n * 4 + 32) as u64 }
                );
            }
        }
    }
    eprintln!(
        "GPU_ISS audited {decisions} decisions and {} eigenvalues; {differences} CPU/GPU selection differences confined to bounded numerical boundaries; max_abs_eigen_error={max_error}",decisions*3
    );
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_iss_validates_configuration_capacity_degeneracy_and_huge_radii() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(65, 1);
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        for config in [
            IssKeypointConfig::with_radii(radius, 0.12),
            IssKeypointConfig::with_radii(0.16, radius),
        ] {
            assert!(IssKeypointDetector::new(config)
                .detect_with_gpu(&runtime, &cloud, 128, 128)
                .is_err());
        }
    }
    for gamma in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        for config in [
            IssKeypointConfig { gamma_21: gamma, ..Default::default() },
            IssKeypointConfig { gamma_32: gamma, ..Default::default() },
        ] {
            assert!(IssKeypointDetector::new(config)
                .detect_with_gpu(&runtime, &cloud, 128, 128)
                .is_err());
        }
    }
    let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.16, 0.12));
    for (salient, non_max) in [(0, 128), (1, 128), (128, 0), (128, 1)] {
        assert!(detector.detect_with_gpu(&runtime, &cloud, salient, non_max).is_err());
    }
    for columns in [
        [vec![0.0; 33], vec![0.0; 33], vec![0.0; 33]],
        [(0..33).map(|i| i as f32 * 0.01).collect(), vec![0.0; 33], vec![0.0; 33]],
        [vec![f32::NAN], vec![0.0], vec![0.0]],
        [vec![f32::MAX, -f32::MAX], vec![0.0; 2], vec![0.0; 2]],
    ] {
        let cloud = from_columns(columns);
        let result = detector.detect_with_gpu(&runtime, &cloud, 128, 128);
        if cloud.len() == 33 {
            assert!(result.unwrap().result.indices.is_empty());
        } else {
            assert!(result.is_err());
        }
    }
    let sparse = fixture(17, 1);
    for config in [
        IssKeypointConfig { min_neighbors: usize::MAX, ..Default::default() },
        IssKeypointConfig { gamma_21: 0.0, ..Default::default() },
    ] {
        assert!(IssKeypointDetector::new(config)
            .detect_with_gpu(&runtime, &sparse, 128, 128)
            .unwrap()
            .result
            .indices
            .is_empty());
    }
    let huge = IssKeypointDetector::new(IssKeypointConfig::with_radii(1e19, 1e19));
    compare(
        &huge.detect_with_gpu(&runtime, &sparse, 32, 32).unwrap().result,
        &huge.detect(&sparse).unwrap(),
    );
    // Local variance too small for a normal f32 square is rejected explicitly.
    let tiny = from_columns([vec![0.0, 1e-22, 2e-22], vec![0.0; 3], vec![0.0; 3]]);
    assert!(IssKeypointDetector::new(IssKeypointConfig {
        min_neighbors: 3,
        ..IssKeypointConfig::with_radii(1.0, 1.0)
    })
    .detect_with_gpu(&runtime, &tiny, 3, 3)
    .is_err());
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_iss_preserves_all_scalar_attribute_bits_semantics_and_metadata() {
    let source = fixture(4097, 1);
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for (field, name) in source.schema().fields().iter().zip(["px", "py", "pz"]) {
        schema = schema.with_field(PointField::scalar(name, field.semantic, field.dtype));
        buffers.insert(name, source.field(&field.name).unwrap().clone());
    }
    for (name, dtype, buffer) in [
        (
            "float",
            DType::F32,
            PointBuffer::F32(
                (0..source.len())
                    .map(|i| f32::from_bits(if i % 2 == 0 { 0x7fc00037 } else { 0x80000000 }))
                    .collect(),
            ),
        ),
        (
            "double",
            DType::F64,
            PointBuffer::F64(
                (0..source.len())
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
        ("byte", DType::U8, PointBuffer::U8((0..source.len()).map(|i| i as u8).collect())),
        ("short", DType::U16, PointBuffer::U16((0..source.len()).map(|i| i as u16).collect())),
        ("index", DType::U32, PointBuffer::U32((0..source.len()).map(|i| i as u32).collect())),
        ("signed", DType::I32, PointBuffer::I32((0..source.len()).map(|i| -(i as i32)).collect())),
    ] {
        schema = schema.with_field(PointField::scalar(name, FieldSemantic::Unknown, dtype));
        buffers.insert(name, buffer);
    }
    let input = PointCloud::try_from_parts(
        schema,
        buffers,
        SpatialMetadata::new("sensor", Timestamp::from_nanos(37)),
    )
    .unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.16, 0.12));
    let actual = detector.detect_with_gpu(&runtime, &input, 128, 128).unwrap().result;
    let expected = detector.detect(&input).unwrap();
    assert_eq!(actual.indices, expected.indices);
    assert!(!actual.indices.is_empty());
    assert_eq!(actual.keypoints.schema(), input.schema());
    assert_eq!(actual.keypoints.metadata(), input.metadata());
    for field in input.schema().fields() {
        match (
            actual.keypoints.field(&field.name).unwrap(),
            expected.keypoints.field(&field.name).unwrap(),
        ) {
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
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_iss_reuses_owned_resident_outputs_and_checks_runtime_and_frame() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(257, 1);
    let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.16, 0.12));
    let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
    let input = GpuIssInput::upload(&runtime, &tree).unwrap();
    let different = GpuIssInput::upload(&runtime, &tree).unwrap();
    let rows = input.neighborhoods_resident(&runtime, 0.16, 0.12, 128, 128).unwrap();
    let expected = audit(
        &runtime,
        &cloud,
        detector.config(),
        &input,
        &rows,
        &detector.detect(&cloud).unwrap(),
    )
    .indices;
    assert!(input.neighborhoods_resident(&other, 0.16, 0.12, 128, 128).is_err());
    assert!(different.keypoints_resident(&runtime, &rows, 0.975, 0.975, 5).is_err());
    assert!(input.keypoints_resident(&other, &rows, 0.975, 0.975, 5).is_err());
    let retained = input.keypoints_resident(&runtime, &rows, 0.975, 0.975, 5).unwrap();
    std::thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        assert_eq!(
                            input
                                .keypoints_resident(&runtime, &rows, 0.975, 0.975, 5)
                                .unwrap()
                                .download(&runtime)
                                .unwrap()
                                .0,
                            expected
                        );
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
    assert_eq!(retained.download(&runtime).unwrap().0, expected);
    assert!(retained.download(&other).is_err());
}

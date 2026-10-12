use super::*;
use spatialrust_core::{
    DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema, SpatialMetadata,
    StandardSchemas, Timestamp,
};

pub(super) fn fixture(len: usize, kind: usize) -> PointCloud {
    let mut columns: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(len));
    for i in 0..len {
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
    let mut buffers = PointBufferSet::new();
    for (name, column) in ["x", "y", "z"].into_iter().zip(columns) {
        buffers.insert(name, PointBuffer::F32(column));
    }
    PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, Default::default()).unwrap()
}

fn previous_detector(config: IssKeypointConfig) -> previous::IssKeypointDetector {
    previous::IssKeypointDetector::new(previous::IssKeypointConfig {
        salient_radius: config.salient_radius,
        non_max_radius: config.non_max_radius,
        gamma_21: config.gamma_21,
        gamma_32: config.gamma_32,
        min_neighbors: config.min_neighbors,
    })
}

pub(super) fn previous_result(
    cloud: &PointCloud,
    config: IssKeypointConfig,
) -> previous::IssKeypointResult {
    previous_detector(config).detect(cloud).unwrap()
}

pub(super) fn assert_result(actual: &IssKeypointResult, expected: &previous::IssKeypointResult) {
    assert_eq!(actual.indices, expected.indices);
    assert_eq!(actual.keypoints, expected.keypoints);
}

#[test]
fn frozen_selection_and_clouds_match_across_worker_bands_and_public_staging() {
    let mut decisions = 0;
    for len in [0usize, 1, 17, 4095, 4096, 4103, 32771] {
        for kind in 0..3 {
            let cloud = fixture(len, kind);
            for radius in [0.075, 0.16] {
                let config = IssKeypointConfig {
                    min_neighbors: if kind == 2 { 0 } else { 5 },
                    ..IssKeypointConfig::with_radii(radius, radius * 0.75)
                };
                let expected = previous_result(&cloud, config);
                for workers in [1, 2, 6] {
                    let actual = IssKeypointDetector::new(config)
                        .detect_with_workers(&cloud, workers)
                        .unwrap();
                    assert_result(&actual, &expected);
                    decisions += len;
                }
                assert_result(&IssKeypointDetector::new(config).detect(&cloud).unwrap(), &expected);
                decisions += len;
            }
        }
    }
    eprintln!("Verified {decisions} exact ISS selection decisions and selected clouds against frozen serial detection.");
}

#[test]
fn independent_covariance_ties_and_degenerate_support_are_preserved() {
    let mut builder = PointCloudBuilder::xyz();
    for x in [-0.06, 0.06] {
        for y in [-0.04, 0.04] {
            for z in [-0.02, 0.02] {
                builder.push_point([x, y, z]).unwrap();
            }
        }
    }
    // An isolated point does not have enough covariance support.
    builder.push_point([100.0, 100.0, 100.0]).unwrap();
    let cloud = builder.build().unwrap();
    let (x, y, z) = cloud.positions3().unwrap();
    let tree = KdTree::from_slices(x, y, z);
    let eigenvalues =
        neighborhood_eigenvalues(x, y, z, &tree.radius_search(x[0], y[0], z[0], 0.2)).unwrap();
    for (actual, extent) in eigenvalues.into_iter().zip([0.02_f32, 0.04, 0.06]) {
        assert!((actual - f64::from(extent).powi(2)).abs() < 1e-14);
    }
    let config = IssKeypointConfig::with_radii(0.2, 1.0);
    for workers in [0, 1, 2, 6, usize::MAX] {
        let actual = IssKeypointDetector::new(config).detect_with_workers(&cloud, workers).unwrap();
        assert_eq!(actual.indices, (0..8).collect::<Vec<_>>(), "equal maxima must all survive");
        assert_eq!(actual.keypoints.len(), 8);
    }
    for points in [
        vec![[0.0, 0.0, 0.0]; 33],
        (0..33).map(|i| [i as f32 * 0.01, 0.0, 0.0]).collect(),
        vec![[-0.0, 0.0, -0.0], [100.0, 0.0, 0.0]],
    ] {
        let mut builder = PointCloudBuilder::xyz();
        for p in points {
            builder.push_point(p).unwrap();
        }
        let cloud = builder.build().unwrap();
        for min_neighbors in [0, 3, usize::MAX] {
            let config =
                IssKeypointConfig { min_neighbors, ..IssKeypointConfig::with_radii(0.025, 0.02) };
            for workers in [1, 2, 6] {
                let actual =
                    IssKeypointDetector::new(config).detect_with_workers(&cloud, workers).unwrap();
                assert!(actual.indices.is_empty());
                assert_result(&actual, &previous_result(&cloud, config));
            }
        }
    }
}

#[test]
fn selected_indices_preserve_schema_metadata_and_all_attribute_words() {
    let source = fixture(32771, 1);
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
    let config = IssKeypointConfig::with_radii(0.16, 0.12);
    let expected = previous_result(&input, config);
    assert!(!expected.indices.is_empty());
    let detector = IssKeypointDetector::new(config);
    for result in
        [detector.detect_with_workers(&input, 6).unwrap(), detector.detect(&input).unwrap()]
    {
        assert_eq!(result.indices, expected.indices);
        assert!(result.indices.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(result.keypoints.schema(), input.schema());
        assert_eq!(result.keypoints.metadata(), input.metadata());
        for field in input.schema().fields() {
            match (
                result.keypoints.field(&field.name).unwrap(),
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
}

#[test]
fn validation_unusual_configuration_and_concurrent_calls_match_serial_contract() {
    let cloud = fixture(17, 1);
    let empty = fixture(0, 0);
    let singleton = fixture(1, 0);
    for input in [&cloud, &empty, &singleton] {
        for config in [
            IssKeypointConfig::with_radii(0.0, 0.1),
            IssKeypointConfig::with_radii(0.1, -1.0),
            IssKeypointConfig::with_radii(f32::NAN, 0.1),
            IssKeypointConfig::with_radii(0.1, f32::NAN),
            IssKeypointConfig::with_radii(f32::INFINITY, f32::INFINITY),
            IssKeypointConfig { gamma_21: f32::NAN, ..IssKeypointConfig::with_radii(0.16, 0.12) },
            IssKeypointConfig { gamma_32: -1.0, ..IssKeypointConfig::with_radii(0.16, 0.12) },
        ] {
            let expected = previous_detector(config)
                .detect(input)
                .map(|r| (r.indices, r.keypoints))
                .map_err(|e| e.to_string());
            for workers in [0, 1, 2, 6, usize::MAX] {
                let actual = IssKeypointDetector::new(config)
                    .detect_with_workers(input, workers)
                    .map(|r| (r.indices, r.keypoints))
                    .map_err(|e| e.to_string());
                assert_eq!(actual, expected);
            }
        }
    }
    let config = IssKeypointConfig::with_radii(0.16, 0.12);
    let cloud = fixture(4103, 1);
    let expected = previous_result(&cloud, config);
    std::thread::scope(|scope| {
        let calls: Vec<_> = [2, 6]
            .into_iter()
            .map(|workers| {
                let cloud = &cloud;
                scope.spawn(move || {
                    IssKeypointDetector::new(config).detect_with_workers(cloud, workers).unwrap()
                })
            })
            .collect();
        for call in calls {
            assert_result(&call.join().unwrap(), &expected);
        }
    });
}

use super::*;
use spatialrust_core::{
    DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema, SpatialMetadata, Timestamp,
};

pub(super) fn fixture(len: usize, kind: usize) -> PointCloud {
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    let mut columns: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::with_capacity(len));
    for i in 0..len {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 128) as f32 * 0.05 - 3.2;
        let y = (j / 128) as f32 * 0.05 - 1.6;
        let (z, normal) = if kind == 0 {
            (0.0, Vec3::new(0.0, 0.0, 1.0))
        } else {
            (0.015 * (x * x + y * y), Vec3::new(-0.03 * x, -0.03 * y, 1.0).normalize())
        };
        for (column, value) in columns.iter_mut().zip([x, y, z, normal.x, normal.y, normal.z]) {
            column.push(value);
        }
    }
    for ((name, semantic), values) in [
        ("x", FieldSemantic::PositionX),
        ("y", FieldSemantic::PositionY),
        ("z", FieldSemantic::PositionZ),
        ("normal_x", FieldSemantic::NormalX),
        ("normal_y", FieldSemantic::NormalY),
        ("normal_z", FieldSemantic::NormalZ),
    ]
    .into_iter()
    .zip(columns)
    {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
        buffers.insert(name, PointBuffer::F32(values));
    }
    PointCloud::try_from_parts(schema, buffers, Default::default()).unwrap()
}

fn previous_detector(config: BoundaryConfig) -> previous::BoundaryDetector {
    previous::BoundaryDetector::new(previous::BoundaryConfig {
        search_radius: config.search_radius,
        angle_threshold: config.angle_threshold,
        min_neighbors: config.min_neighbors,
    })
}

pub(super) fn previous_mask(cloud: &PointCloud, config: BoundaryConfig) -> Vec<bool> {
    previous_detector(config).boundary_mask(cloud).unwrap()
}

#[test]
fn frozen_masks_match_across_worker_bands_and_public_staging() {
    let mut comparisons = 0;
    for len in [0usize, 1, 17, 4095, 4096, 4103, 32771] {
        for kind in 0..3 {
            let cloud = fixture(len, kind);
            for radius in [0.075, 0.16] {
                let config = BoundaryConfig {
                    min_neighbors: if kind == 2 { 0 } else { 3 },
                    ..BoundaryConfig::with_radius(radius)
                };
                let expected = previous_mask(&cloud, config);
                for workers in [1, 2, 6] {
                    let actual = BoundaryDetector::new(config)
                        .boundary_mask_with_workers(&cloud, workers)
                        .unwrap();
                    assert_eq!(actual, expected, "points={len}, kind={kind}, workers={workers}");
                    comparisons += len;
                }
                assert_eq!(BoundaryDetector::new(config).boundary_mask(&cloud).unwrap(), expected);
                comparisons += len;
            }
        }
    }
    eprintln!("Verified {comparisons} exact boundary decisions against frozen serial detection.");
}

#[test]
fn independent_square_rim_and_degenerate_support_are_preserved() {
    let schema = fixture(0, 0).schema().clone();
    let mut builder = PointCloudBuilder::new(schema.clone());
    let side = 17;
    for i in 0..side * side {
        builder
            .push_point([(i % side) as f32 * 0.05, (i / side) as f32 * 0.05, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    }
    let cloud = builder.build().unwrap();
    let config = BoundaryConfig { min_neighbors: 3, ..BoundaryConfig::with_radius(0.075) };
    for workers in [0, 1, 2, 6] {
        let mask =
            BoundaryDetector::new(config).boundary_mask_with_workers(&cloud, workers).unwrap();
        for (i, is_boundary) in mask.into_iter().enumerate() {
            let (x, y) = (i % side, i / side);
            assert_eq!(is_boundary, x == 0 || y == 0 || x == side - 1 || y == side - 1);
        }
    }

    for points in [
        vec![[0.0, 0.0, 0.0]; 33],
        (0..33).map(|i| [i as f32 * 0.01, 0.0, 0.0]).collect(),
        vec![[-0.0, 0.0, -0.0], [100.0, 0.0, 0.0]],
    ] {
        for normal in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.1],
            [f32::NAN, 0.0, 1.0],
            [f32::INFINITY, 0.0, 1.0],
        ] {
            let mut builder = PointCloudBuilder::new(schema.clone());
            for p in &points {
                builder.push_point([p[0], p[1], p[2], normal[0], normal[1], normal[2]]).unwrap();
            }
            let cloud = builder.build().unwrap();
            for (min_neighbors, angle_threshold) in
                [(0, -1.0), (0, 0.0), (3, std::f32::consts::PI), (usize::MAX, 1.0), (0, f32::NAN)]
            {
                let config =
                    BoundaryConfig { search_radius: 0.025, min_neighbors, angle_threshold };
                let expected = previous_mask(&cloud, config);
                for workers in [1, 2, 6] {
                    assert_eq!(
                        BoundaryDetector::new(config)
                            .boundary_mask_with_workers(&cloud, workers)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn selected_indices_preserve_schema_metadata_and_every_buffer_type() {
    let source = fixture(32771, 1);
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for (field, renamed) in
        source.schema().fields().iter().zip(["px", "py", "pz", "nx", "ny", "nz"])
    {
        schema = schema.with_field(PointField::scalar(renamed, field.semantic, field.dtype));
        buffers.insert(renamed, source.field(&field.name).unwrap().clone());
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
    let config = BoundaryConfig { min_neighbors: 3, ..BoundaryConfig::with_radius(0.16) };
    let expected = previous_detector(config).detect(&input).unwrap();
    let result = BoundaryDetector::new(config).detect(&input).unwrap();
    assert_eq!(result.indices, expected.indices);
    assert!(!result.indices.is_empty());
    assert!(result.indices.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(result.boundary.schema(), input.schema());
    assert_eq!(result.boundary.metadata(), input.metadata());
    for field in input.schema().fields() {
        let actual = result.boundary.field(&field.name).unwrap();
        let frozen = expected.boundary.field(&field.name).unwrap();
        match (actual, frozen) {
            (PointBuffer::F32(a), PointBuffer::F32(b)) => {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            (PointBuffer::F64(a), PointBuffer::F64(b)) => {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            (a, b) => {
                assert_eq!(a, b);
            }
        }
    }
}

#[test]
fn validation_empty_shortcuts_and_concurrent_calls_match_serial_contract() {
    let cloud = fixture(4103, 1);
    let mut empty_buffers = PointBufferSet::new();
    for name in ["x", "y", "z"] {
        empty_buffers.insert(name, PointBuffer::F32(Vec::new()));
    }
    let no_normals = PointCloud::try_from_parts(
        spatialrust_core::StandardSchemas::point_xyz(),
        empty_buffers,
        Default::default(),
    )
    .unwrap();
    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([0.0, 0.0, 0.0]).unwrap();
    let missing_normals = builder.build().unwrap();
    for input in [&cloud, &no_normals, &missing_normals] {
        for radius in [0.0, -1.0, f32::NAN, 0.16] {
            let config = BoundaryConfig::with_radius(radius);
            let expected =
                previous_detector(config).boundary_mask(input).map_err(|e| e.to_string());
            let worker_requests =
                if input.len() <= 1 { vec![0, 1, 2, 6, usize::MAX] } else { vec![0, 1, 2, 6] };
            for workers in worker_requests {
                assert_eq!(
                    BoundaryDetector::new(config)
                        .boundary_mask_with_workers(input, workers)
                        .map_err(|e| e.to_string()),
                    expected
                );
            }
        }
    }
    let config = BoundaryConfig::with_radius(0.16);
    let expected = previous_mask(&cloud, config);
    std::thread::scope(|scope| {
        let calls: Vec<_> = [2, 6]
            .into_iter()
            .map(|workers| {
                let cloud = &cloud;
                scope.spawn(move || {
                    BoundaryDetector::new(config)
                        .boundary_mask_with_workers(cloud, workers)
                        .unwrap()
                })
            })
            .collect();
        for call in calls {
            assert_eq!(call.join().unwrap(), expected);
        }
    });
}

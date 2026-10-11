use super::*;
use spatialrust_core::{
    DType, PointCloudBuilder, PointField, PointSchema, SpatialMetadata, Timestamp,
};

pub(super) fn fixture(len: usize, kind: usize) -> PointCloud {
    if len == 0 {
        let mut buffers = PointBufferSet::new();
        for name in ["x", "y", "z"] {
            buffers.insert(name, PointBuffer::F32(Vec::new()));
        }
        return PointCloud::try_from_parts(
            spatialrust_core::StandardSchemas::point_xyz(),
            buffers,
            Default::default(),
        )
        .unwrap();
    }
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..len {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 128) as f32 * 0.05 - 3.2;
        let y = (j / 128) as f32 * 0.05 - 1.6;
        let noise = ((j * 37 % 101) as f32 / 100.0 - 0.5) * 0.02;
        let z = match kind {
            0 => noise,
            1 | 2 => 0.015 * (x * x + y * y) + noise,
            _ => 0.0,
        };
        builder.push_point([x, y, z]).unwrap();
    }
    builder.build().unwrap()
}

pub(super) fn previous_positions(cloud: &PointCloud, config: MlsConfig) -> Vec<Vec3<f32>> {
    previous::MlsSmoothing::new(previous::MlsConfig {
        search_radius: config.search_radius,
        polynomial_order: config.polynomial_order,
        min_neighbors: config.min_neighbors,
    })
    .smoothed_positions(cloud)
    .unwrap()
}

pub(super) fn assert_words(actual: &[Vec3<f32>], expected: &[Vec3<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            [a.x.to_bits(), a.y.to_bits(), a.z.to_bits()],
            [b.x.to_bits(), b.y.to_bits(), b.z.to_bits()],
            "point {i}"
        );
    }
}

#[test]
fn frozen_position_words_match_for_orders_duplicates_and_worker_bands() {
    let mut comparisons = 0;
    for len in [0usize, 1, 17, 4095, 4096, 4103] {
        for kind in 0..3 {
            let cloud = fixture(len, kind);
            // Rotate fit orders on the large cases; exhaust all orders on small ones.
            let orders: Vec<_> = if len < 100 { vec![0, 1, 2] } else { vec![kind as u8] };
            for order in orders {
                let config =
                    MlsConfig { search_radius: 0.16, polynomial_order: order, min_neighbors: 6 };
                let expected = previous_positions(&cloud, config);
                for workers in [1, 2, 6] {
                    let actual = MlsSmoothing::new(config)
                        .smoothed_positions_with_workers(&cloud, workers)
                        .unwrap();
                    assert_words(&actual, &expected);
                    comparisons += len * 3;
                }
                assert_words(
                    &MlsSmoothing::new(config).smoothed_positions(&cloud).unwrap(),
                    &expected,
                );
                comparisons += len * 3;
            }
        }
    }
    eprintln!("Verified {comparisons} exact MLS coordinate words against frozen serial fits.");
}

#[test]
fn sparse_collinear_and_duplicate_support_preserves_fallbacks() {
    for points in [
        vec![[0.0, 0.0, 0.0]; 33],
        (0..33).map(|i| [i as f32 * 0.01, 0.0, 0.0]).collect(),
        vec![[-0.0, 0.0, -0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ] {
        let mut builder = PointCloudBuilder::xyz();
        for p in points {
            builder.push_point(p).unwrap();
        }
        let cloud = builder.build().unwrap();
        for order in [0, 1, 2] {
            for min_neighbors in [0, 3, 6, usize::MAX] {
                let config =
                    MlsConfig { search_radius: 0.025, polynomial_order: order, min_neighbors };
                let expected = previous_positions(&cloud, config);
                for workers in [1, 2, 6] {
                    assert_words(
                        &MlsSmoothing::new(config)
                            .smoothed_positions_with_workers(&cloud, workers)
                            .unwrap(),
                        &expected,
                    );
                }
            }
        }
    }
}

#[test]
fn public_large_cloud_uses_shared_staging_and_preserves_planar_surface() {
    let cloud = fixture(32771, 3);
    let config = MlsConfig { search_radius: 0.075, polynomial_order: 1, min_neighbors: 3 };
    let actual = MlsSmoothing::new(config).smoothed_positions(&cloud).unwrap();
    let expected = previous_positions(&cloud, config);
    assert_words(&actual, &expected);
    let (x, y, _) = cloud.positions3().unwrap();
    for (i, p) in actual.iter().enumerate() {
        assert!((p.x - x[i]).abs() < 1e-5);
        assert!((p.y - y[i]).abs() < 1e-5);
        assert!(p.z.abs() < 1e-6);
    }
}

#[test]
fn filtered_schema_metadata_and_all_attribute_types_are_preserved() {
    let cloud = fixture(4103, 1);
    let (x, y, z) = cloud.positions3().unwrap();
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for (name, semantic, values) in [
        ("px", FieldSemantic::PositionX, x),
        ("py", FieldSemantic::PositionY, y),
        ("pz", FieldSemantic::PositionZ, z),
    ] {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
        buffers.insert(name.to_owned(), PointBuffer::from_f32(values.to_vec()));
    }
    for (name, dtype, values) in [
        ("float", DType::F32, PointBuffer::F32(vec![f32::from_bits(0x7fc00037); cloud.len()])),
        (
            "double",
            DType::F64,
            PointBuffer::F64(vec![f64::from_bits(0x7ff8000000000037); cloud.len()]),
        ),
        ("byte", DType::U8, PointBuffer::U8((0..cloud.len()).map(|i| i as u8).collect())),
        ("short", DType::U16, PointBuffer::U16((0..cloud.len()).map(|i| i as u16).collect())),
        ("index", DType::U32, PointBuffer::U32((0..cloud.len()).map(|i| i as u32).collect())),
        ("signed", DType::I32, PointBuffer::I32((0..cloud.len()).map(|i| -(i as i32)).collect())),
    ] {
        schema = schema.with_field(PointField::scalar(name, FieldSemantic::Unknown, dtype));
        buffers.insert(name.to_owned(), values);
    }
    let metadata = SpatialMetadata::new("map", Timestamp::from_nanos(1234567));
    let input = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
    let config = MlsConfig { search_radius: 0.16, polynomial_order: 2, min_neighbors: 6 };
    let positions = MlsSmoothing::new(config).smoothed_positions_with_workers(&input, 6).unwrap();
    let out = build_output(&input, &positions).unwrap();
    let public = MlsSmoothing::new(config).filter(&input).unwrap();
    assert_eq!(out.schema(), input.schema());
    assert_eq!(out.metadata(), input.metadata());
    assert_eq!(public.metadata(), out.metadata());
    let (px, py, pz) = public.positions3().unwrap();
    assert_words(
        &px.iter().zip(py).zip(pz).map(|((&x, &y), &z)| Vec3::new(x, y, z)).collect::<Vec<_>>(),
        &positions,
    );
    for name in ["byte", "short", "index", "signed"] {
        assert_eq!(out.field(name).unwrap(), input.field(name).unwrap());
        assert_eq!(public.field(name).unwrap(), input.field(name).unwrap());
    }
    for output in [&out, &public] {
        let PointBuffer::F32(values) = output.field("float").unwrap() else { panic!() };
        assert!(values.iter().all(|v| v.to_bits() == 0x7fc00037));
        let PointBuffer::F64(values) = output.field("double").unwrap() else { panic!() };
        assert!(values.iter().all(|v| v.to_bits() == 0x7ff8000000000037));
    }
}

#[test]
fn validation_and_concurrent_calls_match_the_existing_contract() {
    let cloud = fixture(4103, 1);
    for config in [
        MlsConfig::with_radius(0.0),
        MlsConfig::with_radius(-1.0),
        MlsConfig::with_radius(f32::NAN),
        MlsConfig { polynomial_order: 3, ..MlsConfig::with_radius(0.16) },
    ] {
        for workers in [1, 2, 6] {
            assert!(MlsSmoothing::new(config)
                .smoothed_positions_with_workers(&cloud, workers)
                .is_err());
        }
    }
    let config = MlsConfig::with_radius(0.16);
    let expected = previous_positions(&cloud, config);
    std::thread::scope(|scope| {
        let mut calls = Vec::new();
        for workers in [2, 6] {
            let cloud = &cloud;
            calls.push(scope.spawn(move || {
                MlsSmoothing::new(config).smoothed_positions_with_workers(cloud, workers).unwrap()
            }));
        }
        for call in calls {
            assert_words(&call.join().unwrap(), &expected);
        }
    });
}

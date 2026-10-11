use super::*;
use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointField, PointSchema, SpatialMetadata,
    Timestamp,
};

#[allow(dead_code)]
#[path = "../tests/fixtures/fpfh_serial.rs"]
mod previous;

pub(super) fn fixture(len: usize, kind: usize) -> PointCloud {
    let mut values: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::with_capacity(len));
    for i in 0..len {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 128) as f32 * 0.05 - 3.2;
        let y = (j / 128) as f32 * 0.05 - 1.6;
        let z = if kind == 0 { 0.0 } else { 0.02 * (x * x + y * y) };
        let normal = if kind == 0 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(-0.04 * x, -0.04 * y, 1.0).normalize()
        };
        for (axis, value) in values.iter_mut().zip([x, y, z, normal.x, normal.y, normal.z]) {
            axis.push(value);
        }
    }
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for ((name, semantic), values) in [
        ("px", FieldSemantic::PositionX),
        ("py", FieldSemantic::PositionY),
        ("pz", FieldSemantic::PositionZ),
        ("nx", FieldSemantic::NormalX),
        ("ny", FieldSemantic::NormalY),
        ("nz", FieldSemantic::NormalZ),
    ]
    .into_iter()
    .zip(values)
    {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
        buffers.insert(name, PointBuffer::F32(values));
    }
    schema = schema.with_field(PointField::scalar("id", FieldSemantic::Unknown, DType::U32));
    buffers.insert("id", PointBuffer::U32((0..len).map(|i| i as u32).collect()));
    PointCloud::try_from_parts(
        schema,
        buffers,
        SpatialMetadata::new("sensor", Timestamp::from_nanos(37)),
    )
    .unwrap()
}

pub(super) fn previous_descriptors(cloud: &PointCloud, radius: f32) -> Vec<FpfhDescriptor> {
    previous::fpfh_descriptors(cloud, radius).unwrap()
}

pub(super) fn assert_words(actual: &[FpfhDescriptor], expected: &[FpfhDescriptor]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits), "descriptor {i}");
    }
}

#[test]
fn frozen_descriptor_words_match_for_radii_duplicates_and_worker_bands() {
    let mut words = 0;
    for len in [0usize, 1, 17, 4095, 4096, 4103] {
        for kind in 0..3 {
            let cloud = fixture(len, kind);
            for radius in [0.075, 0.16] {
                let expected = previous_descriptors(&cloud, radius);
                for workers in [1, 2, 6] {
                    assert_words(
                        &fpfh_descriptors_with_workers(&cloud, radius, workers).unwrap(),
                        &expected,
                    );
                    words += len * FPFH_DIM;
                }
                assert_words(&fpfh_descriptors(&cloud, radius).unwrap(), &expected);
                words += len * FPFH_DIM;
            }
        }
    }
    // Cross the real two-worker staging boundary with an uneven final band.
    for kind in [1, 2] {
        let cloud = fixture(32771, kind);
        let expected = previous_descriptors(&cloud, 0.11);
        for workers in [1, 2, 6] {
            assert_words(&fpfh_descriptors_with_workers(&cloud, 0.11, workers).unwrap(), &expected);
            words += cloud.len() * FPFH_DIM;
        }
        assert_words(&fpfh_descriptors(&cloud, 0.11).unwrap(), &expected);
        words += cloud.len() * FPFH_DIM;
    }
    eprintln!("Verified {words} exact FPFH descriptor words against the frozen serial API.");
}

#[test]
fn independent_planar_bins_and_degenerate_neighborhoods_are_exact() {
    // One pair: Darboux alpha/phi/theta are zero. Each own and neighboring
    // SPFH contributes exactly 100 to the central bin of each sub-histogram.
    let cloud = fixture(2, 0);
    for workers in [0, 1, 2, 6, usize::MAX] {
        let actual = fpfh_descriptors_with_workers(&cloud, 0.075, workers).unwrap();
        for descriptor in actual {
            for (bin, value) in descriptor.into_iter().enumerate() {
                assert_eq!(value.to_bits(), if bin % BINS == 5 { 200.0_f32.to_bits() } else { 0 });
            }
        }
    }
    // Radius excludes the other point; coincident duplicates and collinear
    // displacement parallel to the normal all have no valid Darboux frame.
    for cloud in [fixture(1, 0), fixture(2, 0), fixture(2, 2)] {
        for workers in [1, 2, 6] {
            let actual = fpfh_descriptors_with_workers(&cloud, 0.001, workers).unwrap();
            assert!(actual.iter().flatten().all(|v| v.to_bits() == 0));
        }
    }
    let mut cloud = fixture(2, 0);
    let (schema, mut buffers, metadata) = cloud.clone().into_parts();
    buffers.insert("px", PointBuffer::F32(vec![0.0, 0.0]));
    buffers.insert("py", PointBuffer::F32(vec![0.0, 0.0]));
    buffers.insert("pz", PointBuffer::F32(vec![0.0, 0.05]));
    cloud = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
    for workers in [1, 2, 6] {
        assert!(fpfh_descriptors_with_workers(&cloud, 0.075, workers)
            .unwrap()
            .iter()
            .flatten()
            .all(|v| v.to_bits() == 0));
    }
}

#[test]
fn validation_preserves_the_serial_error_contract() {
    let cloud = fixture(17, 1);
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        let expected = previous::fpfh_descriptors(&cloud, radius).unwrap_err().to_string();
        for workers in [1, 2, 6] {
            assert_eq!(
                fpfh_descriptors_with_workers(&cloud, radius, workers).unwrap_err().to_string(),
                expected
            );
        }
    }
    for (field, value) in [("px", f32::NAN), ("ny", f32::INFINITY), ("nz", 2.0)] {
        let (schema, mut buffers, metadata) = cloud.clone().into_parts();
        let mut values = vec![0.0; cloud.len()];
        values[0] = value;
        buffers.insert(field, PointBuffer::F32(values));
        let invalid = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
        let expected = previous::fpfh_descriptors(&invalid, 0.11).unwrap_err().to_string();
        for workers in [1, 2, 6] {
            assert_eq!(
                fpfh_descriptors_with_workers(&invalid, 0.11, workers).unwrap_err().to_string(),
                expected
            );
        }
    }
}

#[test]
fn concurrent_calls_preserve_inputs_and_descriptor_order() {
    let cloud = fixture(4103, 2);
    let before = cloud.clone();
    let expected = previous_descriptors(&cloud, 0.11);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = [2, 6]
            .into_iter()
            .map(|workers| {
                let cloud = &cloud;
                scope.spawn(move || fpfh_descriptors_with_workers(cloud, 0.11, workers).unwrap())
            })
            .collect();
        for job in jobs {
            assert_words(&job.join().unwrap(), &expected);
        }
    });
    assert_eq!(cloud, before);
}

#[test]
fn seeded_registration_matches_frozen_transform_and_diagnostics() {
    let (schema, mut buffers, metadata) = fixture(193, 1).into_parts();
    for (name, multiplier, modulus) in [("px", 31, 257), ("py", 97, 263), ("pz", 101, 269)] {
        buffers.insert(
            name,
            PointBuffer::F32(
                (0..193)
                    .map(|i| (i * multiplier % modulus) as f32 / modulus as f32 - 0.5)
                    .collect(),
            ),
        );
    }
    let normals: Vec<_> = (0..193)
        .map(|i| {
            Vec3::new((i * 37 % 41) as f32 / 20.0 - 1.0, (i * 23 % 43) as f32 / 21.0 - 1.0, 1.0)
                .normalize()
        })
        .collect();
    buffers.insert("nx", PointBuffer::F32(normals.iter().map(|n| n.x).collect()));
    buffers.insert("ny", PointBuffer::F32(normals.iter().map(|n| n.y).collect()));
    buffers.insert("nz", PointBuffer::F32(normals.iter().map(|n| n.z).collect()));
    let target = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
    let source = crate::transform::transform_point_cloud(
        &target,
        Isometry3::new(
            spatialrust_math::Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.2),
            Vec3::new(0.1, -0.05, 0.025),
        ),
    )
    .unwrap();
    let config = FpfhRansacConfig {
        feature_radius: 0.5,
        max_correspondence_distance: 0.075,
        ransac_iterations: 97,
        seed: 7,
        ..Default::default()
    };
    let current = FpfhRansacRegistration::new(config).align(&source, &target).unwrap();
    let previous = previous::FpfhRansacRegistration::new(previous::FpfhRansacConfig {
        feature_radius: config.feature_radius,
        max_correspondence_distance: config.max_correspondence_distance,
        ransac_iterations: config.ransac_iterations,
        sample_size: config.sample_size,
        edge_length_tolerance: config.edge_length_tolerance,
        seed: config.seed,
    })
    .align(&source, &target)
    .unwrap();
    assert_eq!(current, previous);
    assert!(current.converged);
    assert!(current.fitness.is_finite());
    assert_eq!(current.fitness.to_bits(), previous.fitness.to_bits());
}

#[path = "fpfh_parallel_benchmark.rs"]
mod benchmark;

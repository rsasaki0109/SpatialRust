use super::*;
use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointField, PointSchema, SpatialMetadata,
    StandardSchemas, Timestamp,
};

fn from_points(points: impl IntoIterator<Item = [f32; 3]>) -> PointCloud {
    let mut columns: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for p in points {
        for (column, value) in columns.iter_mut().zip(p) {
            column.push(value);
        }
    }
    let mut buffers = PointBufferSet::new();
    for (name, values) in ["x", "y", "z"].into_iter().zip(columns) {
        buffers.insert(name, PointBuffer::F32(values));
    }
    PointCloud::try_from_parts(StandardSchemas::point_xyz(), buffers, Default::default()).unwrap()
}

fn fixture(len: usize, kind: usize, shifted: bool) -> PointCloud {
    from_points((0..len).map(|i| {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 128) as f32 * 0.05 - 3.2;
        let y = (j / 128) as f32 * 0.05 - 1.6;
        let z = if kind == 0 {
            0.0
        } else {
            0.015 * (x * x + y * y) + ((j * 37 % 101) as f32 / 100.0 - 0.5) * 0.08
        };
        if shifted {
            [x + 0.017, y - 0.011, z + 0.023]
        } else {
            [x, y, z]
        }
    }))
}

pub(super) fn fixture_pair(len: usize, kind: usize) -> (PointCloud, PointCloud) {
    let target_len = if kind == 2 { len.div_ceil(2) } else { len };
    (fixture(len, kind, false), fixture(target_len, kind.min(1), true))
}

pub(super) fn words(value: CloudDistances) -> [u64; 6] {
    [
        value.mean_sq_a_to_b,
        value.mean_sq_b_to_a,
        value.max_a_to_b,
        value.max_b_to_a,
        value.chamfer(),
        value.hausdorff(),
    ]
    .map(f64::to_bits)
}

pub(super) fn previous_words(a: &PointCloud, b: &PointCloud) -> [u64; 6] {
    let p = previous::cloud_distances(a, b).unwrap();
    [p.mean_sq_a_to_b, p.mean_sq_b_to_a, p.max_a_to_b, p.max_b_to_a, p.chamfer(), p.hausdorff()]
        .map(f64::to_bits)
}

#[test]
fn frozen_metric_words_match_across_staging_windows_and_asymmetric_clouds() {
    let mut comparisons = 0;
    for len in [1usize, 17, 4095, 4096, 4103, 32767, 32768, 32771, 65537] {
        for kind in 0..3 {
            let (a, b) = fixture_pair(len, kind);
            let expected = previous_words(&a, &b);
            for workers in [1, 2, 6] {
                assert_eq!(
                    words(cloud_distances_with_workers(&a, &b, workers, workers).unwrap()),
                    expected,
                    "len={len}, kind={kind}, workers={workers}"
                );
                comparisons += 6;
            }
            assert_eq!(words(cloud_distances(&a, &b).unwrap()), expected);
            comparisons += 6;
            if len == 17 || len == 32771 {
                assert_eq!(chamfer_distance(&a, &b).unwrap().to_bits(), expected[4]);
                assert_eq!(hausdorff_distance(&a, &b).unwrap().to_bits(), expected[5]);
                comparisons += 2;
            }
        }
    }
    eprintln!("Verified {comparisons} exact f64 metric words against frozen serial distances.");
}

#[test]
fn independent_asymmetric_statistics_and_ties_are_preserved() {
    let a = from_points([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let b = from_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
    let expected = [0.5, 5.0 / 3.0, 1.0, 2.0, 0.5 + 5.0 / 3.0, 2.0].map(f64::to_bits);
    for (wa, wb) in [(0, 0), (1, 6), (2, 1), (6, 6), (usize::MAX, usize::MAX)] {
        assert_eq!(words(cloud_distances_with_workers(&a, &b, wa, wb).unwrap()), expected);
    }
    let a = from_points([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let b = from_points([[0.0, 0.0, 0.0]]);
    assert_eq!(
        words(cloud_distances_with_workers(&a, &b, 2, 1).unwrap()),
        [1.0, 1.0, 1.0, 1.0, 2.0, 1.0].map(f64::to_bits)
    );
}

#[test]
fn ordered_accumulation_survives_multiple_windows_and_one_point_tail() {
    let len = 65537;
    let a = from_points(
        (0..len).map(|i| [if i == 0 { 1e8 } else { (i - 1) as f32 / 65536.0 }, 0.0, 0.0]),
    );
    let b = from_points([[0.0, 0.0, 0.0]]);
    let first_sq = f64::from(1e8_f32 * 1e8_f32);
    let ordered_mean = first_sq / len as f64;
    // Every later squared distance is below 1 and rounds away at this
    // magnitude. Distinct small coordinates avoid stressing the unrelated
    // KD-tree builder's equal-coordinate partitioning in this math test.
    let small_sum: f64 = (0..(len - 1))
        .map(|i| {
            let x = i as f32 / 65536.0;
            f64::from(x * x)
        })
        .sum();
    let regrouped_mean = (first_sq + small_sum) / len as f64;
    assert_ne!(ordered_mean.to_bits(), regrouped_mean.to_bits());
    let expected = previous_words(&a, &b);
    assert_eq!(expected[0], ordered_mean.to_bits());
    for (wa, wb) in [(1, 1), (2, 1), (6, 2)] {
        assert_eq!(words(cloud_distances_with_workers(&a, &b, wa, wb).unwrap()), expected);
    }
    assert_eq!(words(cloud_distances(&a, &b).unwrap()), expected);
}

#[test]
fn nonfinite_distances_missing_neighbors_and_empty_errors_match_serial() {
    let zero = from_points([[0.0, 0.0, 0.0]]);
    for points in [
        vec![[f32::from_bits(0x7fc00037), 0.0, 0.0]; 33],
        vec![[f32::INFINITY, 0.0, 0.0]; 33],
        vec![[f32::MAX, f32::MAX, f32::MAX]; 33],
        vec![[-0.0, 0.0, -0.0]; 33],
        (0..33).map(|i| [if i % 2 == 0 { f32::NAN } else { i as f32 }, 0.0, 0.0]).collect(),
    ] {
        let cloud = from_points(points);
        for (a, b) in [(&cloud, &zero), (&zero, &cloud)] {
            let expected = previous_words(a, b);
            for workers in [1, 2, 6] {
                assert_eq!(
                    words(cloud_distances_with_workers(a, b, workers, workers).unwrap()),
                    expected
                );
            }
        }
    }
    let nan = from_points([[f32::NAN, 0.0, 0.0]]);
    assert!(cloud_distances_with_workers(&nan, &zero, 2, 1).unwrap().mean_sq_a_to_b.is_nan());
    let no_target = KdTree::from_slices(&[], &[], &[]);
    assert_eq!(directed(&[0.0, f32::NAN, 1.0], &[0.0; 3], &[0.0; 3], &no_target, 2), (0.0, 0.0));
    let empty = from_points([]);
    for (a, b) in [(&empty, &zero), (&zero, &empty), (&empty, &empty)] {
        let expected = previous::cloud_distances(a, b).unwrap_err().to_string();
        for (wa, wb) in [(0, 0), (1, 1), (2, 6), (usize::MAX, usize::MAX)] {
            assert_eq!(
                cloud_distances_with_workers(a, b, wa, wb).unwrap_err().to_string(),
                expected
            );
        }
        assert_eq!(chamfer_distance(a, b).unwrap_err().to_string(), expected);
        assert_eq!(hausdorff_distance(a, b).unwrap_err().to_string(), expected);
    }
}

#[test]
fn renamed_xyz_attributes_metadata_and_concurrent_calls_are_preserved() {
    let (source, b) = fixture_pair(4103, 1);
    let mut schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    for (field, name) in source.schema().fields().iter().zip(["px", "py", "pz"]) {
        schema = schema.with_field(PointField::scalar(name, field.semantic, field.dtype));
        buffers.insert(name, source.field(&field.name).unwrap().clone());
    }
    let id = PointBuffer::U32((0..source.len()).map(|i| i as u32).collect());
    schema = schema.with_field(PointField::scalar("id", FieldSemantic::Unknown, DType::U32));
    buffers.insert("id", id.clone());
    let a = PointCloud::try_from_parts(
        schema,
        buffers,
        SpatialMetadata::new("sensor", Timestamp::from_nanos(37)),
    )
    .unwrap();
    let input_before = a.clone();
    let expected = previous_words(&a, &b);
    std::thread::scope(|scope| {
        let calls: Vec<_> = [(2, 6), (6, 2)]
            .into_iter()
            .map(|(wa, wb)| {
                let (a, b) = (&a, &b);
                scope.spawn(move || cloud_distances_with_workers(a, b, wa, wb).unwrap())
            })
            .collect();
        for call in calls {
            assert_eq!(words(call.join().unwrap()), expected);
        }
    });
    assert_eq!(a, input_before);
    assert_eq!(a.field("id").unwrap(), &id);
}

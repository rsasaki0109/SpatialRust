#![cfg(all(feature = "register-fpfh-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointCloud, PointField, PointSchema,
    SpatialMetadata,
};
use spatialrust_gpu::{GpuFpfhInput, WgpuRuntime};
use spatialrust_registration::{fpfh_descriptors, fpfh_descriptors_gpu};

pub fn fixture(n: usize, kind: usize) -> PointCloud {
    let mut columns: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
    for i in 0..n {
        let j = if kind == 2 { i / 2 } else { i };
        let x = (j % 32) as f32 * 0.125 - 2.0;
        let y = (j / 32) as f32 * 0.125 - 1.0;
        let z = if kind == 0 { 0.0 } else { 0.02 * (x * x + y * y) };
        let nx = if kind == 0 { 0.0 } else { -0.04 * x };
        let ny = if kind == 0 { 0.0 } else { -0.04 * y };
        let len = (nx * nx + ny * ny + 1.0).sqrt();
        let normal = if kind == 3 {
            match j % 6 {
                0 => [0.0, 0.0, 1.0],
                1 => [0.0, 1.0, 0.0],
                2 => [1.0, 0.0, 0.0],
                3 => [-1.0, 0.0, 0.0],
                4 => [0.0, -1.0, 0.0],
                _ => [0.0, 0.0, -1.0],
            }
        } else {
            [nx / len, ny / len, 1.0 / len]
        };
        for (axis, value) in columns.iter_mut().zip([x, y, z, normal[0], normal[1], normal[2]]) {
            axis.push(value);
        }
    }
    cloud_from_columns(columns)
}

pub fn cloud_from_columns(columns: [Vec<f32>; 6]) -> PointCloud {
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
    .zip(columns)
    {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
        buffers.insert(name, PointBuffer::F32(values));
    }
    PointCloud::try_from_parts(schema, buffers, SpatialMetadata::default()).unwrap()
}

pub fn compare(actual: &[[f32; 33]], expected: &[[f32; 33]], tolerance: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0_f32;
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        for (bin, (&a, &b)) in a.iter().zip(b).enumerate() {
            assert!(a.is_finite());
            let error = (a - b).abs();
            max = max.max(error);
            assert!(error <= tolerance, "row={i} bin={bin} gpu={a} cpu={b} error={error}");
        }
    }
    max
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_fpfh_matches_cpu_curved_planar_duplicate_and_partial_clouds() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut words = 0;
    let mut max_error = 0.0_f32;
    for n in [0, 1, 2, 17, 63, 64, 65, 257, 4097] {
        for kind in 0..4 {
            let cloud = fixture(n, kind);
            for radius in [0.16, 0.35] {
                let expected = fpfh_descriptors(&cloud, radius).unwrap();
                let actual = fpfh_descriptors_gpu(&runtime, &cloud, radius, 128).unwrap();
                max_error = max_error.max(compare(&actual.descriptors, &expected, 0.002));
                words += n * 33;
                let tree = spatialrust_search::KdTree::from_point_cloud(&cloud).unwrap();
                let nodes = tree.layout().nodes().len();
                assert_eq!(
                    actual.transfers.host_to_device_bytes(),
                    if n == 0 { 0 } else { (n * 64 + nodes * 32) as u64 }
                );
                assert_eq!(
                    actual.transfers.device_to_host_bytes(),
                    if n == 0 { 0 } else { (n * 132 + 16) as u64 }
                );
            }
        }
    }
    eprintln!(
        "GPU_FPFH verified {words} descriptor values; maximum absolute CPU difference {max_error}"
    );
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_fpfh_has_independent_planar_and_degenerate_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let pair = fixture(2, 0);
    let output = fpfh_descriptors_gpu(&runtime, &pair, 0.16, 2).unwrap();
    for descriptor in output.descriptors {
        for (bin, value) in descriptor.into_iter().enumerate() {
            assert_eq!(value, if bin % 11 == 5 { 200.0 } else { 0.0 });
        }
    }
    for cloud in [
        fixture(1, 0),
        cloud_from_columns([
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![1.0; 2],
        ]),
        cloud_from_columns([
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0, 0.125],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![1.0; 2],
        ]),
    ] {
        assert!(fpfh_descriptors_gpu(&runtime, &cloud, 0.2, 2)
            .unwrap()
            .descriptors
            .iter()
            .flatten()
            .all(|&v| v == 0.0));
    }
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_fpfh_keeps_cpu_validation_and_rejects_capacity_overflow() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(65, 1);
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert_eq!(
            fpfh_descriptors(&cloud, radius).unwrap_err().to_string(),
            fpfh_descriptors_gpu(&runtime, &cloud, radius, 128).unwrap_err().to_string()
        );
    }
    for (field, value) in [("px", f32::NAN), ("ny", f32::INFINITY), ("nz", 2.0)] {
        let (schema, mut buffers, metadata) = cloud.clone().into_parts();
        let mut values = vec![0.0; 65];
        values[0] = value;
        buffers.insert(field, PointBuffer::F32(values));
        let malformed = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
        assert_eq!(
            fpfh_descriptors(&malformed, 0.2).unwrap_err().to_string(),
            fpfh_descriptors_gpu(&runtime, &malformed, 0.2, 128).unwrap_err().to_string()
        );
    }
    for capacity in [0, 1, 3] {
        let error = fpfh_descriptors_gpu(&runtime, &cloud, 0.35, capacity).unwrap_err().to_string();
        assert!(error.contains("incomplete/malformed"), "{error}");
    }
    let full = fpfh_descriptors_gpu(&runtime, &cloud, 0.35, 128).unwrap();
    compare(&full.descriptors, &fpfh_descriptors(&cloud, 0.35).unwrap(), 0.002);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_fpfh_reuses_resident_neighborhoods_with_owned_outputs_and_runtime_guards() {
    use spatialrust_core::{HasNormals3, HasPositions3};
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cloud = fixture(257, 1);
    let (x, y, z) = cloud.positions3().unwrap();
    let (nx, ny, nz) = cloud.normals3().unwrap();
    let tree = spatialrust_search::KdTree::from_slices(x, y, z);
    let normals: Vec<_> = (0..nx.len()).map(|i| [nx[i], ny[i], nz[i]]).collect();
    let input = GpuFpfhInput::upload(&runtime, &tree, &normals).unwrap();
    let neighbors = input.radius_neighborhoods_resident(&runtime, 0.35, 128).unwrap();
    let expected = fpfh_descriptors(&cloud, 0.35).unwrap();
    let live = input.descriptors_resident(&runtime, &neighbors).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (runtime, input, neighbors, expected) = (&runtime, &input, &neighbors, &expected);
            scope.spawn(move || {
                for _ in 0..2 {
                    compare(
                        &input
                            .descriptors_resident(runtime, neighbors)
                            .unwrap()
                            .download(runtime)
                            .unwrap()
                            .0,
                        expected,
                        0.002,
                    );
                }
            });
        }
    });
    let other = WgpuRuntime::new_headless().unwrap();
    assert!(input.descriptors_resident(&other, &neighbors).is_err());
    assert!(live.download(&other).is_err());
    let second = GpuFpfhInput::upload(&runtime, &tree, &normals).unwrap();
    assert!(second.descriptors_resident(&runtime, &neighbors).is_err());
    drop(input);
    drop(neighbors);
    drop(tree);
    compare(&live.download(&runtime).unwrap().0, &expected, 0.002);
}

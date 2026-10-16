use crate::{
    gather_voxel_first_xyz_and_average_multi_gpu, gather_voxel_first_xyz_and_multi_gpu,
    reduce_voxel_centroids_xyz_and_average_multi_gpu,
    reduce_voxel_centroids_xyz_and_gather_first_multi_gpu, GpuVoxelSegments, WgpuRuntime,
};
use spatialrust_core::{SpatialError, SpatialResult};

#[path = "voxel_mixed_readback_reference.rs"]
mod parent;

type Output = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>);

fn floats(out: &Output) -> impl Iterator<Item = &Vec<f32>> {
    [&out.0, &out.1, &out.2].into_iter().chain(&out.3)
}

fn assert_output(actual: &Output, expected: &Output) {
    assert_eq!(actual.3.len(), expected.3.len());
    for (channel, (a, b)) in floats(actual).zip(floats(expected)).enumerate() {
        assert_eq!(a.len(), b.len());
        for (cell, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "channel {channel}, cell {cell}");
        }
    }
    assert_eq!(actual.4, expected.4);
}

fn assert_capacity(out: &Output, cells: usize, attributes: usize, bytes: usize, old: bool) {
    let channels = 3 + attributes;
    let float_capacity: usize = floats(out).map(Vec::capacity).sum();
    assert_eq!(float_capacity, cells * if old { 2 * channels - 1 } else { channels });
    assert_eq!(out.4.iter().map(Vec::capacity).sum::<usize>(), cells * bytes);
    assert_eq!(out.3.len(), attributes);
    assert_eq!(out.4.len(), bytes);
}

fn words(len: usize) -> Vec<u32> {
    const SPECIAL: [u32; 12] = [
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_9876,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    (0..len)
        .map(|i| {
            SPECIAL.get(i).copied().unwrap_or_else(|| {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(11) ^ 0x85eb_ca6b
            })
        })
        .collect()
}

fn staging(r: &WgpuRuntime, input: &[u32]) -> wgpu::Buffer {
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-mixed-readback-fixture"),
        size: std::mem::size_of_val(input) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(input));
    r.queue().submit([]);
    buffer
}

fn parent_read(r: &WgpuRuntime, buffer: &wgpu::Buffer, n: usize, a: usize, u: usize) -> Output {
    let (flat, raw) =
        parent::parent_read_staging_f32_and_u8(r.device(), buffer, n * (3 + a), n * u * 4).unwrap();
    // Parent public calls skip empty cells and absent byte attributes before this cast.
    let bytes = if n == 0 || u == 0 {
        Vec::new()
    } else {
        parent::parent_unpack_u8_outputs_from_u32_staging(raw, n, u)
    };
    let (x, y, z, attributes) = parent::parent_split_xyz_and_attribute_blocks(flat, a, n);
    (x, y, z, attributes, parent::parent_split_u8_channel_blocks(bytes, u, n))
}

#[test]
fn mapped_voxel_mixed_preserves_float_bits_u8_low_bytes_padding_and_owned_planes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for (a, u) in [(0, 0), (1, 0), (4, 0), (7, 0), (0, 1), (0, 3), (1, 1), (4, 3)] {
        for n in [0, 1, 17, 65_537, 262_145] {
            let input = words(n * (3 + a + u) + 16);
            let buffer = staging(&r, &input);
            let planes: Vec<Vec<f32>> = (0..3 + a)
                .map(|c| input[c * n..(c + 1) * n].iter().map(|&w| f32::from_bits(w)).collect())
                .collect();
            let expected = (
                planes[0].clone(),
                planes[1].clone(),
                planes[2].clone(),
                planes[3..].to_vec(),
                (0..u)
                    .map(|c| {
                        input[(3 + a + c) * n..(4 + a + c) * n]
                            .iter()
                            .map(|w| (w & 255) as u8)
                            .collect()
                    })
                    .collect(),
            );
            let mut live =
                crate::readback::read_staging_xyz_and_attributes(r.device(), &buffer, n, a, u)
                    .unwrap();
            let old = parent_read(&r, &buffer, n, a, u);
            assert_output(&live, &expected);
            assert_output(&old, &expected);
            assert_capacity(&live, n, a, u, false);
            assert_capacity(&old, n, a, u, true);
            if n > 0 {
                live.0[0] = f32::from_bits(0xdead_beef);
                if u > 0 {
                    live.4[0][0] ^= 255;
                }
            }
            let next =
                crate::readback::read_staging_xyz_and_attributes(r.device(), &buffer, n, a, u)
                    .unwrap();
            drop(buffer);
            assert_output(&next, &expected);
            if n > 0 {
                assert_eq!(live.0[0].to_bits(), 0xdead_beef);
                if u > 0 {
                    assert_eq!(live.4[0][0], expected.4[0][0] ^ 255);
                }
            }
            comparisons += 3 * n * (3 + a + u);
        }
    }
    eprintln!("VOXEL_MIXED_RAW verified {comparisons} exact float/byte elements");
}

#[test]
fn mapped_voxel_mixed_unmaps_on_float_and_byte_bounds_panics() {
    let r = WgpuRuntime::new_headless().unwrap();
    let input = words(96);
    let buffer = staging(&r, &input);
    for (n, a, u) in [(33, 0, 0), (16, 1, 3)] {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::readback::read_staging_xyz_and_attributes(r.device(), &buffer, n, a, u).unwrap();
        }));
        assert!(panic.is_err());
        let valid = crate::readback::read_staging_xyz_and_attributes(r.device(), &buffer, 16, 1, 2)
            .unwrap();
        assert_output(&valid, &parent_read(&r, &buffer, 16, 1, 2));
    }
}

struct Fixture {
    floats: Vec<Vec<f32>>,
    bytes: Vec<Vec<u8>>,
    xyz: [wgpu::Buffer; 3],
    segments: GpuVoxelSegments,
    cells: usize,
    occupancy: usize,
}

fn fixture(r: &WgpuRuntime, n: usize, a: usize, u: usize, occupancy: usize) -> Fixture {
    let points = n * occupancy;
    let floats: Vec<Vec<f32>> = (0..3 + a)
        .map(|c| (0..points).map(|i| ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5).collect())
        .collect();
    let bytes = (0..u)
        .map(|c| (0..points).map(|i| ((i % n * 13 + c * 17 + i / n * 20) % 256) as u8).collect())
        .collect();
    let xyz = std::array::from_fn(|c| {
        r.upload_f32_storage("voxel-mixed-xyz", if points == 0 { &[0.0] } else { &floats[c] })
            .unwrap()
    });
    let indices: Vec<u32> =
        (0..n).flat_map(|cell| (0..occupancy).map(move |lane| (lane * n + cell) as u32)).collect();
    let starts: Vec<u32> = (0..n).map(|cell| (cell * occupancy) as u32).collect();
    let keys: Vec<u32> = (0..n).flat_map(|cell| [cell as u32, 0, 0, 0]).collect();
    let upload = |name, values: &[u32]| {
        r.upload_u32_storage(name, if values.is_empty() { &[0] } else { values }).unwrap()
    };
    let segments = GpuVoxelSegments::new(
        n as u32,
        points as u32,
        upload("voxel-mixed-keys", &keys),
        upload("voxel-mixed-indices", &indices),
        upload("voxel-mixed-starts", &starts),
        0,
    );
    Fixture { floats, bytes, xyz, segments, cells: n, occupancy }
}

fn expected(f: &Fixture, mode: usize) -> Output {
    let n = f.cells;
    let mean_xyz = mode >= 2;
    let mean_attrs = mode == 1 || mode == 2;
    let planes: Vec<Vec<f32>> = f
        .floats
        .iter()
        .enumerate()
        .map(|(c, v)| {
            (0..n)
                .map(|cell| {
                    if if c < 3 { mean_xyz } else { mean_attrs } {
                        (0..f.occupancy).map(|lane| v[lane * n + cell]).sum::<f32>()
                            * (1.0 / f.occupancy as f32)
                    } else {
                        v[cell]
                    }
                })
                .collect()
        })
        .collect();
    let bytes = f
        .bytes
        .iter()
        .map(|v| {
            (0..n)
                .map(|cell| {
                    if mean_attrs {
                        let sum: u32 = (0..f.occupancy).map(|lane| v[lane * n + cell] as u32).sum();
                        // These fixtures have integral means, independent of tie rounding policy.
                        assert_eq!(sum % f.occupancy as u32, 0);
                        (sum / f.occupancy as u32) as u8
                    } else {
                        v[cell]
                    }
                })
                .collect()
        })
        .collect();
    (planes[0].clone(), planes[1].clone(), planes[2].clone(), planes[3..].to_vec(), bytes)
}

fn invoke(
    r: &WgpuRuntime,
    f: &Fixture,
    attrs: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
    old: bool,
) -> SpatialResult<Output> {
    let function = match (mode, old) {
        (0, false) => gather_voxel_first_xyz_and_multi_gpu,
        (0, true) => parent::parent_gather_voxel_first_xyz_and_multi_gpu,
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => parent::parent_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => parent::parent_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (3, false) => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        (3, true) => parent::parent_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        _ => unreachable!(),
    };
    function(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attrs, bytes, &f.segments)
}

fn call(r: &WgpuRuntime, f: &Fixture, mode: usize, old: bool) -> Output {
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    invoke(r, f, &attrs, &bytes, mode, old).unwrap()
}

#[test]
fn public_voxel_mixed_keeps_frozen_analytic_modes_lifetimes_and_validation_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for n in [0, 1, 17, 257, 65_537] {
        for (a, u) in [(0, 0), (1, 0), (4, 0), (7, 0), (0, 1), (0, 3), (1, 1), (4, 3)] {
            for occupancy in [1, 4] {
                let mut f = fixture(&r, n, a, u, occupancy);
                if n == 257 {
                    let keys: Vec<_> = (0..n * occupancy).map(|i| ((i % n) as i64, 0, 0)).collect();
                    f.segments =
                        super::voxel_sort::build_voxel_segments_gpu_from_keys(&r, &keys).unwrap();
                    assert_eq!(f.segments.cell_count(), n as u32);
                    assert_eq!(f.segments.construction_readback_bytes(), 4);
                }
                for mode in 0..4 {
                    let expected = expected(&f, mode);
                    let pool = r.buffer_pool().stats();
                    let mut live = call(&r, &f, mode, false);
                    let old = call(&r, &f, mode, true);
                    assert_output(&live, &old);
                    assert_output(&live, &expected);
                    assert_capacity(&live, n, a, u, false);
                    assert_capacity(&old, n, a, u, true);
                    if n == 0 {
                        assert_eq!(r.buffer_pool().stats(), pool);
                    } else {
                        live.0[0] = f32::from_bits(0xdead_beef);
                        if u > 0 {
                            live.4[0][0] ^= 255;
                        }
                    }
                    let next = call(&r, &f, mode, false);
                    r.clear_buffer_pool();
                    assert_output(&next, &expected);
                    if n > 0 {
                        assert_eq!(live.0[0].to_bits(), 0xdead_beef);
                        if u > 0 {
                            assert_eq!(live.4[0][0], expected.4[0][0] ^ 255);
                        }
                    }
                    comparisons += 3 * n * (3 + a + u);
                }
                let live = call(&r, &f, 0, false);
                let original = expected(&f, 0);
                for plane in &mut f.floats[3..] {
                    plane.fill(123.0);
                }
                for plane in &mut f.bytes {
                    plane.fill(231);
                }
                assert_output(&call(&r, &f, 0, false), &expected(&f, 0));
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &original);
            }
        }
    }
    let f = fixture(&r, 17, 2, 2, 4);
    let attrs: Vec<_> = f.floats[3..].iter().map(Vec::as_slice).collect();
    let bytes: Vec<_> = f.bytes.iter().map(Vec::as_slice).collect();
    let mut bad_attrs = attrs.clone();
    bad_attrs[1] = &f.floats[4][..67];
    let mut bad_bytes = bytes.clone();
    bad_bytes[0] = &f.bytes[0][..66];
    for mode in 0..4 {
        for (af, bu, found) in [(&bad_attrs, &bad_bytes, 67), (&attrs, &bad_bytes, 66)] {
            let pool = r.buffer_pool().stats();
            let error = invoke(&r, &f, af, bu, mode, false).unwrap_err();
            assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found });
            assert_eq!(error, invoke(&r, &f, af, bu, mode, true).unwrap_err());
            assert_eq!(r.buffer_pool().stats(), pool);
        }
        assert_output(&call(&r, &f, mode, false), &expected(&f, mode));
        let empty = fixture(&r, 0, 2, 2, 1);
        let empty_output = expected(&empty, mode);
        assert_output(
            &invoke(&r, &empty, &bad_attrs, &bad_bytes, mode, false).unwrap(),
            &empty_output,
        );
        assert_output(
            &invoke(&r, &empty, &bad_attrs, &bad_bytes, mode, true).unwrap(),
            &empty_output,
        );
    }
    eprintln!("VOXEL_MIXED_PUBLIC verified {comparisons} frozen/analytic/repeated elements");
}

#[test]
fn public_voxel_mixed_million_cell_results_have_compact_capacity_in_all_four_modes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 1_048_577, 4, 3, 1);
    for mode in 0..4 {
        let new = call(&r, &f, mode, false);
        let old = call(&r, &f, mode, true);
        assert_output(&new, &old);
        assert_output(&new, &expected(&f, mode));
        assert_capacity(&new, f.cells, 4, 3, false);
        assert_capacity(&old, f.cells, 4, 3, true);
    }
}

#[test]
fn concurrent_voxel_mixed_returns_independent_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for (a, u) in [(4, 0), (0, 3), (4, 3)] {
            let r = &r;
            scope.spawn(move || {
                let f = fixture(r, 16_385, a, u, 4);
                let live = call(r, &f, 0, false);
                for mode in 0..4 {
                    assert_output(&call(r, &f, mode, false), &expected(&f, mode));
                }
                let original = expected(&f, 0);
                drop(f);
                r.clear_buffer_pool();
                assert_output(&live, &original);
            });
        }
    });
}

#[path = "voxel_u8_packed_tests.rs"]
mod packed;

#[path = "voxel_reduce_fused_tests.rs"]
mod fused_reduce;

#[path = "voxel_first_average_tests.rs"]
mod fused_first_average;

#[path = "voxel_centroid_first_tests.rs"]
mod fused_centroid_first;

#[path = "voxel_first_gather_tests.rs"]
mod fused_first_gather;

#[path = "voxel_u8_batch_tests.rs"]
mod voxel_u8_batch_tests;

#[path = "voxel_combined_average_tests.rs"]
mod combined_average;

#[path = "voxel_packed_gather_tests.rs"]
mod packed_gather;

#[path = "voxel_combined_xyz_tests.rs"]
mod packed_xyz;

#[path = "voxel_wide_average_tests.rs"]
mod wide_average;

#[path = "voxel_wide_first_tests.rs"]
mod wide_first;

#[path = "voxel_parallel_readback_tests.rs"]
mod parallel_attributes;

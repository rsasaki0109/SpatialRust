use super::voxel_gather::record_voxel_gather_xyz_pass;
use super::voxel_reduce::record_voxel_reduce_xyz_pass;
use crate::{
    gather_voxel_first_xyz_gpu_buffers, reduce_voxel_centroids_xyz_gpu_buffers, GpuVoxelSegments,
    WgpuRuntime,
};
use spatialrust_core::{SpatialError, SpatialResult};

#[path = "voxel_xyz_readback_reference.rs"]
mod parent;

type Xyz = (Vec<f32>, Vec<f32>, Vec<f32>);

fn axes(xyz: &Xyz) -> [&[f32]; 3] {
    [&xyz.0, &xyz.1, &xyz.2]
}

fn assert_bits(actual: &Xyz, expected: &Xyz) {
    for (axis, (actual, expected)) in axes(actual).into_iter().zip(axes(expected)).enumerate() {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "axis {axis}, cell {i}");
        }
    }
}

fn words(n: usize) -> Vec<u32> {
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
    (0..n)
        .map(|i| {
            if i < SPECIAL.len() {
                SPECIAL[i]
            } else {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(11) ^ 0x85eb_ca6b
            }
        })
        .collect()
}

fn staging(r: &WgpuRuntime, words: &[u32]) -> wgpu::Buffer {
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-xyz-raw-readback-fixture"),
        size: std::mem::size_of_val(words) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(words));
    r.queue().submit([]);
    buffer
}

#[test]
fn mapped_xyz_preserves_all_float_bits_plane_lengths_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for n in [0, 1, 17, 256, 65_537, 262_145] {
        let input = words(n * 3 + 16);
        let buffer = staging(&r, &input);
        let expected = (
            input[..n].iter().map(|&x| f32::from_bits(x)).collect(),
            input[n..n * 2].iter().map(|&x| f32::from_bits(x)).collect(),
            input[n * 2..n * 3].iter().map(|&x| f32::from_bits(x)).collect(),
        );
        let mut live = crate::readback::read_staging_xyz(r.device(), &buffer, n).unwrap();
        assert_bits(&live, &expected);
        let old = parent::parent_read_staging_f32(r.device(), &buffer, n * 3).unwrap();
        let old = parent::parent_split_xyz_blocks(old, n);
        assert_bits(&old, &expected);
        assert_eq!(live.0.capacity() + live.1.capacity() + live.2.capacity(), n * 3);
        assert_eq!(old.0.capacity() + old.1.capacity() + old.2.capacity(), n * 5);
        if n > 0 {
            live.0[0] = f32::from_bits(0xdead_beef);
        }
        let next = crate::readback::read_staging_xyz(r.device(), &buffer, n).unwrap();
        drop(buffer);
        assert_bits(&next, &expected);
        if n > 0 {
            assert_eq!(live.0[0].to_bits(), 0xdead_beef);
        }
        comparisons += n * 9;
    }
    eprintln!("VOXEL_XYZ_RAW verified {comparisons} exact float words");
}

#[test]
fn mapped_xyz_unmaps_on_decoder_panic_and_allows_subsequent_maps() {
    let r = WgpuRuntime::new_headless().unwrap();
    let input = words(48);
    let buffer = staging(&r, &input);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::readback::read_staging_xyz(r.device(), &buffer, 17).unwrap();
    }));
    assert!(panic.is_err());
    let result = crate::readback::read_staging_xyz(r.device(), &buffer, 16).unwrap();
    for (actual, expected) in axes(&result).into_iter().flatten().zip(&input) {
        assert_eq!(actual.to_bits(), *expected);
    }
    assert_eq!(parent::parent_read_staging_f32(r.device(), &buffer, 3).unwrap().len(), 3);
}

struct Fixture {
    buffers: [wgpu::Buffer; 3],
    segments: GpuVoxelSegments,
    first: Xyz,
    mean: Xyz,
}

fn fixture(r: &WgpuRuntime, cells: usize, occupancy: usize) -> Fixture {
    let points = cells * occupancy;
    let source: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..points).map(|i| ((i * 17 + axis * 29) % 4096) as f32 * 0.25 - 511.5).collect()
    });
    let buffers = std::array::from_fn(|axis| {
        r.upload_f32_storage("voxel-xyz-input", if points == 0 { &[0.0] } else { &source[axis] })
            .unwrap()
    });
    let indices: Vec<u32> = (0..cells)
        .flat_map(|cell| (0..occupancy).map(move |lane| (lane * cells + cell) as u32))
        .collect();
    let starts: Vec<u32> = (0..cells).map(|cell| (cell * occupancy) as u32).collect();
    let keys: Vec<u32> = (0..cells).flat_map(|cell| [cell as u32, 0, 0, 0]).collect();
    let upload = |name, data: &[u32]| {
        r.upload_u32_storage(name, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    let segments = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        upload("voxel-xyz-keys", &keys),
        upload("voxel-xyz-indices", &indices),
        upload("voxel-xyz-starts", &starts),
        0,
    );
    let first: [Vec<f32>; 3] = std::array::from_fn(|axis| source[axis][..cells].to_vec());
    let mean: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..cells)
            .map(|cell| {
                let sum: f32 = (0..occupancy).map(|lane| source[axis][lane * cells + cell]).sum();
                sum * (1.0 / occupancy as f32)
            })
            .collect()
    });
    let [fx, fy, fz] = first;
    let [mx, my, mz] = mean;
    Fixture { buffers, segments, first: (fx, fy, fz), mean: (mx, my, mz) }
}

fn call(r: &WgpuRuntime, f: &Fixture, reduce: bool, old: bool) -> SpatialResult<Xyz> {
    let [x, y, z] = &f.buffers;
    match (reduce, old) {
        (false, false) => gather_voxel_first_xyz_gpu_buffers(r, x, y, z, &f.segments),
        (false, true) => parent::parent_gather_voxel_first_xyz_gpu_buffers(r, x, y, z, &f.segments),
        (true, false) => reduce_voxel_centroids_xyz_gpu_buffers(r, x, y, z, &f.segments),
        (true, true) => {
            parent::parent_reduce_voxel_centroids_xyz_gpu_buffers(r, x, y, z, &f.segments)
        }
    }
}

#[test]
fn public_voxel_xyz_calls_match_frozen_and_analytic_results_keep_empty_and_live_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut comparisons = 0;
    for cells in [0, 1, 17, 257, 65_537] {
        for occupancy in [1, 2, 4] {
            let mut f = fixture(&r, cells, occupancy);
            if cells == 257 {
                let keys: Vec<_> =
                    (0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect();
                f.segments =
                    super::voxel_sort::build_voxel_segments_gpu_from_keys(&r, &keys).unwrap();
                assert_eq!(f.segments.cell_count(), cells as u32);
                assert_eq!(f.segments.construction_readback_bytes(), 4);
            }
            for reduce in [false, true] {
                let expected = if reduce { &f.mean } else { &f.first };
                let before = r.buffer_pool().stats();
                let mut new = call(&r, &f, reduce, false).unwrap();
                let old = call(&r, &f, reduce, true).unwrap();
                assert_bits(&new, &old);
                assert_bits(&new, expected);
                if cells == 0 {
                    assert_eq!(r.buffer_pool().stats(), before);
                } else {
                    assert_eq!(new.0.capacity(), cells);
                    assert_eq!(old.0.capacity(), cells * 3);
                    new.0[0] = f32::from_bits(0xdead_beef);
                }
                let next = call(&r, &f, reduce, false).unwrap();
                r.clear_buffer_pool();
                assert_bits(&next, expected);
                if cells > 0 {
                    assert_eq!(new.0[0].to_bits(), 0xdead_beef);
                }
                comparisons += cells * 9;
            }
            let live = call(&r, &f, false, false).unwrap();
            let expected = f.first.clone();
            drop(f);
            r.clear_buffer_pool();
            assert_bits(&live, &expected);
        }
    }
    eprintln!("VOXEL_XYZ_PUBLIC verified {comparisons} frozen/analytic/repeated float words");
}

#[test]
fn concurrent_voxel_xyz_calls_keep_outputs_and_staging_independent() {
    let r = WgpuRuntime::new_headless().unwrap();
    std::thread::scope(|scope| {
        for occupancy in [1, 2, 4] {
            let r = &r;
            scope.spawn(move || {
                let f = fixture(r, 65_537, occupancy);
                let live = call(r, &f, false, false).unwrap();
                for _ in 0..3 {
                    assert_bits(&call(r, &f, false, false).unwrap(), &f.first);
                    assert_bits(&call(r, &f, true, false).unwrap(), &f.mean);
                }
                let expected = f.first.clone();
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&live, &expected);
            });
        }
    });
}

#[path = "voxel_xyz_readback_benchmark.rs"]
mod benchmark;

#[path = "voxel_xyz_packed_tests.rs"]
mod packed;

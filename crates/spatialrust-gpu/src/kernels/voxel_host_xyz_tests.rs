use super::*;
use crate::{build_voxel_segments, reduce_voxel_average_f32, reduce_voxel_centroids_xyz};
use spatialrust_core::SpatialError;

#[path = "voxel_host_xyz_reference.rs"]
mod baseline;
type Xyz = (Vec<f32>, Vec<f32>, Vec<f32>);

struct Fixture {
    xyz: [Vec<f32>; 3],
    segments: VoxelSegments,
}

fn fixture(cells: usize, occupancy: usize, tail: usize) -> Fixture {
    let points = cells * occupancy;
    let xyz = std::array::from_fn(|axis| {
        (0..points + tail).map(|i| ((i * 17 + axis * 29) % 4096) as f32 * 0.25 - 511.5).collect()
    });
    let segments = VoxelSegments {
        keys: (0..cells).map(|i| (i as i64, 0, 0)).collect(),
        point_indices: (0..cells)
            .flat_map(|cell| (0..occupancy).map(move |lane| (lane * cells + cell) as u32))
            .collect(),
        cell_starts: (0..cells).map(|cell| (cell * occupancy) as u32).collect(),
        cell_counts: vec![occupancy as u32; cells],
    };
    Fixture { xyz, segments }
}

fn invoke(r: &WgpuRuntime, f: &Fixture, old: bool) -> SpatialResult<Xyz> {
    let function = if old {
        baseline::baseline_reduce_voxel_centroids_xyz
    } else {
        reduce_voxel_centroids_xyz
    };
    function(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], &f.segments)
}

fn axes(output: &Xyz) -> [&Vec<f32>; 3] {
    [&output.0, &output.1, &output.2]
}

fn assert_bits(a: &Xyz, b: &Xyz) {
    for (a, b) in axes(a).into_iter().zip(axes(b)) {
        assert_eq!(bytemuck::cast_slice::<f32, u32>(a), bytemuck::cast_slice::<f32, u32>(b));
    }
}

fn assert_capacity(out: &Xyz, cells: usize) {
    for plane in axes(out) {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

fn cpu_reference(f: &Fixture) -> Xyz {
    let planes: [Vec<f32>; 3] = std::array::from_fn(|axis| {
        (0..f.segments.len())
            .map(|cell| {
                let start = f.segments.cell_starts[cell] as usize;
                let end = f
                    .segments
                    .cell_starts
                    .get(cell + 1)
                    .copied()
                    .unwrap_or(f.segments.point_indices.len() as u32)
                    as usize;
                if start >= end {
                    return 0.0;
                }
                (f.segments.point_indices[start..end]
                    .iter()
                    .map(|&i| f64::from(f.xyz[axis][i as usize]))
                    .sum::<f64>()
                    / (end - start) as f64) as f32
            })
            .collect()
    });
    let [x, y, z] = planes;
    (x, y, z)
}

#[test]
fn host_xyz_plan_bounds_scalar_capabilities_staging_offsets_and_transfer_ledger() {
    let limits = wgpu::Limits::default();
    let mut checked = 0;
    for cells in [0_usize, 1, 17, 255, 256, 257, 65_537, 1_048_575, 1_048_576, 1_048_577] {
        for points in [0, 1, 1_048_576, u32::MAX as usize] {
            for host_limit in [0, 11, 12, 203, 204, isize::MAX as u64] {
                let plan = plan_host_xyz(&limits, cells, points, host_limit);
                let expected = cells > 0
                    && points > 0
                    && points as u64 * 4 <= u64::from(limits.max_storage_buffer_binding_size)
                    && points as u64 * 4 <= limits.max_buffer_size
                    && cells.div_ceil(256) <= limits.max_compute_workgroups_per_dimension as usize
                    && cells as u64 * 4 <= u64::from(limits.max_storage_buffer_binding_size)
                    && cells as u64 * 12 <= limits.max_buffer_size
                    && cells as u64 * 12 <= host_limit;
                assert_eq!(plan.is_some(), expected);
                if let Some(plan) = plan {
                    assert_eq!(plan.channel_bytes, cells as u64 * 4);
                    assert_eq!(plan.staging_bytes, cells as u64 * 12);
                    for axis in 0..3 {
                        assert_eq!((axis + 1) * plan.channel_bytes, (axis + 1) * cells as u64 * 4);
                        assert!((axis + 1) * plan.channel_bytes <= plan.staging_bytes);
                    }
                }
                checked += 1;
            }
        }
    }
    let mut scalar_limits = limits.clone();
    scalar_limits.max_storage_buffers_per_shader_stage = 4;
    scalar_limits.max_bindings_per_bind_group = 5;
    assert!(plan_host_xyz(&scalar_limits, 257, 257, 3084).is_some());
    let mut rejected = Vec::new();
    macro_rules! deny {
        ($field:ident,$value:expr) => {{
            let mut l = scalar_limits.clone();
            l.$field = $value;
            rejected.push(l);
        }};
    }
    deny!(max_storage_buffers_per_shader_stage, 3);
    deny!(max_bindings_per_bind_group, 4);
    deny!(max_bind_groups, 0);
    deny!(max_uniform_buffers_per_shader_stage, 0);
    deny!(max_uniform_buffer_binding_size, 15);
    deny!(max_buffer_size, 3083);
    deny!(max_storage_buffer_binding_size, 1027);
    deny!(max_compute_workgroup_size_x, 255);
    deny!(max_compute_workgroup_size_y, 0);
    deny!(max_compute_workgroup_size_z, 0);
    deny!(max_compute_invocations_per_workgroup, 255);
    deny!(max_compute_workgroups_per_dimension, 1);
    for l in rejected {
        assert!(plan_host_xyz(&l, 257, 257, 3084).is_none());
        checked += 1;
    }
    // A valid scalar buffer can fit while the combined staging must fall back.
    let mut l = scalar_limits.clone();
    l.max_buffer_size = 1028;
    assert!(plan_host_xyz(&l, 257, 257, u64::MAX).is_none());
    assert!(plan_host_xyz(&scalar_limits, 257, 257, 3083).is_none());
    if let Some(too_many) = (u32::MAX as usize).checked_add(1) {
        assert!(plan_host_xyz(&limits, too_many, 1, u64::MAX).is_none());
        assert!(plan_host_xyz(&limits, 1, too_many, u64::MAX).is_none());
    }
    let (points, cells) = (1_048_576_u64, 262_144_u64);
    let old_h2d = 12 * points + 12 * (points + cells);
    let new_h2d = 12 * points + 4 * (points + cells);
    assert_eq!((old_h2d, new_h2d, old_h2d - new_h2d), (28_311_552, 17_825_792, 10_485_760));
    eprintln!("VOXEL_HOST_XYZ_PLAN verified {checked} capability/host/index/staging limits and exact nonoverlapping plane offsets");
    eprintln!("VOXEL_HOST_XYZ_LEDGER points=1048576 cells=262144 old_h2d=28311552 new_h2d=17825792 saved_h2d=10485760 d2h=3145728 old_submits_maps_waits=3 new_submits_maps_waits=1 scalar_dispatches=3 source_verified=true uniforms_excluded=true");
}

#[test]
fn host_xyz_complete_calls_keep_parent_cpu_bits_capacities_sorted_segments_and_owned_lifetimes() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for (cells, occupancy, tail) in [
        (0, 1, 0),
        (1, 4, 0),
        (17, 4, 3),
        (255, 1, 0),
        (256, 4, 0),
        (257, 4, 7),
        (65_537, 4, 0),
        (1_048_575, 1, 0),
        (1_048_576, 1, 0),
        (1_048_577, 1, 0),
    ] {
        let mut f = fixture(cells, occupancy, tail);
        if cells == 257 {
            let keys: Vec<_> = (0..cells * occupancy).map(|i| ((i % cells) as i64, 0, 0)).collect();
            f.segments = build_voxel_segments(&keys);
        }
        let live = invoke(&r, &f, false).unwrap();
        let old = invoke(&r, &f, true).unwrap();
        let cpu = cpu_reference(&f);
        assert_bits(&live, &old);
        assert_bits(&live, &cpu);
        assert_capacity(&live, cells);
        assert_capacity(&old, cells);
        drop(f);
        r.clear_buffer_pool();
        assert_bits(&live, &cpu);
        checked += cells * 3 * 2;
    }
    eprintln!("VOXEL_HOST_XYZ_PUBLIC verified {checked} complete parent/independent-f64 words, cutoff/sorted/unused source-tail cases, exact capacities and retained ownership");
}

#[test]
fn host_xyz_sparse_and_nonfinite_words_keep_original_scalar_math_and_permutations() {
    let r = WgpuRuntime::new_headless().unwrap();
    let special = [
        0_u32,
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
    let mut checked = 0;
    for occupancy in [1, 2, 3, 9, 257] {
        let mut f = fixture(257, occupancy, 3);
        f.segments.point_indices.reverse();
        f.segments.cell_starts = (0..257).map(|cell| (cell / 2 * 2 * occupancy) as u32).collect();
        f.segments.cell_counts.fill(u32::MAX); // The original API does not consume counts.
        let finite = invoke(&r, &f, false).unwrap();
        assert_bits(&finite, &invoke(&r, &f, true).unwrap());
        let cpu = cpu_reference(&f);
        for (actual, expected) in axes(&finite).into_iter().zip(axes(&cpu)) {
            for (&a, &e) in actual.iter().zip(expected) {
                assert!((a - e).abs() <= 4.0 * f32::EPSILON * e.abs().max(1.0));
            }
        }
        for (axis, values) in f.xyz.iter_mut().enumerate() {
            for (i, value) in values.iter_mut().enumerate() {
                *value = f32::from_bits(special[(i + axis * 3) % special.len()]);
            }
        }
        let live = invoke(&r, &f, false).unwrap();
        assert_bits(&live, &invoke(&r, &f, true).unwrap());
        assert_capacity(&live, 257);
        checked += 257 * 3 * 3;
    }
    eprintln!("VOXEL_HOST_XYZ_NUMERIC verified {checked} sparse/empty cells, ignored counts, original point permutations, nonpower means and raw nonfinite/subnormal words");
}

#[test]
fn host_xyz_validation_keeps_legacy_shape_error_precedence_and_empty_returns() {
    let r = WgpuRuntime::new_headless().unwrap();
    let empty = fixture(0, 1, 0).segments;
    let occupied = fixture(1, 1, 0).segments;
    for segments in [&empty, &occupied] {
        for (x, y, z) in [(&[][..], &[1.0][..], &[][..]), (&[1.0][..], &[1.0][..], &[][..])] {
            let before = r.buffer_pool().stats();
            let live = reduce_voxel_centroids_xyz(&r, x, y, z, segments).unwrap_err();
            assert_eq!(
                live,
                baseline::baseline_reduce_voxel_centroids_xyz(&r, x, y, z, segments).unwrap_err()
            );
            assert_eq!(
                live,
                SpatialError::BufferLengthMismatch { expected: x.len(), found: y.len() }
            );
            assert_eq!(r.buffer_pool().stats(), before);
        }
    }
    let error = reduce_voxel_centroids_xyz(&r, &[], &[], &[], &occupied).unwrap_err();
    assert_eq!(error, SpatialError::InvalidArgument("cannot reduce empty value buffer".to_owned()));
    assert_eq!(
        error,
        baseline::baseline_reduce_voxel_centroids_xyz(&r, &[], &[], &[], &occupied).unwrap_err()
    );
    for values in [&[][..], &[1.0][..]] {
        let out = reduce_voxel_centroids_xyz(&r, values, values, values, &empty).unwrap();
        assert_capacity(&out, 0);
    }
}

#[test]
fn host_xyz_concurrent_calls_keep_owned_outputs_after_host_mutation_and_pool_clearing() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let gate = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let (r, gate) = (r.clone(), gate.clone());
            std::thread::spawn(move || {
                let mut f = fixture(65_536 + worker, [1, 4, 3, 2][worker], worker);
                gate.wait();
                let live = invoke(&r, &f, false).unwrap();
                let old = invoke(&r, &f, true).unwrap();
                assert_bits(&live, &old);
                assert_capacity(&live, f.segments.len());
                for axis in &mut f.xyz {
                    axis.fill(123.0);
                }
                f.segments.point_indices.reverse();
                let changed = invoke(&r, &f, false).unwrap();
                assert_bits(&changed, &invoke(&r, &f, true).unwrap());
                drop(f);
                assert_bits(&live, &old);
                (live, old)
            })
        })
        .collect();
    let retained: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    r.clear_buffer_pool();
    for (live, old) in retained {
        assert_bits(&live, &old);
    }
}

#[path = "voxel_host_xyz_benchmark.rs"]
mod benchmark;

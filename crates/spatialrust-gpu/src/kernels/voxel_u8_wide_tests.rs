use super::*;
use crate::{
    gather_voxel_first_xyz_and_average_multi_gpu, gather_voxel_first_xyz_and_multi_gpu,
    reduce_voxel_centroids_xyz_and_average_multi_gpu,
    reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
};
use spatialrust_core::SpatialError;

#[path = "voxel_u8_wide_reference.rs"]
mod baseline;

type Output = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>);

struct Fixture {
    floats: Vec<Vec<f32>>,
    bytes: Vec<Vec<u8>>,
    xyz: [wgpu::Buffer; 3],
    segments: GpuVoxelSegments,
    starts: Vec<u32>,
    indices: Vec<u32>,
}

fn gpu_segments(r: &WgpuRuntime, indices: &[u32], starts: &[u32]) -> GpuVoxelSegments {
    let upload = |label, data: &[u32]| {
        r.upload_u32_storage(label, if data.is_empty() { &[0] } else { data }).unwrap()
    };
    GpuVoxelSegments::new(
        starts.len() as u32,
        indices.len() as u32,
        upload("wide-u8-keys", &[0]),
        upload("wide-u8-indices", indices),
        upload("wide-u8-starts", starts),
        0,
    )
}

fn fixture(r: &WgpuRuntime, cells: usize, attrs: usize, bytes: usize, occupancy: usize) -> Fixture {
    let points = cells * occupancy;
    let floats: Vec<Vec<f32>> = (0..3 + attrs)
        .map(|c| (0..points).map(|i| ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5).collect())
        .collect();
    let xyz = std::array::from_fn(|c| {
        r.upload_f32_storage("wide-u8-xyz", if points == 0 { &[0.0] } else { &floats[c] }).unwrap()
    });
    let indices = (0..cells)
        .flat_map(|cell| (0..occupancy).map(move |lane| (lane * cells + cell) as u32))
        .collect::<Vec<_>>();
    let starts = (0..cells).map(|c| (c * occupancy) as u32).collect::<Vec<_>>();
    Fixture {
        floats,
        bytes: (0..bytes)
            .map(|c| (0..points).map(|i| ((i * 4 + c * 28) % 256) as u8).collect())
            .collect(),
        xyz,
        segments: gpu_segments(r, &indices, &starts),
        indices,
        starts,
    }
}

fn invoke(
    r: &WgpuRuntime,
    f: &Fixture,
    attrs: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
    old: bool,
) -> SpatialResult<Output> {
    let fun = match (mode, old) {
        (0, false) => gather_voxel_first_xyz_and_multi_gpu,
        (0, true) => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (3, false) => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        (3, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        _ => unreachable!(),
    };
    fun(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attrs, bytes, &f.segments)
}

fn call(r: &WgpuRuntime, f: &Fixture, mode: usize, old: bool) -> Output {
    invoke(
        r,
        f,
        &f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        mode,
        old,
    )
    .unwrap()
}

fn planes(out: &Output) -> impl Iterator<Item = &Vec<f32>> {
    [&out.0, &out.1, &out.2].into_iter().chain(&out.3)
}

fn assert_output(a: &Output, b: &Output) {
    assert_eq!(a.3.len(), b.3.len());
    for (a, b) in planes(a).zip(planes(b)) {
        assert_eq!(bytemuck::cast_slice::<f32, u32>(a), bytemuck::cast_slice::<f32, u32>(b));
    }
    assert_eq!(a.4, b.4);
}

fn assert_capacity(out: &Output, cells: usize, attrs: usize, bytes: usize) {
    assert_eq!(out.3.len(), attrs);
    assert_eq!(out.4.len(), bytes);
    for plane in planes(out) {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
    for plane in &out.4 {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

fn bounds(f: &Fixture, cell: usize) -> (usize, usize) {
    (
        f.starts[cell] as usize,
        f.starts.get(cell + 1).copied().unwrap_or(f.indices.len() as u32) as usize,
    )
}

fn expected(f: &Fixture, mode: usize) -> Output {
    let float = |channel: usize, first: bool| {
        (0..f.starts.len())
            .map(|cell| {
                let (start, end) = bounds(f, cell);
                if start == end {
                    0.0
                } else if first {
                    f.floats[channel][f.indices[start] as usize]
                } else {
                    (f.indices[start..end]
                        .iter()
                        .map(|&i| f64::from(f.floats[channel][i as usize]))
                        .sum::<f64>()
                        / (end - start) as f64) as f32
                }
            })
            .collect::<Vec<_>>()
    };
    let axes: [Vec<f32>; 3] = std::array::from_fn(|c| float(c, mode < 2));
    let [x, y, z] = axes;
    (
        x,
        y,
        z,
        (3..f.floats.len()).map(|c| float(c, mode == 0 || mode == 3)).collect(),
        f.bytes
            .iter()
            .map(|v| {
                (0..f.starts.len())
                    .map(|cell| {
                        let (start, end) = bounds(f, cell);
                        if start == end {
                            0
                        } else if mode == 0 || mode == 3 {
                            v[f.indices[start] as usize]
                        } else {
                            let sum: u64 = f.indices[start..end]
                                .iter()
                                .map(|&i| u64::from(v[i as usize]))
                                .sum();
                            // Regular fixtures use multiples of four and power-of-two occupancy: no half ties.
                            (sum as f64 / (end - start) as f64).round() as u8
                        }
                    })
                    .collect()
            })
            .collect(),
    )
}

fn read_words(r: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| send.send(result).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    recv.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    result
}

fn compare_raw(r: &WgpuRuntime, f: &Fixture, average: bool, cpu: bool) -> usize {
    let cells = f.starts.len();
    let channels = f.bytes.len();
    let stride = cells.div_ceil(4);
    let sentinel = vec![0xdead_beef; 13 + stride * channels + 19];
    let staging: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
        let b = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wide-u8-guarded-staging"),
            size: (sentinel.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        r.queue().write_buffer(&b, 0, bytemuck::cast_slice(&sentinel));
        b
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (c, channel) in f.bytes.iter().enumerate() {
        let values = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wide-u8-original-input"),
            contents: &crate::readback::u8_storage_upload_contents(channel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wide-u8-original-output"),
            size: (stride * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let record = if average {
            crate::kernels::voxel_reduce::record_voxel_reduce_u8_packed_pass
        } else {
            crate::kernels::voxel_gather::record_voxel_gather_u8_packed_pass
        };
        record(
            &mut encoder,
            r,
            &values,
            f.segments.point_indices_buffer(),
            f.segments.cell_starts_buffer(),
            cells as u32,
            f.indices.len() as u32,
            &output,
        )
        .unwrap();
        encoder.copy_buffer_to_buffer(
            &output,
            0,
            &staging[0],
            ((13 + c * stride) * 4) as u64,
            (stride * 4) as u64,
        );
    }
    assert!(try_record_wide_voxel_u8_to_staging(
        &mut encoder,
        r,
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.segments,
        &staging[1],
        52,
        average
    )
    .unwrap());
    r.queue().submit(Some(encoder.finish()));
    let old = read_words(r, &staging[0]);
    let new = read_words(r, &staging[1]);
    assert_eq!(new, old);
    assert_eq!(&new[..13], &sentinel[..13]);
    assert_eq!(&new[13 + stride * channels..], &sentinel[13 + stride * channels..]);
    for (c, input) in f.bytes.iter().enumerate() {
        let packed = new[13 + c * stride..13 + (c + 1) * stride]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        assert!(packed[cells..].iter().all(|&byte| byte == 0));
        if cpu {
            for (cell, &byte) in packed[..cells].iter().enumerate() {
                let (start, end) = bounds(f, cell);
                if start == end {
                    assert_eq!(byte, 0);
                } else if !average {
                    assert_eq!(byte, input[f.indices[start] as usize]);
                } else {
                    let sum: u64 =
                        f.indices[start..end].iter().map(|&i| u64::from(input[i as usize])).sum();
                    let count = (end - start) as u64;
                    let (q, rem) = (sum / count, sum % count);
                    if 2 * rem == count {
                        assert!(u64::from(byte) == q || u64::from(byte) == q + 1);
                    } else {
                        assert_eq!(u64::from(byte), q + u64::from(2 * rem > count));
                    }
                }
            }
        }
    }
    cells * channels * if cpu { 2 } else { 1 }
}

#[test]
fn wide_voxel_u8_plan_checks_complete_regions_capabilities_host_and_index_bounds() {
    let mut limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 7,
        max_bindings_per_bind_group: 8,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_storage_size: 4096,
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffer_binding_size: 1040,
        max_buffer_size: 8192,
        ..wgpu::Limits::downlevel_defaults()
    };
    let mut checked = 0;
    for cells in [0_u32, 1, 2, 3, 4, 5, 255, 256, 257, 512, 513] {
        for channels in [0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 13, 17] {
            let stride = u64::from(cells).div_ceil(4) * 4;
            let end = 52 + stride * channels as u64;
            let want = cells > 0 && cells <= 257 && channels > 4;
            let plan = plan_wide_voxel_u8(&limits, cells, channels, end, 52, end);
            assert_eq!(plan.is_some(), want);
            if let Some(plan) = plan {
                assert_eq!(plan, WideU8Plan { channel_bytes: stride, output_bytes: stride * 4 });
                assert!(plan_wide_voxel_u8(&limits, cells, channels, end - 1, 52, end).is_none());
                assert!(plan_wide_voxel_u8(&limits, cells, channels, end, 52, end - 1).is_none());
            }
            checked += 1;
        }
    }
    let exact = plan_wide_voxel_u8(&limits, 257, 17, 4472, 52, 4472).unwrap();
    assert_eq!(exact.channel_bytes, 260);
    for bound in 0..14 {
        let mut l = limits.clone();
        match bound {
            0 => l.max_storage_buffers_per_shader_stage = 6,
            1 => l.max_bindings_per_bind_group = 7,
            2 => l.max_compute_workgroup_size_x = 255,
            3 => l.max_compute_invocations_per_workgroup = 255,
            4 => l.max_compute_workgroup_storage_size = 4095,
            5 => l.max_compute_workgroups_per_dimension = 1,
            6 => l.max_storage_buffer_binding_size = 1039,
            7 => l.max_buffer_size = 4471,
            8 => l.max_bind_groups = 0,
            9 => l.max_uniform_buffers_per_shader_stage = 0,
            10 => l.max_uniform_buffer_binding_size = 15,
            11 => l.max_compute_workgroup_size_y = 0,
            12 => l.max_compute_workgroup_size_z = 0,
            _ => l.max_buffer_size = 15,
        }
        assert!(plan_wide_voxel_u8(&l, 257, 17, 4472, 52, 4472).is_none());
        checked += 1;
    }
    for offset in [1, 2, 3, 53, u64::MAX - 3] {
        assert!(plan_wide_voxel_u8(&limits, 257, 17, 8192, offset, 8192).is_none());
        checked += 1;
    }
    limits.max_buffer_size = u64::MAX;
    limits.max_storage_buffer_binding_size = u32::MAX;
    limits.max_compute_workgroups_per_dimension = u32::MAX;
    assert!(plan_wide_voxel_u8(&limits, u32::MAX, 17, u64::MAX, 0, u64::MAX).is_none());
    assert!(plan_wide_voxel_u8(&limits, 1, usize::MAX, u64::MAX, 0, u64::MAX).is_none());
    let c = (i32::MAX as u64 / 20) as u32 * 4;
    let end = u64::from(c).div_ceil(4) * 4 * 5;
    assert!(end <= i32::MAX as u64);
    assert!(plan_wide_voxel_u8(&limits, c, 5, end, 0, i32::MAX as u64).is_some());
    assert!(plan_wide_voxel_u8(&limits, c, 5, end, 0, end - 1).is_none());
    let beyond = u64::from(c + 4).div_ceil(4) * 4 * 5;
    assert!(plan_wide_voxel_u8(&limits, c + 4, 5, beyond, 0, i32::MAX as u64).is_none());
    checked += 6;
    eprintln!("VOXEL_U8_WIDE_PLAN verified {checked} device/host/offset/word bounds; 17 channels use 5 passes and one output");
}

#[test]
fn wide_voxel_u8_recording_preserves_sparse_packed_tails_and_adjacent_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_wide_voxel_u8(&r.device().limits(), 257, 5, 1352, 52, 1352).is_none() {
        return;
    }
    let mut compared = 0;
    for cells in [1, 2, 3, 4, 5, 17, 255, 256, 257, 65_537] {
        for channels in [5, 6, 7, 8, 9, 11, 12, 13, 16, 17] {
            let mut f = fixture(&r, cells, 0, channels, 4);
            for (cell, start) in f.starts.iter_mut().enumerate() {
                *start = (cell / 5 * 20 + [0, 0, 1, 4, 8][cell % 5]) as u32;
            }
            if cells > 1 {
                *f.starts.last_mut().unwrap() = f.indices.len() as u32;
            }
            f.indices.reverse();
            f.segments = gpu_segments(&r, &f.indices, &f.starts);
            for (c, v) in f.bytes.iter_mut().enumerate() {
                for (i, v) in v.iter_mut().enumerate() {
                    *v = ((i * 37 + c * 71) % 256) as u8;
                }
            }
            for average in [false, true] {
                compared += compare_raw(&r, &f, average, true);
            }
        }
    }
    eprintln!("VOXEL_U8_WIDE_RAW verified {compared} exact parent/CPU cell bytes; all packed tails, sparse/empty/permuted cells and offset guards");
}

#[test]
fn wide_voxel_u8_keeps_every_ordered_half_tie_long_f32_sums_and_rejected_recorders() {
    let r = WgpuRuntime::new_headless().unwrap();
    if plan_wide_voxel_u8(&r.device().limits(), 1, 5, 72, 52, 72).is_none() {
        return;
    }
    let mut f = fixture(&r, 65_536, 0, 5, 2);
    f.indices = (0..131_072).collect();
    f.segments = gpu_segments(&r, &f.indices, &f.starts);
    for (c, v) in f.bytes.iter_mut().enumerate() {
        for (i, v) in v.iter_mut().enumerate() {
            let pair = i / 2;
            let value = if i % 2 == 0 { (pair / 256) as u8 } else { (pair % 256) as u8 };
            *v = if c % 2 == 0 { value } else { 255 - value };
        }
    }
    let mut compared = compare_raw(&r, &f, true, true);
    let mut long = fixture(&r, 1, 0, 9, 262_145);
    long.indices.reverse();
    long.segments = gpu_segments(&r, &long.indices, &long.starts);
    for (c, v) in long.bytes.iter_mut().enumerate() {
        for (i, v) in v.iter_mut().enumerate() {
            *v = ((i * 37 + c * 71) % 256) as u8;
        }
    }
    compared += compare_raw(&r, &long, true, false);
    let small = fixture(&r, 17, 0, 5, 1);
    let sentinel = vec![0xdead_beef_u32; 32];
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wide-u8-rejected-staging"),
        size: 128,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&staging, 0, bytemuck::cast_slice(&sentinel));
    let refs = small.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for count in 0..=5 {
        for average in [false, true] {
            assert!(!try_record_wide_voxel_u8_to_staging(
                &mut encoder,
                &r,
                &refs[..count],
                &small.segments,
                &staging,
                52,
                average
            )
            .unwrap());
        }
    }
    r.queue().submit(Some(encoder.finish()));
    assert_eq!(read_words(&r, &staging), sentinel);
    eprintln!("VOXEL_U8_WIDE_ROUND verified {compared} parent/CPU cell bytes; all 65536 ordered byte pairs, f32 sums beyond 2^24, and no-write rejections");
}

#[test]
fn wide_voxel_u8_public_apis_match_complete_parents_cpu_capacities_and_error_order() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [0, 1, 2, 3, 4, 5, 17, 255, 256, 257, 65_537, 1_048_577] {
        for channels in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17] {
            if cells > 65_537 && ![5, 7, 9, 17].contains(&channels) {
                continue;
            }
            for attrs in [0, 4, 7] {
                if cells > 257 && attrs != 4 {
                    continue;
                }
                let f = fixture(&r, cells, attrs, channels, if cells > 257 { 1 } else { 4 });
                for mode in 0..4 {
                    let old = call(&r, &f, mode, true);
                    let new = call(&r, &f, mode, false);
                    assert_output(&new, &old);
                    assert_output(&new, &expected(&f, mode));
                    assert_capacity(&new, cells, attrs, channels);
                    r.clear_buffer_pool();
                    assert_output(&new, &old);
                    compared += cells * (3 + attrs + channels) * 2;
                }
            }
        }
    }
    let f = fixture(&r, 17, 7, 17, 4);
    let attrs = f.floats[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bytes = f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for mode in 0..4 {
        for bad in [0, 4, 8, 12, 16] {
            let mut invalid = bytes.clone();
            invalid[bad] = &invalid[bad][..67 - bad];
            if bad < 16 {
                invalid[16] = &invalid[16][..1];
            }
            let error = invoke(&r, &f, &attrs, &invalid, mode, false).unwrap_err();
            assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found: 67 - bad });
            assert_eq!(error, invoke(&r, &f, &attrs, &invalid, mode, true).unwrap_err());
            let mut bad_attrs = attrs.clone();
            bad_attrs[1] = &bad_attrs[1][..66];
            let error = invoke(&r, &f, &bad_attrs, &invalid, mode, false).unwrap_err();
            assert_eq!(error, SpatialError::BufferLengthMismatch { expected: 68, found: 66 });
            let empty = fixture(&r, 0, 7, 17, 1);
            let new = invoke(&r, &empty, &bad_attrs, &invalid, mode, false).unwrap();
            assert_output(&new, &invoke(&r, &empty, &bad_attrs, &invalid, mode, true).unwrap());
            assert_capacity(&new, 0, 7, 17);
        }
    }
    eprintln!("VOXEL_U8_WIDE_PUBLIC verified {compared} complete parent/CPU elements across four public APIs");
}

#[test]
fn wide_voxel_u8_concurrent_calls_reupload_mutations_and_keep_owned_results() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles = (0..4)
        .map(|mode| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut f = fixture(&r, 65_536 + mode, 4 + mode, 5 + 4 * mode, 1);
                barrier.wait();
                let retained = call(&r, &f, mode, false);
                let snapshot = expected(&f, mode);
                assert_output(&retained, &snapshot);
                for v in &mut f.bytes {
                    v.fill(231);
                }
                for v in &mut f.floats[3..] {
                    v.fill(123.25);
                }
                let changed = call(&r, &f, mode, false);
                assert_output(&changed, &call(&r, &f, mode, true));
                assert_output(&changed, &expected(&f, mode));
                drop(f);
                r.clear_buffer_pool();
                assert_output(&retained, &snapshot);
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[path = "voxel_u8_wide_benchmark.rs"]
mod benchmark;

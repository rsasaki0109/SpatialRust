use super::*;
use crate::{
    gather_voxel_first_xyz_and_average_multi_gpu, reduce_voxel_average_f32_multi_gpu,
    reduce_voxel_centroids_xyz_and_average_multi_gpu,
};
use spatialrust_core::SpatialError;

#[path = "voxel_average_reuse_reference.rs"]
mod baseline;

type Output = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<u8>>);

struct Fixture {
    source: Vec<Vec<f32>>,
    bytes: Vec<Vec<u8>>,
    xyz: [wgpu::Buffer; 3],
    segments: GpuVoxelSegments,
    indices: Vec<u32>,
    starts: Vec<u32>,
}

fn gpu_segments(r: &WgpuRuntime, indices: &[u32], starts: &[u32]) -> GpuVoxelSegments {
    let upload = |label, values: &[u32]| {
        r.upload_u32_storage(label, if values.is_empty() { &[0] } else { values }).unwrap()
    };
    GpuVoxelSegments::new(
        starts.len() as u32,
        indices.len() as u32,
        upload("average-reuse-keys", &[0]),
        upload("average-reuse-indices", indices),
        upload("average-reuse-starts", starts),
        0,
    )
}

fn fixture(
    r: &WgpuRuntime,
    cells: usize,
    attributes: usize,
    bytes: usize,
    occupancy: usize,
) -> Fixture {
    let points = cells * occupancy;
    let source: Vec<Vec<f32>> = (0..3 + attributes)
        .map(|c| (0..points).map(|i| ((i * 17 + c * 29) % 4096) as f32 * 0.25 - 511.5).collect())
        .collect();
    let xyz = std::array::from_fn(|c| {
        r.upload_f32_storage("average-reuse-xyz", if points == 0 { &[0.0] } else { &source[c] })
            .unwrap()
    });
    let indices = (0..cells)
        .flat_map(|cell| (0..occupancy).map(move |lane| (lane * cells + cell) as u32))
        .collect::<Vec<_>>();
    let starts = (0..cells).map(|c| (c * occupancy) as u32).collect::<Vec<_>>();
    Fixture {
        source,
        bytes: (0..bytes).map(|c| vec![(c * 37 + 11) as u8; points]).collect(),
        xyz,
        segments: gpu_segments(r, &indices, &starts),
        indices,
        starts,
    }
}

fn invoke_with_refs(
    r: &WgpuRuntime,
    f: &Fixture,
    attrs: &[&[f32]],
    bytes: &[&[u8]],
    mode: usize,
    old: bool,
) -> SpatialResult<Output> {
    if mode == 0 {
        let fun = if old {
            baseline::baseline_reduce_voxel_average_f32_multi_gpu
        } else {
            reduce_voxel_average_f32_multi_gpu
        };
        return Ok((Vec::new(), Vec::new(), Vec::new(), fun(r, attrs, &f.segments)?, Vec::new()));
    }
    let fun = match (mode, old) {
        (1, false) => gather_voxel_first_xyz_and_average_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_average_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_average_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_average_multi_gpu,
        _ => unreachable!(),
    };
    fun(r, &f.xyz[0], &f.xyz[1], &f.xyz[2], attrs, bytes, &f.segments)
}

fn invoke(r: &WgpuRuntime, f: &Fixture, mode: usize, old: bool) -> SpatialResult<Output> {
    invoke_with_refs(
        r,
        f,
        &f.source[3..].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        mode,
        old,
    )
}

fn floats(out: &Output) -> impl Iterator<Item = &Vec<f32>> {
    [&out.0, &out.1, &out.2].into_iter().chain(&out.3)
}

fn assert_bits(a: &Output, b: &Output) {
    assert_eq!(a.3.len(), b.3.len());
    for (a, b) in floats(a).zip(floats(b)) {
        assert_eq!(bytemuck::cast_slice::<f32, u32>(a), bytemuck::cast_slice::<f32, u32>(b));
    }
    assert_eq!(a.4, b.4);
}

fn assert_capacity(out: &Output, cells: usize, attributes: usize, bytes: usize, mode: usize) {
    assert_eq!(out.3.len(), attributes);
    for plane in floats(out).skip(if mode == 0 { 3 } else { 0 }) {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
    assert_eq!(out.4.len(), if mode == 0 { 0 } else { bytes });
    for plane in &out.4 {
        assert_eq!(plane.len(), cells);
        assert_eq!(plane.capacity(), cells);
    }
}

fn cpu_plane(f: &Fixture, channel: usize, first: bool) -> Vec<f32> {
    (0..f.starts.len())
        .map(|cell| {
            let start = f.starts[cell] as usize;
            let end = f.starts.get(cell + 1).copied().unwrap_or(f.indices.len() as u32) as usize;
            if start == end {
                0.0
            } else if first {
                f.source[channel][f.indices[start] as usize]
            } else {
                (f.indices[start..end]
                    .iter()
                    .map(|&i| f64::from(f.source[channel][i as usize]))
                    .sum::<f64>()
                    / (end - start) as f64) as f32
            }
        })
        .collect()
}

fn cpu_output(f: &Fixture, mode: usize) -> Output {
    let axes: [Vec<f32>; 3] =
        std::array::from_fn(|c| if mode == 0 { Vec::new() } else { cpu_plane(f, c, mode == 1) });
    let [x, y, z] = axes;
    (
        x,
        y,
        z,
        (3..f.source.len()).map(|c| cpu_plane(f, c, false)).collect(),
        if mode == 0 {
            Vec::new()
        } else {
            f.bytes.iter().map(|v| vec![v.first().copied().unwrap_or(0); f.starts.len()]).collect()
        },
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

#[test]
fn voxel_average_reuse_preserves_partial_batches_empty_cells_and_staging_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [1, 17, 257, 65_537] {
        for attributes in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17] {
            let mut f = fixture(&r, cells, attributes, 0, 4);
            // Empty and non-power-of-two cells, a full reversed point permutation,
            // and an empty final cell exercise all active-plane write paths.
            for (cell, start) in f.starts.iter_mut().enumerate() {
                *start = (cell / 5 * 20 + [0, 0, 1, 4, 8][cell % 5]) as u32;
            }
            f.indices.reverse();
            if cells > 1 {
                *f.starts.last_mut().unwrap() = f.indices.len() as u32;
            }
            f.segments = gpu_segments(&r, &f.indices, &f.starts);
            for nonfinite in [false, true] {
                if nonfinite {
                    for (c, v) in f.source[3..].iter_mut().enumerate() {
                        for (i, v) in v.iter_mut().enumerate() {
                            *v = f32::from_bits(
                                [
                                    0,
                                    0x8000_0000,
                                    0x7fc0_1234,
                                    0x7f80_0001,
                                    0x7f80_0000,
                                    0xff80_0000,
                                    1,
                                    0x007f_ffff,
                                    0x0080_0000,
                                    0x7f7f_ffff,
                                    0xff7f_ffff,
                                    0x3fa0_0000,
                                ][(i + c * 7) % 12],
                            );
                        }
                    }
                }
                let refs = f.source[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
                let words = 13 + cells * attributes + 19;
                let sentinel = vec![0xdead_beef; words];
                let buffers: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
                    let b = r.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("average-reuse-guarded-staging"),
                        size: (words * 4) as u64,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    r.queue().write_buffer(&b, 0, bytemuck::cast_slice(&sentinel));
                    b
                });
                let mut encoder = r.device().create_command_encoder(&Default::default());
                baseline::voxel_average_batch::record_voxel_attribute_averages_to_staging(
                    &mut encoder,
                    &r,
                    &refs,
                    &f.segments,
                    &buffers[0],
                    52,
                    (cells * 4) as u64,
                )
                .unwrap();
                record_voxel_attribute_averages_to_staging(
                    &mut encoder,
                    &r,
                    &refs,
                    &f.segments,
                    &buffers[1],
                    52,
                    (cells * 4) as u64,
                )
                .unwrap();
                r.queue().submit(Some(encoder.finish()));
                let old = read_words(&r, &buffers[0]);
                let new = read_words(&r, &buffers[1]);
                assert_eq!(new, old);
                assert_eq!(&new[..13], &sentinel[..13]);
                assert_eq!(&new[13 + cells * attributes..], &sentinel[13 + cells * attributes..]);
                if !nonfinite {
                    for c in 0..attributes {
                        for (&actual, expected) in new[13 + c * cells..13 + (c + 1) * cells]
                            .iter()
                            .zip(cpu_plane(&f, c + 3, false))
                        {
                            let actual = f32::from_bits(actual);
                            assert!(
                                (actual - expected).abs()
                                    <= 4.0 * f32::EPSILON * expected.abs().max(1.0)
                            );
                        }
                    }
                }
                compared += cells * attributes;
            }
        }
    }
    eprintln!("VOXEL_AVERAGE_REUSE_RAW verified {compared} exact parent words; inactive tails, sparse/permuted cells, nonfinite values and adjacent guards");
}

#[test]
fn voxel_average_reuse_complete_apis_match_current_parent_cpu_and_owned_capacities() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    for cells in [0, 1, 17, 255, 256, 257, 65_537, 262_145, 1_048_577] {
        for attributes in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17] {
            if cells > 257 && ![5, 9, 13, 17].contains(&attributes) {
                continue;
            }
            let f = fixture(&r, cells, attributes, 3, if cells > 257 { 1 } else { 4 });
            for mode in 0..3 {
                let old = invoke(&r, &f, mode, true).unwrap();
                let new = invoke(&r, &f, mode, false).unwrap();
                assert_bits(&new, &old);
                assert_bits(&new, &cpu_output(&f, mode));
                assert_capacity(&new, cells, attributes, 3, mode);
                r.clear_buffer_pool();
                assert_bits(&new, &old);
                compared += cells * (attributes + if mode == 0 { 0 } else { 6 }) * 2;
            }
        }
    }
    eprintln!("VOXEL_AVERAGE_REUSE_PUBLIC verified {compared} complete parent/CPU elements across three public APIs");
}

#[test]
fn voxel_average_reuse_keeps_original_error_order_and_empty_short_circuits() {
    let r = WgpuRuntime::new_headless().unwrap();
    let f = fixture(&r, 17, 17, 3, 4);
    let good = f.source[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
    let good_bytes = f.bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for mode in 0..3 {
        for bad in [0, 4, 8, 12, 16] {
            let mut attrs = good.clone();
            attrs[bad] = &attrs[bad][..67 - bad];
            if bad < 16 {
                attrs[16] = &attrs[16][..1];
            }
            let new = invoke_with_refs(&r, &f, &attrs, &good_bytes, mode, false).unwrap_err();
            assert_eq!(new, SpatialError::BufferLengthMismatch { expected: 68, found: 67 - bad });
            assert_eq!(new, invoke_with_refs(&r, &f, &attrs, &good_bytes, mode, true).unwrap_err());
        }
        if mode > 0 {
            let mut bytes = good_bytes.clone();
            bytes[1] = &bytes[1][..65];
            let new = invoke_with_refs(&r, &f, &good, &bytes, mode, false).unwrap_err();
            assert_eq!(new, SpatialError::BufferLengthMismatch { expected: 68, found: 65 });
            assert_eq!(new, invoke_with_refs(&r, &f, &good, &bytes, mode, true).unwrap_err());
        }
        let empty = fixture(&r, 0, 17, 3, 4);
        let new = invoke_with_refs(&r, &empty, &good, &good_bytes, mode, false).unwrap();
        assert_bits(&new, &invoke_with_refs(&r, &empty, &good, &good_bytes, mode, true).unwrap());
        assert_capacity(&new, 0, 17, 3, mode);
    }
}

#[test]
fn voxel_average_reuse_concurrent_calls_keep_private_scratch_and_mutated_sources() {
    let r = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles = [0, 1, 2, 0]
        .into_iter()
        .enumerate()
        .map(|(worker, mode)| {
            let r = r.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut f = fixture(&r, 65_537 + worker, 13 + worker, 3, 1);
                barrier.wait();
                let retained = invoke(&r, &f, mode, false).unwrap();
                let expected = cpu_output(&f, mode);
                assert_bits(&retained, &expected);
                for channel in &mut f.source[3..] {
                    channel.fill(123.25 + worker as f32);
                }
                let next = invoke(&r, &f, mode, false).unwrap();
                assert_bits(&next, &invoke(&r, &f, mode, true).unwrap());
                assert_bits(&next, &cpu_output(&f, mode));
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&retained, &expected);
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[path = "voxel_average_reuse_benchmark.rs"]
mod benchmark;

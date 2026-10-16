use super::*;
use crate::{
    gather_voxel_first_f32_multi_gpu, gather_voxel_first_xyz_and_multi_gpu,
    reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
};
use spatialrust_core::SpatialError;

#[path = "voxel_first_reuse_reference.rs"]
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
        upload("first-reuse-keys", &[0]),
        upload("first-reuse-indices", indices),
        upload("first-reuse-starts", starts),
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
        r.upload_f32_storage("first-reuse-xyz", if points == 0 { &[0.0] } else { &source[c] })
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
            baseline::baseline_gather_voxel_first_f32_multi_gpu
        } else {
            gather_voxel_first_f32_multi_gpu
        };
        return Ok((Vec::new(), Vec::new(), Vec::new(), fun(r, attrs, &f.segments)?, Vec::new()));
    }
    let fun = match (mode, old) {
        (1, false) => gather_voxel_first_xyz_and_multi_gpu,
        (1, true) => baseline::baseline_gather_voxel_first_xyz_and_multi_gpu,
        (2, false) => reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
        (2, true) => baseline::baseline_reduce_voxel_centroids_xyz_and_gather_first_multi_gpu,
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
        (3..f.source.len()).map(|c| cpu_plane(f, c, true)).collect(),
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

fn record_raw(
    encoder: &mut wgpu::CommandEncoder,
    r: &WgpuRuntime,
    f: &Fixture,
    channels: &[&[f32]],
    staging: &wgpu::Buffer,
    variant: (usize, bool, bool),
) -> Vec<wgpu::Buffer> {
    let (width, combined, old) = variant;
    let mut recycle = Vec::new();
    if combined {
        if old {
            baseline::voxel_gather_packed::record_packed_voxel_attributes(
                encoder,
                r,
                channels,
                &f.segments,
                staging,
                &mut recycle,
                baseline::voxel_gather_packed::PackedGatherPlan {
                    width,
                    channel_bytes: f.starts.len() as u64 * 4,
                    offset: 52,
                },
            )
            .unwrap();
        } else {
            crate::kernels::voxel_gather_packed::record_packed_voxel_attributes(
                encoder,
                r,
                channels,
                &f.segments,
                staging,
                &mut recycle,
                crate::kernels::voxel_gather_packed::PackedGatherPlan {
                    width,
                    channel_bytes: f.starts.len() as u64 * 4,
                    offset: 52,
                },
            )
            .unwrap();
        }
        assert_eq!(recycle.len(), channels.len());
    } else if old {
        let plan = baseline::voxel_standalone_packed::plan_packed_voxel_submission(
            &r.device().limits(),
            f.starts.len() as u32,
            channels.len().max(1),
            width,
            isize::MAX as u64,
        )
        .unwrap();
        baseline::voxel_standalone_packed::record_packed_voxel_window(
            encoder,
            r,
            channels,
            &f.segments,
            staging,
            52,
            plan,
        );
    } else {
        let plan = plan_packed_voxel_submission(
            &r.device().limits(),
            f.starts.len() as u32,
            channels.len().max(1),
            width,
            isize::MAX as u64,
        )
        .unwrap();
        record_packed_voxel_window(encoder, r, channels, &f.segments, staging, 52, plan);
    }
    recycle
}

fn first_words(f: &Fixture, channel: usize, scalar_empty: bool) -> Vec<u32> {
    (0..f.starts.len())
        .map(|cell| {
            let start = f.starts[cell] as usize;
            let end = f.starts.get(cell + 1).copied().unwrap_or(f.indices.len() as u32) as usize;
            if start >= end {
                if scalar_empty {
                    0
                } else {
                    f.source[channel][0].to_bits()
                }
            } else {
                f.source[channel][f.indices[start] as usize].to_bits()
            }
        })
        .collect()
}

#[test]
fn voxel_first_reuse_preserves_both_widths_scalar_empty_semantics_and_staging_guards() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    let mut forced = 0;
    for cells in [1, 17, 257, 65_537] {
        for attributes in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17] {
            let mut f = fixture(&r, cells, attributes, 0, 4);
            for (cell, start) in f.starts.iter_mut().enumerate() {
                *start = (cell / 5 * 20 + [0, 0, 1, 4, 8][cell % 5]) as u32;
            }
            if cells > 1 {
                *f.starts.last_mut().unwrap() = f.indices.len() as u32;
            }
            f.indices.reverse();
            f.segments = gpu_segments(&r, &f.indices, &f.starts);
            for (c, channel) in f.source[3..].iter_mut().enumerate() {
                for (i, value) in channel.iter_mut().enumerate() {
                    *value = f32::from_bits(
                        [
                            0x3fa0_0000,
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
                            0xffc0_9876,
                        ][(i * 17 + c * 7) % 12],
                    );
                }
            }
            let refs = f.source[3..].iter().map(Vec::as_slice).collect::<Vec<_>>();
            for width in [2, 4] {
                if plan_packed_voxel_submission(
                    &r.device().limits(),
                    cells as u32,
                    attributes.max(1),
                    width,
                    isize::MAX as u64,
                )
                .is_none()
                {
                    continue;
                }
                for combined in [false, true] {
                    // Standalone recording uses the original at-most-two-batch window.
                    let channels =
                        if combined { &refs[..] } else { &refs[..refs.len().min(2 * width)] };
                    let words = 13 + cells * channels.len() + 19;
                    let sentinel = vec![0xdead_beef; words];
                    let buffers: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
                        let b = r.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some("first-reuse-guarded-staging"),
                            size: (words * 4) as u64,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        r.queue().write_buffer(&b, 0, bytemuck::cast_slice(&sentinel));
                        b
                    });
                    let mut encoder = r.device().create_command_encoder(&Default::default());
                    let old_leases = record_raw(
                        &mut encoder,
                        &r,
                        &f,
                        channels,
                        &buffers[0],
                        (width, combined, true),
                    );
                    let new_leases = record_raw(
                        &mut encoder,
                        &r,
                        &f,
                        channels,
                        &buffers[1],
                        (width, combined, false),
                    );
                    r.queue().submit(Some(encoder.finish()));
                    let old = read_words(&r, &buffers[0]);
                    let new = read_words(&r, &buffers[1]);
                    assert_eq!(new, old);
                    assert_eq!(&new[..13], &sentinel[..13]);
                    assert_eq!(
                        &new[13 + cells * channels.len()..],
                        &sentinel[13 + cells * channels.len()..]
                    );
                    for c in 0..channels.len() {
                        let scalar_empty =
                            combined && channels.len() % width == 1 && c + 1 == channels.len();
                        assert_eq!(
                            &new[13 + c * cells..13 + (c + 1) * cells],
                            first_words(&f, c + 3, scalar_empty)
                        );
                    }
                    for b in old_leases.into_iter().chain(new_leases) {
                        r.recycle_storage(b.size(), b);
                    }
                    compared += cells * channels.len() * 2;
                }
                if attributes > 0 {
                    let plan = plan_packed_voxel_submission(
                        &r.device().limits(),
                        cells as u32,
                        attributes,
                        width,
                        isize::MAX as u64,
                    )
                    .unwrap();
                    let old_plan = baseline::voxel_standalone_packed::plan_packed_voxel_submission(
                        &r.device().limits(),
                        cells as u32,
                        attributes,
                        width,
                        isize::MAX as u64,
                    )
                    .unwrap();
                    let old =
                        baseline::voxel_standalone_packed::gather_voxel_channels_packed_with_plan(
                            &r,
                            &refs,
                            &f.segments,
                            old_plan,
                        )
                        .unwrap();
                    let new = gather_voxel_channels_packed_with_plan(&r, &refs, &f.segments, plan)
                        .unwrap();
                    for (c, plane) in new.iter().enumerate() {
                        assert_eq!(
                            bytemuck::cast_slice::<f32, u32>(plane),
                            bytemuck::cast_slice::<f32, u32>(&old[c])
                        );
                        assert_eq!(
                            bytemuck::cast_slice::<f32, u32>(plane),
                            first_words(&f, c + 3, false)
                        );
                        assert_eq!(plane.capacity(), cells);
                    }
                    forced += cells * attributes * 2;
                }
            }
        }
    }
    eprintln!("VOXEL_FIRST_REUSE_RAW verified {compared} exact parent/CPU words; width 2/4, sparse/nonfinite, scalar empty behavior and adjacent guards");
    eprintln!("VOXEL_FIRST_REUSE_FORCED verified {forced} complete forced-width parent/CPU words across original bounded submission windows");
}

#[test]
fn voxel_first_reuse_complete_apis_match_current_parent_cpu_and_owned_capacities() {
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
    eprintln!("VOXEL_FIRST_REUSE_PUBLIC verified {compared} complete parent/CPU elements across three public APIs");
}

#[test]
fn voxel_first_reuse_keeps_original_error_order_and_empty_short_circuits() {
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
fn voxel_first_reuse_concurrent_calls_keep_private_scratch_and_mutated_sources() {
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

#[path = "voxel_first_reuse_benchmark.rs"]
mod benchmark;

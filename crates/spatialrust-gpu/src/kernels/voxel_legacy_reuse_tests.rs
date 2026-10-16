use super::*;

#[path = "voxel_legacy_reuse_reference.rs"]
mod baseline;

type Channels = Vec<Vec<f32>>;

struct Fixture {
    channels: Channels,
    segments: GpuVoxelSegments,
    indices: Vec<u32>,
    starts: Vec<u32>,
}

fn fixture(r: &WgpuRuntime, cells: usize, channels: usize, seed: usize) -> Fixture {
    const SPECIAL: [u32; 12] = [
        0x8000_0000,
        0,
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
    let points = cells + 17;
    let channels = (0..channels)
        .map(|channel| {
            (0..points)
                .map(|i| {
                    f32::from_bits(if i < SPECIAL.len() {
                        SPECIAL[(i + channel + seed) % SPECIAL.len()]
                    } else {
                        ((i + channel * 31 + seed) as u32).wrapping_mul(0x9e37_79b9).rotate_left(7)
                    })
                })
                .collect()
        })
        .collect();
    let indices: Vec<_> = (0..points as u32).rev().collect();
    let starts: Vec<_> = (0..cells)
        .map(|i| match i % 5 {
            0 | 1 => 0,
            2 => 7,
            3 => 3,
            _ => points as u32 - 1,
        })
        .collect();
    let segments = GpuVoxelSegments::new(
        cells as u32,
        points as u32,
        r.upload_u32_storage("legacy-reuse-keys", &[0]).unwrap(),
        r.upload_u32_storage("legacy-reuse-indices", &indices).unwrap(),
        r.upload_u32_storage("legacy-reuse-starts", &starts).unwrap(),
        0,
    );
    Fixture { channels, segments, indices, starts }
}

fn cpu(f: &Fixture) -> Channels {
    f.channels
        .iter()
        .map(|source| {
            (0..f.starts.len())
                .map(|cell| {
                    let start = f.starts[cell];
                    let end = f.starts.get(cell + 1).copied().unwrap_or(f.indices.len() as u32);
                    source[if end <= start { 0 } else { f.indices[start as usize] as usize }]
                })
                .collect()
        })
        .collect()
}

fn assert_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (channel, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), b.len());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "channel {channel}"
        );
    }
}

fn invoke(r: &WgpuRuntime, f: &Fixture, plan: GatherSubmissionPlan, old: bool) -> Channels {
    let refs: Vec<_> = f.channels.iter().map(Vec::as_slice).collect();
    if old {
        baseline::baseline_gather_voxel_channels_with_plan(r, &refs, &f.segments, plan)
    } else {
        gather_voxel_channels_with_plan(r, &refs, &f.segments, plan)
    }
    .unwrap()
}

#[test]
fn legacy_voxel_window_reuse_keeps_raw_sparse_tails_and_adjacent_staging_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for width in [2, 4] {
        if width > r.max_gather_channels() as usize {
            continue;
        }
        for cells in [1, 3, 17, 255, 256, 257, 65_537] {
            for channels in 0..=2 * width {
                let f = fixture(&r, cells, channels, width);
                let refs: Vec<_> = f.channels.iter().map(Vec::as_slice).collect();
                let plan = plan_voxel_gather_submission(
                    &r.device().limits(),
                    cells as u32,
                    2 * width + 1,
                    width,
                )
                .unwrap();
                let empty = r.upload_u32_storage("legacy-reuse-empty", &[0]).unwrap();
                let mut expected: Vec<_> = cpu(&f).iter().flatten().map(|v| v.to_bits()).collect();
                expected.extend_from_slice(&[0xdead_beef; 32]);
                for old in [true, false] {
                    let target = r.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("legacy-reuse-guarded-staging"),
                        size: expected.len() as u64 * 4,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    r.queue().write_buffer(
                        &target,
                        0,
                        bytemuck::cast_slice(&vec![0xdead_beef_u32; expected.len()]),
                    );
                    let mut encoder = r.device().create_command_encoder(&Default::default());
                    if old {
                        baseline::baseline_record_window(
                            &mut encoder,
                            &r,
                            &refs,
                            &f.segments,
                            &empty,
                            &target,
                            plan,
                        )
                    } else {
                        record_window(&mut encoder, &r, &refs, &f.segments, &empty, &target, plan)
                    }
                    .unwrap();
                    r.queue().submit(Some(encoder.finish()));
                    let values =
                        crate::readback::read_staging_f32(r.device(), &target, expected.len())
                            .unwrap();
                    assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
                    checked += expected.len();
                }
            }
        }
    }
    eprintln!("VOXEL_LEGACY_REUSE_RAW verified {checked} parent/CPU words, empty windows and 32-word suffix guards");
}

#[test]
fn legacy_voxel_complete_fallback_reuse_preserves_plans_capacities_and_owned_channels() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for width in [2, 4] {
        if width > r.max_gather_channels() as usize {
            continue;
        }
        for cells in [1, 17, 257, 65_537, 262_145] {
            for channels in [width + 1, 2 * width - 1, 2 * width, 2 * width + 1, 17] {
                let mut f = fixture(&r, cells, channels, 0);
                let plan = plan_voxel_gather_submission(
                    &r.device().limits(),
                    cells as u32,
                    channels,
                    width,
                )
                .unwrap();
                // These valid individual planes reject the preferred packed binding.
                let limited = wgpu::Limits {
                    max_storage_buffer_binding_size: (plan.channel_bytes * width as u64 - 1) as u32,
                    ..r.device().limits()
                };
                assert!(
                    plan_voxel_gather_submission(&limited, cells as u32, channels, width).is_some()
                );
                assert!(crate::kernels::voxel_standalone_packed::plan_packed_voxel_submission(
                    &limited,
                    cells as u32,
                    channels,
                    width,
                    isize::MAX as u64,
                )
                .is_none());
                let expected = cpu(&f);
                let old = invoke(&r, &f, plan, true);
                let live = invoke(&r, &f, plan, false);
                assert_bits(&live, &old);
                assert_bits(&live, &expected);
                assert!(live.iter().all(|plane| plane.len() == cells && plane.capacity() == cells));
                for source in &mut f.channels {
                    source.fill(-0.0);
                }
                let changed = invoke(&r, &f, plan, false);
                assert!(changed.iter().flatten().all(|v| v.to_bits() == 0x8000_0000));
                drop(f);
                r.clear_buffer_pool();
                assert_bits(&live, &old);
                assert_bits(&live, &expected);
                checked += cells * channels * 4;
            }
        }
    }
    eprintln!("VOXEL_LEGACY_REUSE_CALLS verified {checked} complete fallback/parent/CPU words and rejected packed bindings");
}

#[test]
fn legacy_voxel_concurrent_output_reuse_keeps_each_call_owned() {
    let r = WgpuRuntime::new_headless().unwrap();
    let width = r.max_gather_channels() as usize;
    assert!([2, 4].contains(&width));
    let barrier = std::sync::Barrier::new(4);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let cells = 16_385 + worker;
                    let channels = 2 * width + worker + 1;
                    let f = fixture(r, cells, channels, worker);
                    let plan = plan_voxel_gather_submission(
                        &r.device().limits(),
                        cells as u32,
                        channels,
                        width,
                    )
                    .unwrap();
                    let expected = cpu(&f);
                    barrier.wait();
                    let live = invoke(r, &f, plan, false);
                    assert_bits(&live, &expected);
                    (live, expected)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    r.wait_idle();
    drop(r);
    let mut checked = 0;
    for (live, expected) in results {
        assert_bits(&live, &expected);
        checked += live.iter().map(Vec::len).sum::<usize>();
    }
    eprintln!("VOXEL_LEGACY_REUSE_CONCURRENT verified {checked} owned words after four calls and runtime drop");
}

#[path = "voxel_legacy_reuse_benchmark.rs"]
mod benchmark;

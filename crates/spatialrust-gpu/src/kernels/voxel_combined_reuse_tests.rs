use super::*;
use crate::kernels::voxel_gather_packed;

#[path = "voxel_combined_reuse_reference.rs"]
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
        r.upload_u32_storage("combined-reuse-keys", &[0]).unwrap(),
        r.upload_u32_storage("combined-reuse-indices", &indices).unwrap(),
        r.upload_u32_storage("combined-reuse-starts", &starts).unwrap(),
        0,
    );
    Fixture { channels, segments, indices, starts }
}

fn cpu(f: &Fixture) -> Channels {
    f.channels
        .iter()
        .enumerate()
        .map(|(channel, source)| {
            (0..f.starts.len())
                .map(|cell| {
                    let start = f.starts[cell];
                    let end = f.starts.get(cell + 1).copied().unwrap_or(f.indices.len() as u32);
                    if end <= start && f.channels.len() % 4 == 1 && channel + 1 == f.channels.len()
                    {
                        0.0
                    } else {
                        source[if end <= start { 0 } else { f.indices[start as usize] as usize }]
                    }
                })
                .collect()
        })
        .collect()
}

fn runtime4() -> Option<WgpuRuntime> {
    let r = WgpuRuntime::new_headless().unwrap();
    if r.max_gather_channels() < 4 {
        eprintln!("VOXEL_COMBINED_REUSE skipped: four-channel gather unavailable");
        None
    } else {
        Some(r)
    }
}

// The XYZ prefix is a guard region. This invokes the complete attribute recorder,
// with an explicit fallback switch only in this native test fixture.
fn invoke(r: &WgpuRuntime, f: &Fixture, route: usize, extra_words: usize) -> Vec<u32> {
    assert!(route < 4 && (route < 2 || extra_words == 0));
    let channel_words = f.starts.len() + extra_words;
    let words = channel_words * (3 + f.channels.len()) + 32;
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("combined-reuse-guarded-staging"),
        size: words as u64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&staging, 0, bytemuck::cast_slice(&vec![0xdead_beef_u32; words]));
    let mut encoder = r.device().create_command_encoder(&Default::default());
    let refs: Vec<_> = f.channels.iter().map(Vec::as_slice).collect();
    let mut uploads = vec![r.upload_f32_storage("combined-reuse-prior-input", &[17.0]).unwrap()];
    let recorder = match route {
        0 => baseline::baseline_record_fallback,
        1 => record_gather_f32_attribute_fallback_to_staging,
        2 => record_gather_f32_attribute_channels_to_staging,
        _ => baseline::baseline_record_preferred,
    };
    recorder(&mut encoder, r, &refs, &f.segments, &staging, channel_words as u64 * 4, &mut uploads)
        .unwrap();
    assert_eq!(uploads.len(), 1 + f.channels.len());
    assert_eq!(uploads[0].size(), 4);
    r.queue().submit(Some(encoder.finish()));
    let result = crate::readback::read_staging_f32(r.device(), &staging, words).unwrap();
    for buffer in uploads {
        r.recycle_storage(buffer.size(), buffer);
    }
    result.into_iter().map(f32::to_bits).collect()
}

fn expected_flat(f: &Fixture, extra_words: usize) -> Vec<u32> {
    let mut words = vec![0xdead_beef; 3 * (f.starts.len() + extra_words)];
    for channel in cpu(f) {
        words.extend(channel.iter().map(|v| v.to_bits()));
        // The kernels write exactly C cells, leaving the zero-initialized tail.
        words.extend(std::iter::repeat(0).take(extra_words));
    }
    words.extend_from_slice(&[0xdead_beef; 32]);
    words
}

#[test]
fn voxel_combined_reuse_preserves_all_tail_widths_and_adjacent_staging_words() {
    let Some(r) = runtime4() else { return };
    let mut checked = 0;
    for cells in [1, 3, 17, 255, 256, 257, 65_537] {
        for channels in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17, 19] {
            let f = fixture(&r, cells, channels, channels % 3);
            let expected = expected_flat(&f, 16);
            assert_eq!(invoke(&r, &f, 0, 16), expected);
            assert_eq!(invoke(&r, &f, 1, 16), expected);
            checked += 2 * expected.len();
        }
    }
    eprintln!("VOXEL_COMBINED_REUSE_RAW verified {checked} parent/CPU words with XYZ prefix, zero output tails and staging suffix guards");
}

#[test]
fn voxel_combined_reuse_keeps_preferred_selection_uploads_and_owned_results() {
    let Some(r) = runtime4() else { return };
    let mut checked = 0;
    for cells in [1, 17, 257, 65_537, 262_145] {
        for channels in [0, 1, 2, 4, 5, 6, 7, 8, 9, 17, 19, 31] {
            let mut f = fixture(&r, cells, channels, 0);
            let expected = expected_flat(&f, 0);
            let live = invoke(&r, &f, 1, 0);
            assert_eq!(invoke(&r, &f, 0, 0), expected);
            assert_eq!(live, expected);
            assert_eq!(invoke(&r, &f, 2, 0), expected);
            assert_eq!(invoke(&r, &f, 3, 0), expected);
            for channel in &mut f.channels {
                channel.fill(-0.0);
            }
            let changed = invoke(&r, &f, 1, 0);
            assert_eq!(changed, expected_flat(&f, 0));
            drop(f);
            r.clear_buffer_pool();
            assert_eq!(live, expected);
            checked += expected.len() * 6;
        }
    }
    eprintln!("VOXEL_COMBINED_REUSE_CALLS verified {checked} guarded original/fallback/preferred/CPU words, upload recycling and retained results");
}

#[test]
fn voxel_combined_reuse_concurrent_calls_keep_independent_owned_results() {
    let Some(r) = runtime4() else { return };
    let barrier = std::sync::Barrier::new(4);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = [5, 6, 7, 19]
            .into_iter()
            .enumerate()
            .map(|(worker, channels)| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let f = fixture(r, 16_385 + worker, channels, worker);
                    let expected = expected_flat(&f, 0);
                    barrier.wait();
                    let live = invoke(r, &f, 1, 0);
                    assert_eq!(live, expected);
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
        assert_eq!(live, expected);
        checked += live.len();
    }
    eprintln!("VOXEL_COMBINED_REUSE_CONCURRENT verified {checked} guarded owned words after four independent calls and runtime drop");
}

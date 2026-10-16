use super::*;

#[path = "voxel_gather_dummy_reference.rs"]
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
        eprintln!("VOXEL_GATHER_DUMMY skipped: four-channel gather unavailable");
        None
    } else {
        Some(r)
    }
}

fn output(r: &WgpuRuntime, bytes: u64) -> wgpu::Buffer {
    r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-output-active-fixture"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn target(r: &WgpuRuntime, words: usize) -> wgpu::Buffer {
    let buffer = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-output-guarded-target"),
        size: words as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    r.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&vec![0xdead_beef_u32; words]));
    buffer
}

fn record(
    r: &WgpuRuntime,
    encoder: &mut wgpu::CommandEncoder,
    inputs: &[wgpu::Buffer],
    f: &Fixture,
    outputs: &[wgpu::Buffer],
    old: bool,
) {
    let cells = f.starts.len();
    match inputs.len() {
        1 => record_voxel_gather_f32_pass(
            encoder,
            r,
            &inputs[0],
            f.segments.point_indices_buffer(),
            f.segments.cell_starts_buffer(),
            cells as u32,
            f.indices.len() as u32,
            &outputs[0],
        ),
        2 => record_voxel_gather_multi2_pass(
            encoder,
            r,
            &[&inputs[0], &inputs[1]],
            &f.segments,
            &[&outputs[0], &outputs[1]],
            2,
        ),
        _ if old => baseline::baseline_record_voxel_gather_multi4_active_outputs(
            encoder,
            r,
            inputs,
            &f.segments,
            outputs,
            cells as u64 * 4,
        ),
        _ => record_voxel_gather_multi4_active_outputs(
            encoder,
            r,
            inputs,
            &f.segments,
            outputs,
            cells as u64 * 4,
        ),
    }
    .unwrap();
}

fn expected_flat(f: &Fixture, extra_output_words: usize) -> Vec<u32> {
    let mut words = vec![0xdead_beef; 3 * f.starts.len()];
    for channel in cpu(f) {
        words.extend(channel.iter().map(|v| v.to_bits()));
        words.extend(std::iter::repeat(0xdead_beef).take(extra_output_words));
    }
    words.extend_from_slice(&[0xdead_beef; 32]);
    words
}

// Executes the extracted block with the original surrounding upload/output/copy
// shape. This fixture deliberately bypasses public packed/fused selection.
fn invoke(r: &WgpuRuntime, f: &Fixture, old: bool, extra_output_words: usize) -> Vec<u32> {
    let cells = f.starts.len();
    let channel_bytes = (cells + extra_output_words) as u64 * 4;
    let words = cells * 3 + (cells + extra_output_words) * f.channels.len() + 32;
    let staging = target(r, words);
    let mut encoder = r.device().create_command_encoder(&Default::default());
    for (batch, chunk) in f.channels.chunks(4).enumerate() {
        let inputs: Vec<_> = chunk
            .iter()
            .map(|channel| r.upload_f32_storage("dummy-output-input-fixture", channel).unwrap())
            .collect();
        let outputs: Vec<_> = (0..chunk.len())
            .map(|_| {
                let buffer = output(r, channel_bytes);
                if extra_output_words != 0 {
                    r.queue().write_buffer(
                        &buffer,
                        0,
                        bytemuck::cast_slice(&vec![0xdead_beef_u32; cells + extra_output_words]),
                    );
                }
                buffer
            })
            .collect();
        record(r, &mut encoder, &inputs, f, &outputs, old);
        for (i, output) in outputs.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                output,
                0,
                &staging,
                (cells * 3) as u64 * 4 + channel_bytes * (batch * 4 + i) as u64,
                channel_bytes,
            );
        }
    }
    r.queue().submit(Some(encoder.finish()));
    crate::readback::read_staging_f32(r.device(), &staging, words)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
fn voxel_gather_inactive_outputs_preserve_raw_bits_and_active_xyz_guard_regions() {
    let Some(r) = runtime4() else {
        return;
    };
    let mut checked = 0;
    for cells in [1, 3, 17, 255, 256, 257, 65_537] {
        for channels in [3, 4] {
            for seed in 0..3 {
                let f = fixture(&r, cells, channels, seed);
                let expected = expected_flat(&f, 16);
                let old = invoke(&r, &f, true, 16);
                let live = invoke(&r, &f, false, 16);
                assert_eq!(old, expected);
                assert_eq!(live, expected);
                checked += expected.len() * 2;
            }
        }
    }
    eprintln!("VOXEL_GATHER_DUMMY_RAW verified {checked} parent/CPU words with XYZ prefix, active-output tails and staging suffix guards");
}

#[test]
fn voxel_gather_inactive_outputs_keep_batched_tail_semantics_and_owned_results() {
    let Some(r) = runtime4() else {
        return;
    };
    let mut checked = 0;
    for cells in [1, 17, 257, 65_537, 262_145] {
        for channels in [3, 4, 5, 7, 8, 9, 11, 12, 17, 19] {
            let mut f = fixture(&r, cells, channels, 0);
            let expected = expected_flat(&f, 0);
            let live = invoke(&r, &f, false, 0);
            assert_eq!(invoke(&r, &f, true, 0), expected);
            assert_eq!(live, expected);
            for channel in &mut f.channels {
                channel.fill(-0.0);
            }
            let changed = invoke(&r, &f, false, 0);
            assert_eq!(changed, expected_flat(&f, 0));
            drop(f);
            r.clear_buffer_pool();
            assert_eq!(live, expected);
            checked += expected.len() * 3;
        }
    }
    eprintln!("VOXEL_GATHER_DUMMY_BATCH verified {checked} guarded batch/parent/CPU words, scalar zero tails and multi-channel point-zero tails");
}

#[test]
fn voxel_gather_inactive_outputs_concurrent_calls_keep_live_owned_results() {
    let Some(r) = runtime4() else {
        return;
    };
    let barrier = std::sync::Barrier::new(4);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = [3, 4, 7, 8]
            .into_iter()
            .enumerate()
            .map(|(worker, channels)| {
                let r = &r;
                let barrier = &barrier;
                scope.spawn(move || {
                    let f = fixture(r, 16_385 + worker, channels, worker);
                    let expected = expected_flat(&f, 0);
                    barrier.wait();
                    let live = invoke(r, &f, false, 0);
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
    eprintln!("VOXEL_GATHER_DUMMY_CONCURRENT verified {checked} guarded owned words after four independent calls and runtime drop");
}

#[path = "voxel_gather_dummy_benchmark.rs"]
mod benchmark;

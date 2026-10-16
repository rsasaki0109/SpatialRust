use super::*;

#[path = "voxel_gather_control_reference.rs"]
mod baseline;

#[test]
fn voxel_packed_controls_keep_queued_parameter_changes_and_owned_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut controls = PackedGatherControls::default();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    let mut results = Vec::new();
    // Adjacent contexts change only the cell count, then only the point count.
    // Channel changes and exact repeats exercise immutable replacement/reuse.
    for (context, (cells, points)) in [
        (1_usize, 19_usize),
        (17, 19),
        (17, 23),
        (17, 23),
        (255, 301),
        (256, 301),
        (257, 301),
        (65_537, 65_554),
    ]
    .into_iter()
    .enumerate()
    {
        let indices: Vec<_> = (0..points).map(|i| ((i + 2) % points) as u32).collect();
        let starts: Vec<_> = (0..cells)
            .map(|i| {
                if i + 1 == cells {
                    (points - 1) as u32
                } else {
                    match i % 5 {
                        0 | 1 => 0,
                        2 => 7,
                        3 => 3,
                        _ => (points - 1) as u32,
                    }
                }
            })
            .collect();
        let segments = GpuVoxelSegments::new(
            cells as u32,
            points as u32,
            r.upload_u32_storage("packed-control-keys", &[0]).unwrap(),
            r.upload_u32_storage("packed-control-indices", &indices).unwrap(),
            r.upload_u32_storage("packed-control-starts", &starts).unwrap(),
            0,
        );
        for (round, channels) in [4, 4, 3, 3, 2, 2, 1, 4].into_iter().enumerate() {
            let source: Vec<Vec<u32>> = (0..channels)
                .map(|channel| {
                    (0..points)
                        .map(|i| {
                            [
                                0x8000_0000_u32,
                                0,
                                0x7f80_0000,
                                0xff80_0000,
                                0x7fc0_1234,
                                0x7f80_0001,
                                1,
                                0xffc0_9876,
                            ][(i + channel + round + context) % 8]
                        })
                        .collect()
                })
                .collect();
            let values: Vec<_> = source
                .iter()
                .map(|words| r.upload_u32_storage("packed-control-values", words).unwrap())
                .collect();
            let output_words = channels * cells + 8;
            let words = 16 + 2 * output_words + 16;
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed-control-staging"),
                size: words as u64 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            r.queue().write_buffer(
                &staging,
                0,
                bytemuck::cast_slice(&vec![0xdead_beef_u32; words]),
            );
            let mut expected = vec![0xdead_beef_u32; 16];
            let mut packed = Vec::new();
            for channel in &source {
                for cell in 0..cells {
                    let start = starts[cell];
                    let end = starts.get(cell + 1).copied().unwrap_or(points as u32);
                    let index = if end <= start { 0 } else { indices[start as usize] as usize };
                    packed.push(channel[index]);
                }
            }
            packed.extend_from_slice(&[0xdead_beef; 8]);
            for old in [false, true] {
                let output = r.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("packed-control-output"),
                    size: output_words as u64 * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                r.queue().write_buffer(
                    &output,
                    0,
                    bytemuck::cast_slice(&vec![0xdead_beef_u32; output_words]),
                );
                if old {
                    baseline::baseline_record_packed_voxel_gather(
                        &mut encoder,
                        &r,
                        &values,
                        &segments,
                        &output,
                    );
                } else {
                    record_packed_voxel_gather_with_controls(
                        &mut encoder,
                        &r,
                        &values,
                        &segments,
                        &output,
                        &mut controls,
                    );
                }
                encoder.copy_buffer_to_buffer(
                    &output,
                    0,
                    &staging,
                    (16 + usize::from(old) * output_words) as u64 * 4,
                    output_words as u64 * 4,
                );
                expected.extend_from_slice(&packed);
            }
            expected.extend_from_slice(&[0xdead_beef; 16]);
            results.push((staging, expected));
        }
    }
    // All 128 passes are queued before the first submit/map. A rewritten shared
    // uniform or an incomplete cache key would corrupt earlier command results.
    r.queue().submit(Some(encoder.finish()));
    let mut retained = Vec::new();
    let mut checked = 0;
    for (staging, expected) in results {
        let actual = crate::readback::read_staging_f32(r.device(), &staging, expected.len())
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        checked += expected.len();
        retained.push((actual, expected));
    }
    drop(controls);
    r.clear_buffer_pool();
    drop(r);
    for (actual, expected) in retained {
        assert_eq!(actual, expected);
    }
    eprintln!("PACKED_GATHER_CONTROLS_QUEUED verified {checked} guarded original/CPU words across 64 contexts / 128 queued passes and runtime drop");
}

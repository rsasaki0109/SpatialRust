use super::*;

#[path = "voxel_u8_control_reference.rs"]
mod baseline;

#[test]
fn voxel_u8_controls_keep_queued_modes_parameter_changes_and_owned_words() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut controls = BatchedU8Controls::default();
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
        // Monotonic starts include an empty first cell and nonempty final cell.
        let starts: Vec<_> = (0..cells)
            .map(|i| (i.saturating_sub(1) * points / cells.saturating_sub(1).max(1)) as u32)
            .collect();
        let segments = GpuVoxelSegments::new(
            cells as u32,
            points as u32,
            r.upload_u32_storage("u8-control-keys", &[0]).unwrap(),
            r.upload_u32_storage("u8-control-indices", &indices).unwrap(),
            r.upload_u32_storage("u8-control-starts", &starts).unwrap(),
            0,
        );
        for (round, channels) in [4, 4, 3, 3, 2, 2, 1, 4].into_iter().enumerate() {
            let source: Vec<Vec<u8>> = (0..channels)
                .map(|channel| {
                    (0..points)
                        // Multiples of four avoid half ties in these segments.
                        .map(|i| {
                            (((i * 17 + channel * 11 + round * 7 + context * 3) % 64) * 4) as u8
                        })
                        .collect()
                })
                .collect();
            let values: Vec<_> = source
                .iter()
                .map(|bytes| {
                    r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("u8-control-values"),
                        contents: &crate::readback::u8_storage_upload_contents(bytes),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                })
                .collect();
            // Exercise both pipelines with exactly the same cached uniform.
            for average in [false, true] {
                let output_words = channels * cells.div_ceil(4) + 8;
                let words = 16 + 2 * output_words + 16;
                let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("u8-control-staging"),
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
                    let mut bytes = vec![0_u8; cells.div_ceil(4) * 4];
                    for cell in 0..cells {
                        let start = starts[cell] as usize;
                        let end = starts.get(cell + 1).copied().unwrap_or(points as u32) as usize;
                        if start == end {
                            continue;
                        }
                        bytes[cell] = if average {
                            let sum: u64 = indices[start..end]
                                .iter()
                                .map(|&i| u64::from(channel[i as usize]))
                                .sum();
                            (sum as f64 / (end - start) as f64).round() as u8
                        } else {
                            channel[indices[start] as usize]
                        };
                    }
                    packed.extend(
                        bytes
                            .chunks_exact(4)
                            .map(|word| u32::from_le_bytes(word.try_into().unwrap())),
                    );
                }
                packed.extend_from_slice(&[0xdead_beef; 8]);
                for old in [false, true] {
                    let output = r.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("u8-control-output"),
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
                        baseline::baseline_record_voxel_u8_batch_pass(
                            &mut encoder,
                            &r,
                            &values,
                            &segments,
                            &output,
                            average,
                        );
                    } else {
                        record_voxel_u8_batch_pass_with_controls(
                            &mut encoder,
                            &r,
                            &values,
                            &segments,
                            &output,
                            average,
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
    }
    // All 256 passes are queued before the first submit/map. A rewritten shared
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
    eprintln!("VOXEL_U8_CONTROLS_QUEUED verified {checked} guarded original/CPU words across 128 mode/parameter contexts / 256 queued passes and runtime drop");
}

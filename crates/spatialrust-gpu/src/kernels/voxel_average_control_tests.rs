use super::*;

#[path = "voxel_average_control_reference.rs"]
mod baseline;

#[test]
fn voxel_average_controls_keep_queued_parameter_changes_and_owned_results() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut controls = AverageBatchControls::default();
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
            r.upload_u32_storage("average-control-keys", &[0]).unwrap(),
            r.upload_u32_storage("average-control-indices", &indices).unwrap(),
            r.upload_u32_storage("average-control-starts", &starts).unwrap(),
            0,
        );
        for (round, channels) in [4, 4, 3, 3, 2, 2, 1, 4].into_iter().enumerate() {
            let source: Vec<Vec<u32>> = (0..channels)
                .map(|channel| {
                    (0..points)
                        .map(|i| {
                            // Bounded dyadic inputs keep sequential sums exact.
                            (((i % 31) as i32 - 15 + channel as i32 * 3 + round as i32
                                - context as i32) as f32
                                * 0.25)
                                .to_bits()
                        })
                        .collect()
                })
                .collect();
            let values: Vec<_> = source
                .iter()
                .map(|words| r.upload_u32_storage("average-control-values", words).unwrap())
                .collect();
            let output_words = channels * cells + 8;
            let words = 16 + 2 * output_words + 16;
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("average-control-staging"),
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
                    let count = end - start;
                    let sum = indices[start as usize..end as usize]
                        .iter()
                        .fold(0.0_f32, |sum, index| sum + f32::from_bits(channel[*index as usize]));
                    packed.push(if count == 0 { 0.0_f32 } else { sum / count as f32 }.to_bits());
                }
            }
            packed.extend_from_slice(&[0xdead_beef; 8]);
            for old in [false, true] {
                let output = r.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("average-control-output"),
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
                    baseline::baseline_record_voxel_average_batch(
                        &mut encoder,
                        &r,
                        &values,
                        &segments,
                        &output,
                    );
                } else {
                    record_voxel_average_batch_with_controls(
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
            results.push((staging, expected, output_words));
        }
    }
    // All 128 passes are queued before the first submit/map. A rewritten shared
    // uniform or an incomplete cache key would corrupt earlier command results.
    r.queue().submit(Some(encoder.finish()));
    let mut retained = Vec::new();
    let mut checked = 0;
    for (staging, expected, output_words) in results {
        let actual = crate::readback::read_staging_f32(r.device(), &staging, expected.len())
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_averages(&actual, &expected, output_words);
        checked += expected.len();
        retained.push((actual, expected, output_words));
    }
    drop(controls);
    r.clear_buffer_pool();
    drop(r);
    for (actual, expected, output_words) in retained {
        assert_averages(&actual, &expected, output_words);
    }
    eprintln!("VOXEL_AVERAGE_CONTROLS_QUEUED verified {checked} guarded original/CPU words across 64 contexts / 128 queued passes and runtime drop");
}

fn assert_averages(actual: &[u32], expected: &[u32], output_words: usize) {
    assert_eq!(actual.len(), expected.len());
    // Complete cached/original GPU outputs, including tail guards, match bitwise.
    assert_eq!(&actual[16..16 + output_words], &actual[16 + output_words..16 + 2 * output_words]);
    for (i, (&word, &cpu_word)) in actual.iter().zip(expected).enumerate() {
        let active = (16..16 + output_words - 8).contains(&i)
            || (16 + output_words..16 + 2 * output_words - 8).contains(&i);
        if active {
            let value = f32::from_bits(word);
            let cpu = f32::from_bits(cpu_word);
            assert!(
                (value - cpu).abs() <= 4.0 * f32::EPSILON * cpu.abs().max(1.0),
                "word {i}: {value} vs {cpu}"
            );
            if cpu == 0.0 {
                assert_eq!(word, cpu_word, "zero word {i}");
            }
        } else {
            assert_eq!(word, cpu_word, "guard word {i}");
        }
    }
}

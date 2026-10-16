use super::*;

#[path = "voxel_legacy_control_reference.rs"]
mod baseline;

#[test]
fn voxel_legacy_controls_preserve_queued_widths_keys_inactive_outputs_and_owned_bits() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut controls = LegacyGatherControls::default();
    let mut encoder = r.device().create_command_encoder(&Default::default());
    let mut results = Vec::new();
    let mut contexts = 0;
    // Adjacent contexts change only cells, then only points; later ones repeat
    // every parameter with different inputs. Width changes keep uniform data.
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
            r.upload_u32_storage("legacy-control-keys", &[0]).unwrap(),
            r.upload_u32_storage("legacy-control-indices", &indices).unwrap(),
            r.upload_u32_storage("legacy-control-starts", &starts).unwrap(),
            0,
        );
        for (round, (width, channels)) in [
            (4, 4),
            (4, 4),
            (4, 3),
            (4, 3),
            (4, 2),
            (2, 2),
            (2, 2),
            (4, 2),
            (2, 1),
            (2, 1),
            (4, 1),
            (4, 4),
        ]
        .into_iter()
        .enumerate()
        {
            if width == 4 && r.pipelines().voxel_gather.multi4_pipeline.is_none() {
                continue;
            }
            let source: Vec<Vec<u32>> = (0..4)
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
                .map(|v| r.upload_u32_storage("legacy-control-values", v).unwrap())
                .collect();
            let plane_words = cells + 8;
            let words = 16 + 8 * plane_words + 16;
            let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("legacy-control-staging"),
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
            let mut planes = Vec::new();
            for (channel, data) in source.iter().enumerate() {
                for cell in 0..cells {
                    let start = starts[cell];
                    let end = starts.get(cell + 1).copied().unwrap_or(points as u32);
                    let index = if end <= start { 0 } else { indices[start as usize] as usize };
                    planes.push(if channel < channels { data[index] } else { 0xdead_beef });
                }
                planes.extend_from_slice(&[0xdead_beef; 8]);
            }
            for old in [false, true] {
                let outputs: Vec<_> = (0..4)
                    .map(|_| {
                        r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("legacy-control-output"),
                            contents: bytemuck::cast_slice(&vec![0xdead_beef_u32; plane_words]),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        })
                    })
                    .collect();
                if width == 4 {
                    let inputs = std::array::from_fn(|i| &values[i]);
                    let targets = std::array::from_fn(|i| &outputs[i]);
                    if old {
                        baseline::baseline_record_voxel_gather_multi4_pass(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &targets,
                            channels as u32,
                        )
                        .unwrap();
                    } else {
                        record_voxel_gather_multi4_pass_with_controls(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &targets,
                            channels as u32,
                            &mut controls,
                        )
                        .unwrap();
                    }
                } else {
                    let inputs = std::array::from_fn(|i| &values[i]);
                    let targets = std::array::from_fn(|i| &outputs[i]);
                    if old {
                        baseline::baseline_record_voxel_gather_multi2_pass(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &targets,
                            channels as u32,
                        )
                        .unwrap();
                    } else {
                        record_voxel_gather_multi2_pass_with_controls(
                            &mut encoder,
                            &r,
                            &inputs,
                            &segments,
                            &targets,
                            channels as u32,
                            &mut controls,
                        )
                        .unwrap();
                    }
                }
                for (channel, output) in outputs.iter().enumerate() {
                    encoder.copy_buffer_to_buffer(
                        output,
                        0,
                        &staging,
                        (16 + (usize::from(old) * 4 + channel) * plane_words) as u64 * 4,
                        plane_words as u64 * 4,
                    );
                }
                expected.extend_from_slice(&planes);
            }
            expected.extend_from_slice(&[0xdead_beef; 16]);
            results.push((staging, expected));
            contexts += 1;
        }
    }
    // Changed parameters never rewrite a uniform used by an earlier pass.
    r.queue().submit(Some(encoder.finish()));
    let mut retained = Vec::new();
    let mut checked = 0;
    for (staging, expected) in results {
        let actual: Vec<_> =
            crate::readback::read_staging_f32(r.device(), &staging, expected.len())
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
        assert_eq!(actual, expected);
        checked += actual.len();
        retained.push((actual, expected));
    }
    drop(controls);
    r.clear_buffer_pool();
    drop(r);
    for (actual, expected) in retained {
        assert_eq!(actual, expected);
    }
    eprintln!("VOXEL_LEGACY_CONTROLS_QUEUED verified {checked} guarded original/CPU words across {contexts} contexts / {} queued passes and runtime drop", contexts * 2);
}

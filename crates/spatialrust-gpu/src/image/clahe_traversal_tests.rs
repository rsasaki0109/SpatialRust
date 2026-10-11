use super::*;

// Repeated signed reflection is deliberately independent of shader modulo and
// lane assignment. Keep raw counts observable: clipping/LUT rounding can conceal
// missing pixels, especially in constant images or highly clipped histograms.
fn reflected(mut position: i64, length: u32) -> u32 {
    if length == 1 {
        return 0;
    }
    while position < 0 || position >= i64::from(length) {
        position = if position < 0 { -position } else { 2 * (i64::from(length) - 1) - position };
    }
    position as u32
}

#[test]
fn traversal_counts_cover_lane_boundaries_reflected_padding_and_reused_workgroups() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let pipelines = ClahePipelines::new(runtime.device());
    // Use the production traversal verbatim, replacing only the final observable
    // result, retaining the production shader's workgroup barriers.
    let shader = LOOKUP_SHADER.replace("mapped(cumulative, scale)", "atomicLoad(&bins[lane])");
    assert_ne!(shader, LOOKUP_SHADER);
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clahe-raw-count-proof"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&pipelines.layouts[1]],
        push_constant_ranges: &[],
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("clahe_lookup"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut compared_bins = 0;
    for (width, height) in [
        (1, 1027),
        (2, 257),
        (3, 257),
        (15, 33),
        (16, 33),
        (17, 33),
        (31, 33),
        (32, 33),
        (33, 33),
        (63, 33),
        (64, 33),
        (65, 33),
        (255, 1),
        (256, 1),
        (257, 1),
        (129, 2),
        (129, 3),
        (1027, 1),
        (257, 259),
    ] {
        for hot in [false, true] {
            let data: Vec<_> = (0..width * height)
                .map(|i| if hot { 43 } else { ((i * 37 + i / 97) % 256) as u8 })
                .collect();
            let image =
                spatialrust_image::Image::<u8, 1>::try_new(width as usize, height as usize, data)
                    .unwrap();
            let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
            let view = source.view();
            for (x, y) in [(1, 1), (3, 5), (17, 19)] {
                let mut p = plan(width, height, 2., x, y, &runtime.device().limits()).unwrap();
                p.params.tile_groups = 1; // One group must clear and reuse bins for every tile.
                let uniform =
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&p.params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                let dirty = vec![u32::MAX; (p.lookup_bytes / 4) as usize];
                let output =
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&dirty),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
                let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipelines.layouts[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                    ],
                });
                let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: p.lookup_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = runtime.device().create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, p.lookup_bytes);
                runtime.queue().submit(Some(encoder.finish()));
                let bytes = read_staging_bytes(runtime.device(), &staging, p.lookup_bytes as usize)
                    .unwrap();
                for tile in 0..p.params.tile_count {
                    let ox = tile % x * p.params.tile_width;
                    let oy = tile / x * p.params.tile_height;
                    let mut expected = [0_u32; 256];
                    for row in 0..p.params.tile_height {
                        for column in 0..p.params.tile_width {
                            let index = reflected(i64::from(oy + row), height) * width
                                + reflected(i64::from(ox + column), width);
                            expected[image.as_slice()[index as usize] as usize] += 1;
                        }
                    }
                    for (bin, expected) in expected.into_iter().enumerate() {
                        let offset = (tile as usize * 256 + bin) * 4;
                        let actual =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        assert_eq!(
                            actual, expected,
                            "{width}x{height}, grid{x}x{y}, hot{hot}, tile{tile}, bin{bin}"
                        );
                        compared_bins += 1;
                    }
                }
            }
            source.recycle(&runtime);
        }
    }
    assert_eq!(compared_bins, 3_297_792);
    eprintln!("Verified {compared_bins} raw histogram bins across 114 padded tile layouts");
}

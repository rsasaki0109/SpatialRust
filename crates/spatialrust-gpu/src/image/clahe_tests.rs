use super::super::gpu_image::read_staging_bytes;
use super::*;

#[test]
fn clahe_plan_bounds_grid_padding_buffers_and_device_dispatch() {
    let limits = wgpu::Limits::default();
    let p = plan(3840, 2160, 2.0, 8, 8, &limits).unwrap();
    assert_eq!((p.params.tile_width, p.params.tile_height), (480, 270));
    assert_eq!(p.lookup_bytes, 64 * 1024);
    assert_eq!(p.geometry_bytes, 6000 * 16);
    assert_eq!(p.params.clip_count, 1012);
    assert_eq!((p.params.groups_x, p.params.groups_y), (64, 64));
    let p = plan(1, 1, f64::MAX, 17, 19, &limits).unwrap();
    assert_eq!(p.params.clip_count, 1);
    assert_eq!(p.params.tile_groups, 256);
    assert_eq!(plan(17, 19, 0.0, 5, 7, &limits).unwrap().params.clip_count, 0);
    assert_eq!(plan(17, 19, f64::MIN_POSITIVE, 5, 7, &limits).unwrap().params.clip_count, 1);
    let mut reduced = limits.clone();
    reduced.max_compute_workgroups_per_dimension = 3;
    let p = plan(3840, 2160, 2.0, 8, 8, &reduced).unwrap();
    assert_eq!(
        (p.params.tile_groups, p.params.geometry_groups, p.params.groups_x, p.params.groups_y),
        (3, 3, 3, 3)
    );
    for (w, h, x, y) in [
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (1, 1, u32::MAX, u32::MAX),
        (8193, 1, 1, 1),
    ] {
        assert!(plan(w, h, 2.0, x, y, &limits).is_err());
    }
    for clip in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(plan(17, 19, clip, 5, 7, &limits).is_err());
    }
    for field in 0..11 {
        let mut invalid = limits.clone();
        match field {
            0 => invalid.max_compute_invocations_per_workgroup = 255,
            1 => invalid.max_compute_workgroup_size_x = 255,
            2 => invalid.max_compute_workgroup_size_y = 15,
            3 => invalid.max_compute_workgroup_storage_size = 4095,
            4 => invalid.max_compute_workgroups_per_dimension = 0,
            5 => invalid.max_storage_buffers_per_shader_stage = 1,
            6 => invalid.max_sampled_textures_per_shader_stage = 0,
            7 => invalid.max_storage_textures_per_shader_stage = 0,
            8 => invalid.max_uniform_buffer_binding_size = 47,
            9 => invalid.max_buffer_size = 1023,
            _ => invalid.max_storage_buffer_binding_size = 1023,
        }
        assert!(plan(17, 19, 2.0, 5, 7, &invalid).is_err());
    }
    let mut huge = limits;
    huge.max_texture_dimension_2d = u32::MAX;
    assert!(plan(65536, 65536, 2.0, 1, 1, &huge).is_err());
    assert!(plan(u32::MAX, 1, 2.0, 1, 1, &huge).is_err());
    assert!(plan(1, 1, 2.0, 131073, 1, &huge).is_err());
}

#[test]
fn gpu_positive_binary64_and_geometry_match_cpu_bits() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let constants = [
        0.0_f64,
        0.5,
        1.0,
        255.0,
        1.0 - f64::EPSILON / 2.0,
        1.0 + f64::EPSILON,
        2.0_f64.powi(-32),
        2.0_f64.powi(-64),
    ];
    let mut seed = 19_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let words = |value: f64| {
        let b = value.to_bits();
        [b as u32, (b >> 32) as u32]
    };
    for i in 0..65_536 {
        let mut random = || {
            let exponent = i64::from(next() % 80) - 64 + 1023;
            f64::from_bits(
                ((exponent as u64) << 52) | (u64::from(next() & 0xfffff) << 32) | u64::from(next()),
            )
        };
        let a = if i % 11 == 0 { constants[i % constants.len()] } else { random() };
        let b = if i % 13 == 0 { constants[(i / 7) % constants.len()] } else { random() };
        let (n, d) = if i < 32768 {
            ((i % 257) as u32, (i / 257 + 1) as u32)
        } else {
            (if i % 17 == 0 { 0 } else { next() }, next().max(1))
        };
        let tiles = next() % 4096 + 1;
        inputs.extend(words(a));
        inputs.extend(words(b));
        inputs.extend([n, d, tiles, 0]);
        for value in [a + b, a * b, (a - b).abs(), f64::from(n) / f64::from(d)] {
            expected.extend(words(value));
        }
        let tx = f64::from(n) / f64::from(d) - 0.5;
        let raw = tx.floor() as i64;
        expected.extend([
            raw.clamp(0, i64::from(tiles) - 1) as u32,
            (raw + 1).clamp(0, i64::from(tiles) - 1) as u32,
        ]);
        expected.extend(words(tx - tx.floor()));
    }
    let shader = format!(
        "{GEOMETRY_SHADER}\n{}",
        r#"
struct TestInput { a: vec2<u32>, b: vec2<u32>, n: u32, d: u32, tiles: u32, padding: u32, };
struct TestOutput { sum: vec2<u32>, product: vec2<u32>, difference: vec2<u32>, ratio: vec2<u32>, geometry: vec4<u32>, };
@group(0) @binding(2) var<storage, read> test_inputs: array<TestInput>;
@group(0) @binding(3) var<storage, read_write> test_outputs: array<TestOutput>;
@compute @workgroup_size(256)
fn prove_binary64(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&test_inputs) { return; }
    let i = test_inputs[id.x];
    let a = fp_decode(i.a); let b = fp_decode(i.b);
    test_outputs[id.x] = TestOutput(fp_encode(fp_sum(a,b)), fp_encode(fp_product(a,b)),
        fp_encode(fp_difference(a,b)), fp_encode(fp_ratio(i.n,i.d)), axis_geometry(i.n,i.d,i.tiles));
}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clahe-binary64-proof"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clahe-binary64-proof"),
        layout: None,
        module: &module,
        entry_point: Some("prove_binary64"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = (expected.len() * 4) as u64;
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
    runtime.queue().submit(Some(encoder.finish()));
    let bytes = read_staging_bytes(runtime.device(), &staging, size as usize).unwrap();
    for (i, (actual, expected)) in bytes.chunks_exact(4).zip(expected).enumerate() {
        assert_eq!(
            u32::from_le_bytes(actual.try_into().unwrap()),
            expected,
            "row {}, field {}, input {:?}",
            i / 12,
            i % 12,
            &inputs[i / 12 * 8..i / 12 * 8 + 8]
        );
    }
    eprintln!("Verified 65536 GPU arithmetic/ratio/geometry rows against CPU binary64 bits");
}

#[test]
fn production_interpolation_matches_cpu_at_half_boundaries_and_equal_corners() {
    use spatialrust_image::Image;
    let runtime = WgpuRuntime::new_headless().unwrap();
    let width = 257;
    let height = 257;
    let cases: [[u32; 4]; 16] = [
        [0, 1, 0, 1],
        [1, 2, 1, 2],
        [127, 128, 127, 128],
        [126, 127, 126, 127],
        [0, 255, 0, 255],
        [0, 255, 255, 0],
        [0, 254, 1, 255],
        [255, 1, 2, 254],
        [0, 0, 0, 0],
        [43, 43, 43, 43],
        [255, 255, 255, 255],
        [127, 127, 128, 128],
        [17, 33, 96, 71],
        [254, 255, 254, 255],
        [1, 0, 0, 1],
        [128, 127, 127, 128],
    ];
    let weights = |length: usize, divisor: f64| -> Vec<f64> {
        (0..length)
            .map(|i| match i {
                0..=15 => 0.5,
                16 => 0.0,
                17 => 1.0,
                18 => 0.5 - f64::EPSILON / 2.0,
                19 => 0.5 + f64::EPSILON / 2.0,
                20 => 1.0 - f64::EPSILON / 2.0,
                _ => {
                    let tx = i as f64 / divisor - 0.5;
                    tx - tx.floor()
                }
            })
            .collect()
    };
    let horizontal = weights(width, 50.0);
    let vertical = weights(height, 74.0);
    let mut geometry = Vec::new();
    for &weight in horizontal.iter().chain(&vertical) {
        let bits = weight.to_bits();
        geometry.extend([0, 1, bits as u32, (bits >> 32) as u32]);
    }
    let lookups: Vec<_> = (0..4)
        .flat_map(|corner| (0..256).map(move |value| cases[value % cases.len()][corner]))
        .collect();
    let pixels: Vec<_> = (0..width * height).map(|i| (i % cases.len()) as u8).collect();
    let mut expected = Vec::new();
    for (y, &ya) in vertical.iter().enumerate() {
        for (x, &xa) in horizontal.iter().enumerate() {
            let [a, b, c, d] = cases[usize::from(pixels[y * width + x])].map(f64::from);
            let top = a * (1.0 - xa) + b * xa;
            let bottom = c * (1.0 - xa) + d * xa;
            let value = top * (1.0 - ya) + bottom * ya;
            let floor = value.floor();
            let fraction = value - floor;
            let rounded = floor as u32
                + u32::from(fraction > 0.5 || (fraction == 0.5 && floor as u32 % 2 != 0));
            expected.push(rounded.min(255) as u8);
        }
    }
    assert_eq!(&expected[..4], &[0, 2, 128, 126]);
    let host = Image::<u8, 1>::try_new(width, height, pixels).unwrap();
    let source = GpuImage::upload_u8(&runtime, host.view()).unwrap();
    let p = plan(width as u32, height as u32, 0.0, 2, 2, &runtime.device().limits()).unwrap();
    let pipelines = ClahePipelines::new(runtime.device());
    let make = |contents: &[u8], usage| {
        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clahe-interpolation-proof"),
            contents,
            usage,
        })
    };
    // Test-only LUT/geometry upload isolates the actual application shader from
    // histogram construction. The production API builds both entirely on GPU.
    let geometry = make(bytemuck::cast_slice(&geometry), wgpu::BufferUsages::STORAGE);
    let lookups = make(bytemuck::cast_slice(&lookups), wgpu::BufferUsages::STORAGE);
    let uniform = make(bytemuck::bytes_of(&p.params), wgpu::BufferUsages::UNIFORM);
    let texture =
        create_texture(&runtime, width as u32, height as u32, "clahe-interpolation-proof");
    let source_view = source.view();
    let output_view = texture.create_view(&Default::default());
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipelines.layouts[2],
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: geometry.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lookups.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipelines.pipelines[2]);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.params.groups_x, p.params.groups_y, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    let mut result = GpuImage::from_parts(
        &runtime,
        width as u32,
        height as u32,
        1,
        texture,
        host.metadata(),
        Default::default(),
    )
    .unwrap();
    let actual = result.readback_u8::<1>(&runtime).unwrap();
    for (i, (&a, &b)) in actual.as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(
            a,
            b,
            "interpolation pixel ({},{}), corners {:?}, weights {}, {}",
            i % width,
            i / width,
            cases[i % cases.len()],
            horizontal[i % width],
            vertical[i / width]
        );
    }
    result.recycle(&runtime);
    source.recycle(&runtime);
    eprintln!("Verified 66049 production interpolation pixels, including exact/neighboring half boundaries and equal corners");
}

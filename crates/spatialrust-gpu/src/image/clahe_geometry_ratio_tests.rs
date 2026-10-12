use super::*;

#[test]
fn power_two_ratios_and_geometry_match_cpu_and_frozen_words() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut cases = Vec::new();
    for d in 1..=256u32 {
        for n in 0..=1024u32 {
            cases.push([n, d, n % 4096 + 1, 0]);
        }
    }
    let mut seed = 19u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for bit in 0..32 {
        let d = 1u32 << bit;
        for n in [0, 1, 2, 3, d - 1, d, d.saturating_add(1), u32::MAX - 1, u32::MAX] {
            cases.push([n, d, 1, 0]);
            cases.push([n, d, u32::MAX, 0]);
        }
        for _ in 0..128 {
            cases.push([next(), d, next().max(1), 0]);
        }
    }
    for d in
        [3u32, 5, 7, 15, 30, 63, 80, 135, 240, 270, 480, 1023, 1025, 65535, (1 << 31) - 1, u32::MAX]
    {
        for multiplier in [1u32, 2, 3, 17, 255, 1024, u32::MAX / d] {
            if let Some(n) = d.checked_mul(multiplier) {
                for p in [n.saturating_sub(1), n, n.saturating_add(1)] {
                    cases.push([p, d, next().max(1), 0]);
                }
            }
        }
    }
    for i in 0..65536 {
        let n = if i % 17 == 0 { 0 } else { next() };
        cases.push([n, next().max(1), next().max(1), 0]);
    }
    let mut expected = Vec::with_capacity(cases.len() * 6);
    let words = |f: f64| {
        let b = f.to_bits();
        [b as u32, (b >> 32) as u32]
    };
    for &[n, d, tiles, _] in &cases {
        let ratio = f64::from(n) / f64::from(d);
        let coordinate = ratio - 0.5;
        let raw = coordinate.floor() as i64;
        expected.extend(words(ratio));
        expected.extend([
            raw.clamp(0, i64::from(tiles) - 1) as u32,
            (raw + 1).clamp(0, i64::from(tiles) - 1) as u32,
        ]);
        expected.extend(words(coordinate - coordinate.floor()));
    }
    let body = r#"
struct RatioInput { n: u32, d: u32, tiles: u32, padding: u32, };
struct RatioOutput { ratio: vec2<u32>, geometry: vec4<u32>, };
@group(0) @binding(2) var<storage, read> inputs: array<RatioInput>;
@group(0) @binding(3) var<storage, read_write> outputs: array<RatioOutput>;
@compute @workgroup_size(256)
fn verify_ratios(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x>=arrayLength(&inputs) {return;}
    let p=inputs[id.x];
    outputs[id.x]=RatioOutput(fp_encode(fp_ratio(p.n,p.d)),axis_geometry(p.n,p.d,p.tiles));
}
"#;
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // vec4 alignment pads RatioOutput to 32 bytes, including a word pair after ratio.
    let size = cases.len() as u64 * 32;
    for (variant, shader) in
        [("current", GEOMETRY_SHADER), ("frozen", super::geometry_benchmark::PREVIOUS_GEOMETRY)]
    {
        let source = format!("{shader}\n{body}");
        let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("verify_ratios"),
            compilation_options: Default::default(),
            cache: None,
        });
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
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
        runtime.queue().submit(Some(encoder.finish()));
        let bytes = read_staging_bytes(runtime.device(), &staging, size as usize).unwrap();
        for (i, row) in bytes.chunks_exact(32).enumerate() {
            let actual: Vec<_> = [0usize, 4, 16, 20, 24, 28]
                .iter()
                .map(|&start| u32::from_le_bytes(row[start..start + 4].try_into().unwrap()))
                .collect();
            assert_eq!(
                actual,
                &expected[i * 6..i * 6 + 6],
                "variant={variant}, input={:?}",
                cases[i]
            );
        }
    }
    eprintln!("Verified {} ratio/axis cases against CPU and frozen shader, {} exact result words per variant",cases.len(),expected.len());
}

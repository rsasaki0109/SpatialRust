use super::*;

#[test]
fn double_single_arithmetic_matches_independent_binary64_values() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut inputs: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 3.0, 0.0],
        [1.0, 2.0f32.powi(-24), 1.0, -2.0f32.powi(-24)],
        [16777216.0, 0.0, 1.0, 0.0],
        [8.2388, 1e-7, 209642.0, 0.0],
        [-2548.44, 0.00007, 209642.0, 0.0],
        [-64.0, 0.0, 128.7, 0.0],
        [-0.78, 1e-8, 131.25, 1e-7],
    ];
    let mut seed = 20261013u32;
    for i in 0..512 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = f32::from_bits(
            ((100 + (i % 50)) as u32) << 23 | (seed & 0x7fffff) | ((i as u32 & 1) << 31),
        );
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = f32::from_bits(
            ((100 + ((i * 7) % 50)) as u32) << 23
                | (seed & 0x7fffff)
                | (((i as u32 >> 1) & 1) << 31),
        );
        inputs.push([a, a * 2.0f32.powi(-27), b, -b * 2.0f32.powi(-27)]);
    }
    let source = include_str!("../shaders/icp_reduction.wgsl");
    let start = source.find("fn add(").unwrap();
    let end = source.find("fn valid_match(").unwrap();
    let shader=format!("struct Params {{ unused:u32,count:u32,pad0:u32,pad1:u32 }}\n@group(0) @binding(0) var<uniform> params:Params;\n@group(0) @binding(1) var<storage,read> inputs:array<vec4<f32>>;\n@group(0) @binding(2) var<storage,read_write> output:array<vec2<f32>>;\n{}\n{}\n@compute @workgroup_size(64) fn test_math(@builtin(global_invocation_id) id:vec3<u32>) {{ if id.x>=params.count {{ return; }} let v=inputs[id.x]; output[4u*id.x]=precise_add(v.xy,v.z); output[4u*id.x+1u]=two_product(v.x,v.z); output[4u*id.x+2u]=ds_divide(v.xy,v.z); output[4u*id.x+3u]=ds_product(v.xy,v.zw); }}", &source[start..end],include_str!("../shaders/icp_trim_moments.wgsl"));
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("test_math"),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0u32, inputs.len() as u32, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: inputs.len() as u64 * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let entries: Vec<_> = [&uniform, &input, &output]
        .iter()
        .enumerate()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    dispatch(&mut encoder, &pipeline, &group, (inputs.len() as u32).div_ceil(64), 1);
    runtime.queue().submit(Some(encoder.finish()));
    let words = read(&runtime, &output, output.size());
    let mut maximum_relative_error = 0.0_f64;
    for (i, v) in inputs.iter().enumerate() {
        let a = f64::from(v[0]) + f64::from(v[1]);
        let b = f64::from(v[2]) + f64::from(v[3]);
        let expected =
            [a + f64::from(v[2]), f64::from(v[0]) * f64::from(v[2]), a / f64::from(v[2]), a * b];
        for (op, expected) in expected.into_iter().enumerate() {
            let hi = f32::from_bits(words[8 * i + 2 * op]);
            let lo = f32::from_bits(words[8 * i + 2 * op + 1]);
            let actual = f64::from(hi) + f64::from(lo);
            maximum_relative_error =
                maximum_relative_error.max((actual - expected).abs() / expected.abs().max(1e-12));
            assert!(
                (actual - expected).abs() <= 1e-11 * expected.abs().max(1e-12),
                "input={i} operation={op}"
            );
        }
    }
    eprintln!(
        "GPU_ICP_PRECISE_MATH comparisons={} maximum_relative_error={maximum_relative_error}",
        inputs.len() * 4
    );
}

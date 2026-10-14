use super::*;

#[test]
#[ignore = "requires native GPU adapter"]
fn arbitrary_candidates_preserve_cpu_and_frozen_quotient_rounding() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut inputs: Vec<[u32; 8]> = Vec::new();
    let mut expected = Vec::new();
    let mut push = |n: f64, d: f64, candidate: u32| {
        assert!(n == 0.0 || n.is_normal());
        assert!(d.is_normal() && d > 0.0);
        let (n, d) = (n.to_bits(), d.to_bits());
        inputs.push([n as u32, (n >> 32) as u32, d as u32, (d >> 32) as u32, candidate, 0, 0, 0]);
        expected.push((f64::from_bits(n) / f64::from_bits(d)).round().clamp(0.0, 255.0) as u32);
    };
    // Every proposed byte, including intentionally distant wrong guesses,
    // at all half-byte thresholds and predecessor/successor numerators.
    for k in 0..255 {
        for d in [1.0, 3.0, f64::from_bits(0x3ff0_0000_ffff_ffff)] {
            let center = ((k as f64 + 0.5) * d).to_bits();
            for offset in -1i64..=1 {
                let n = f64::from_bits((center as i64 + offset) as u64);
                for candidate in 0..=255 {
                    push(n, d, candidate);
                }
            }
        }
    }
    // All reachable exponent gaps, extreme normalized limbs and zero numerator.
    for gap in -1022..=1023 {
        for bits in [0u64, 0xffff_ffff, 0x000f_ffff_ffff_ffff] {
            let n = f64::from_bits(((gap + 1023) as u64) << 52 | bits);
            for d in [1.0, f64::from_bits(0x3fff_ffff_ffff_ffff)] {
                for candidate in [0, 1, 127, 254, 255] {
                    push(n, d, candidate);
                }
            }
        }
    }
    for k in 0..255 {
        for d in [2.0_f64.powi(-1000), 2.0_f64.powi(900), 1.0 + f64::EPSILON] {
            let center = ((k as f64 + 0.5) * d).to_bits();
            for offset in -3i64..=3 {
                push(f64::from_bits((center as i64 + offset) as u64), d, k);
            }
        }
    }
    for candidate in 0..=255 {
        push(0.0, f64::MIN_POSITIVE, candidate);
        push(f64::MIN_POSITIVE, f64::MAX, candidate);
        push(f64::MAX, f64::MIN_POSITIVE, candidate);
    }
    let mut reference = include_str!("bilateral_round_reference.wgsl").to_string();
    for name in ["product128", "at_least128", "rounded_ratio_reaches", "bilateral_byte"] {
        reference = reference.replace(name, &format!("reference_{name}"));
    }
    let shader = format!(
        "{}\n{}\n{}\n{}\n{}",
        include_str!("../rounding_u32.wgsl"),
        include_str!("../binary64_positive.wgsl"),
        include_str!("bilateral_round.wgsl"),
        reference,
        r#"
struct Input {bits:vec4<u32>,candidate:u32,pad0:u32,pad1:u32,pad2:u32,};
@group(0) @binding(0) var<storage,read> test_inputs:array<Input>;
@group(0) @binding(1) var<storage,read_write> test_outputs:array<vec4<u32>>;
@compute @workgroup_size(256)
fn proof(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&test_inputs) {return;}
    let v=test_inputs[id.x];let n=fp_decode(v.bits.xy);let d=fp_decode(v.bits.zw);
    var forced=0u;if any(n.mantissa!=vec2<u32>(0u)) {forced=bilateral_byte_bracket(n,d,v.candidate);}
    test_outputs[id.x]=vec4<u32>(bilateral_byte(n,d),reference_bilateral_byte(n,d),forced,0u);
}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bilateral-candidate-proof"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("proof"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (inputs.len() * 16) as u64;
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let actual = read_staging_bytes(runtime.device(), &staging, bytes as usize).unwrap();
    for (index, (row, &expected)) in actual.chunks_exact(16).zip(&expected).enumerate() {
        for (stage, word) in row.chunks_exact(4).take(3).enumerate() {
            assert_eq!(
                u32::from_le_bytes(word.try_into().unwrap()),
                expected,
                "row={index} stage={stage} input={:?}",
                inputs[index]
            );
        }
    }
    println!(
        "GPU_BILATERAL_CANDIDATE_PROOF rows={} exact_outputs={}",
        inputs.len(),
        inputs.len() * 3
    );
}

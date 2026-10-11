use super::super::gpu_image::read_staging_bytes;
use super::*;

// Independent serial CPU clipping, explicit residual positions and cumulative
// counts, including counters that cannot be exercised with a small GPU texture.
fn reference(counts: &[u32], limit: u32) -> Vec<u32> {
    let area = counts.iter().map(|&n| u64::from(n)).sum::<u64>();
    assert!((1..=u64::from(u32::MAX)).contains(&area));
    let mut bounded: Vec<u64> =
        counts.iter().map(|&n| u64::from(if limit == 0 { n } else { n.min(limit) })).collect();
    let excess = area - bounded.iter().sum::<u64>();
    for n in &mut bounded {
        *n += excess / 256;
    }
    let residual = (excess % 256) as usize;
    let step = 256_usize.checked_div(residual).unwrap_or(1).max(1);
    for bin in (0..256).step_by(step).take(residual) {
        bounded[bin] += 1;
    }
    let mut count = 0;
    let scale = 255.0 / area as f64;
    bounded
        .into_iter()
        .flat_map(|n| {
            count += n;
            let value = count as f64 * scale;
            let floor = value.floor();
            let fraction = value - floor;
            let rounded = if fraction > 0.5 || (fraction == 0.5 && floor as u32 % 2 == 1) {
                floor + 1.0
            } else {
                floor
            };
            [count as u32, rounded.min(255.0) as u32]
        })
        .collect()
}

#[test]
fn parallel_clahe_scan_matches_serial_counts_all_residuals_and_u32_limits() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut input = Vec::new();
    let mut expected = Vec::new();
    let mut seed = 53_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for row in 0..4096 {
        let mut counts = [0_u32; 256];
        let limit = if row < 1536 {
            let hot = [0, 127, 255][row / 256 % 3];
            counts[hot] = if row < 768 {
                65_537 + (row % 256) as u32
            } else {
                u32::MAX - 255 + (row % 256) as u32
            };
            1
        } else {
            let area = match row % 7 {
                0 => 1,
                1 => 50,
                2 => 255,
                3 => 256,
                4 => 257,
                5 => u32::MAX,
                _ => next().max(1),
            };
            let mut remaining = area;
            let offset = next() as usize % 256;
            for bin in 0..255 {
                let n = match row % 3 {
                    0 => remaining / (256 - bin) as u32,
                    1 => (u64::from(remaining) * u64::from(next() % 65536) / 65536) as u32,
                    _ => remaining.min(next() % 65536),
                };
                counts[(bin + offset) % 256] = n;
                remaining -= n;
            }
            counts[(255 + offset) % 256] = remaining;
            [0, 1, 17, area / 256, area / 2, area, u32::MAX][row / 7 % 7]
        };
        let area = counts.iter().map(|&n| u64::from(n)).sum::<u64>() as u32;
        input.extend([area, limit, 0, 0]);
        input.extend(counts);
        let result = reference(&counts, limit);
        assert_eq!(result[510], area);
        expected.extend(result);
    }
    let source = format!(
        "{}\n{}\n{}",
        include_str!("rounding_u32.wgsl"),
        include_str!("clahe_clip_scan.wgsl"),
        r#"
struct Input { area: u32, limit: u32, pad: vec2<u32>, counts: array<u32,256>, };
@group(0) @binding(0) var<storage,read> inputs: array<Input>;
@group(0) @binding(1) var<storage,read_write> outputs: array<vec2<u32>>;
var<workgroup> scale: vec4<u32>;
@compute @workgroup_size(256)
fn prove_scan(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    // Several histograms per workgroup also exercise shared-memory reuse.
    for (var row = group.x; row < arrayLength(&inputs); row += 64u) {
        let count = clahe_cumulative(lane, inputs[row].counts[lane], inputs[row].limit);
        if lane == 0u { scale = reciprocal_info(inputs[row].area); }
        workgroupBarrier();
        outputs[row*256u+lane] = vec2<u32>(count,mapped(count,scale));
        workgroupBarrier();
    }
}
"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clahe-parallel-scan-proof"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clahe-parallel-scan-proof"),
        layout: None,
        module: &module,
        entry_point: Some("prove_scan"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&input),
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
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
    runtime.queue().submit(Some(encoder.finish()));
    let bytes = read_staging_bytes(runtime.device(), &staging, size as usize).unwrap();
    for (i, (actual, expected)) in bytes.chunks_exact(4).zip(expected).enumerate() {
        assert_eq!(
            u32::from_le_bytes(actual.try_into().unwrap()),
            expected,
            "histogram {}, bin {}, field {}",
            i / 512,
            i / 2 % 256,
            i % 2
        );
    }
    eprintln!("Verified 4096 histograms / 1048576 bins, cumulative and LUT words, all clipping residuals and u32::MAX mass");
}

#[path = "clahe_lookup_benchmark.rs"]
mod benchmark;

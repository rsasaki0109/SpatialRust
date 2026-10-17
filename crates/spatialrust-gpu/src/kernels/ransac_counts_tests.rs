use super::*;

#[path = "ransac_counts_reference.rs"]
mod old;

const PARENT: &str = include_str!("ransac_counts_parallel_reference.wgsl");
const CURRENT: &str = include_str!("../shaders/ransac_plane_parallel.wgsl");

fn parent_pipeline(r: &WgpuRuntime, cache: &RansacPlanePipelines) -> wgpu::ComputePipeline {
    let module = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ransac-counts-parent"),
        source: wgpu::ShaderSource::Wgsl(PARENT.into()),
    });
    let layout = r.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn reduction(source: &str) -> &str {
    let start = source.find("        for (var stride").unwrap();
    let end = source[start..].find("        if lane == 0u").unwrap() + start;
    &source[start..end]
}

fn read_words(r: &WgpuRuntime, output: &wgpu::Buffer) -> Vec<u32> {
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, output.size());
    r.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    r.device().poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    words
}

#[test]
fn ransac_counts_actual_reduction_matches_u64_totals_through_maximum_u32() {
    let r = WgpuRuntime::new_headless().unwrap();
    let mut rows = vec![[0u32; 256]];
    for lane in 0..256 {
        let mut row = [0; 256];
        row[lane] = u32::MAX;
        rows.push(row);
    }
    for total in
        [1, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, u32::MAX - 1, u32::MAX]
    {
        for rotation in 0..8 {
            let mut row = [total / 256; 256];
            for i in 0..total % 256 {
                row[((i + rotation * 31) % 256) as usize] += 1;
            }
            rows.push(row);
        }
    }
    for seed in 0..512u32 {
        let total = if seed % 2 == 0 { u32::MAX } else { seed * 7919 };
        let weights: [u64; 256] = std::array::from_fn(|i| {
            u64::from(
                (i as u32).wrapping_mul(747_796_405).wrapping_add(seed * 7919).rotate_left(13)
                    & 65535,
            ) + 1
        });
        let mass: u64 = weights.iter().sum();
        let mut row = weights.map(|w| (w * u64::from(total) / mass) as u32);
        let used: u64 = row.iter().map(|&v| u64::from(v)).sum();
        row[seed as usize % 256] += (u64::from(total) - used) as u32;
        rows.push(row);
    }
    let expected: Vec<u32> = rows
        .iter()
        .map(|row| {
            let sum: u64 = row.iter().map(|&v| u64::from(v)).sum();
            u32::try_from(sum).unwrap()
        })
        .collect();
    let input = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&rows),
        usage: wgpu::BufferUsages::STORAGE,
    });
    for source in [PARENT, CURRENT] {
        // Execute the actual production loop, with raw integer inputs instead of distances.
        let shader = format!("@group(0) @binding(0) var<storage,read> input:array<u32>;\n@group(0) @binding(1) var<storage,read_write> output:array<u32>;\nvar<workgroup> counts:array<u32,256>;\n@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {{\ncounts[lane]=input[group.x*256u+lane]; workgroupBarrier();\n{}\nif lane==0u {{ output[group.x+4u]=counts[0]; }}\n}}", reduction(source));
        let module = r.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = r.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dirty = vec![0xdead_beefu32; rows.len() + 8];
        let output = r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&dirty),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let group = r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = r.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(rows.len() as u32, 1, 1);
        }
        r.queue().submit(Some(encoder.finish()));
        let words = read_words(&r, &output);
        assert_eq!(&words[4..4 + rows.len()], expected);
        assert_eq!(&words[..4], &[0xdead_beef; 4]);
        assert_eq!(&words[4 + rows.len()..], &[0xdead_beef; 4]);
    }
    eprintln!("RANSAC_COUNTS_SUM verified {} independent u64 totals per shader, maximum u32 mass, every single active lane and guarded outputs", rows.len());
}

#[test]
fn ransac_counts_parallel_scores_match_parent_words_and_independent_axis_counts() {
    let r = WgpuRuntime::new_headless().unwrap();
    let cache = RansacPlanePipelines::new(r.device());
    let parent = parent_pipeline(&r, &cache);
    let current = cache.pipeline(r.device(), true);
    let mut words = 0;
    for (n, h, cap) in [
        (1, 7, 1),
        (3, 7, 1),
        (255, 513, 1),
        (256, 513, 1),
        (257, 513, 1),
        (1023, 513, 3),
        (1024, 513, 256),
        (1025, 513, 1),
        (4097, 513, 3),
        (65537, 33, 1),
        (1000003, 7, 1),
    ] {
        let (axes, hypotheses) = fixture(n, h);
        let mut p = plan(n as usize, h as usize, &r.device().limits()).unwrap();
        p.groups = h.min(cap);
        for threshold in
            [0., 0.25, f32::from_bits(0.25f32.to_bits() + 1), -1., f32::NAN, f32::INFINITY]
        {
            let (outputs, groups) = bindings(&r, &cache, &p, &axes, &hypotheses, threshold);
            let mut encoder = r.device().create_command_encoder(&Default::default());
            for (i, pipeline) in [&parent, current].iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &groups[i], &[]);
                pass.dispatch_workgroups(p.groups, 1, 1);
            }
            r.queue().submit(Some(encoder.finish()));
            let before = read_scores(&r, &outputs[0], p.output_bytes);
            let after = read_scores(&r, &outputs[1], p.output_bytes);
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&before),
                bytemuck::cast_slice::<_, u32>(&after),
                "n={n},h={h},groups={},threshold={threshold}",
                p.groups
            );
            if n >= 7 {
                check_axes(&axes, threshold, &after);
            }
            words += before.len() * 10;
        }
    }
    eprintln!("RANSAC_COUNTS_RAW verified {words} exact parent score words, dirty outputs, degenerate models, threshold boundaries and repeated hypotheses per group");
}

#[test]
fn ransac_counts_complete_parent_api_keeps_scores_pool_reuse_and_owned_outputs() {
    let r = WgpuRuntime::new_headless().unwrap();
    let prior = old::ParentPipelines::new(r.device());
    let mut words = 0;
    for (n, h) in [(7, 17), (1023, 257), (1024, 257), (1025, 257), (8193, 513), (65537, 33)] {
        let (axes, hypotheses) = fixture(n, h);
        let mut retained: Option<Vec<GpuPlaneScore>> = None;
        for threshold in
            [0., 0.25, f32::from_bits(0.25f32.to_bits() + 1), -1., f32::NAN, f32::INFINITY]
        {
            let before =
                old::previous(&prior, &r, &axes[0], &axes[1], &axes[2], &hypotheses, threshold)
                    .unwrap();
            let after = score_ransac_plane_hypotheses_gpu(
                &r,
                &axes[0],
                &axes[1],
                &axes[2],
                &hypotheses,
                threshold,
            )
            .unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&before),
                bytemuck::cast_slice::<_, u32>(&after)
            );
            check_axes(&axes, threshold, &after);
            if let Some(ref previous) = retained {
                check_axes(&axes, 0.25, previous);
            }
            if threshold == 0.25 {
                retained = Some(after);
            }
            words += before.len() * 10;
        }
        let before = r.buffer_pool().stats().allocation_count;
        let scores =
            score_ransac_plane_hypotheses_gpu(&r, &axes[0], &axes[1], &axes[2], &hypotheses, 0.25)
                .unwrap();
        check_axes(&axes, 0.25, &scores);
        assert_eq!(r.buffer_pool().stats().allocation_count, before);
    }
    eprintln!("RANSAC_COUNTS_PUBLIC verified {words} complete-parent score words across serial controls, parallel threshold, reused buffers and retained outputs");
}

#[path = "ransac_counts_benchmark.rs"]
mod benchmark;

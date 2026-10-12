use super::*;
use wgpu::util::DeviceExt;

fn positions(n: usize) -> Vec<f32> {
    // Each six-point band has one singleton, one pair and one triple.
    (0..n).map(|i| (i / 6 * 12 + [0, 4, 4, 8, 8, 8][i % 6]) as f32).collect()
}

fn fallback_positions(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i / 3 * 8 + [0, 4, 4][i % 3]) as f32).collect()
}

fn isotropic(n: usize, epsilon: f32) -> Vec<[f32; 6]> {
    vec![[epsilon, epsilon, epsilon, 0.0, 0.0, 0.0]; n]
}

fn bits(values: &[[f32; 6]]) -> Vec<[u32; 6]> {
    values.iter().map(|v| v.map(f32::to_bits)).collect()
}

fn pipeline(runtime: &WgpuRuntime, source: &str) -> wgpu::ComputePipeline {
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("covariance-output-probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("covariance-output-probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn raw(
    runtime: &WgpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    x: &[f32],
    epsilon: f32,
    poison: u32,
) -> Vec<u32> {
    let n = x.len();
    let cells = *x.last().unwrap() as usize + 1;
    let mut starts = vec![0u32; cells + 1];
    for &value in x {
        starts[value as usize + 1] += 1;
    }
    for i in 1..starts.len() {
        starts[i] += starts[i - 1];
    }
    let sorted: Vec<u32> = (0..n as u32).collect();
    let zero = vec![0f32; n];
    let params = CovUniform {
        origin: [0.0; 4],
        dims: [cells as u32, 1, 1, n as u32],
        inv_cell: 1.0,
        radius_sq: 1.0,
        epsilon,
        _pad: 0.0,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("covariance-output-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let inputs: Vec<_> = [
        bytemuck::cast_slice(x),
        bytemuck::cast_slice(&zero),
        bytemuck::cast_slice(&zero),
        bytemuck::cast_slice(&sorted),
        bytemuck::cast_slice(&starts),
    ]
    .map(|contents| {
        runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("covariance-output-input"),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    })
    .into_iter()
    .collect();
    // One extra vec4 is a canary for the partially occupied final workgroup.
    let words = 8 * n + 4;
    let output = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("covariance-output-dirty"),
        contents: bytemuck::cast_slice(&vec![poison; words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("covariance-output-staging"),
        size: words as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    for (i, buffer) in inputs.iter().chain(std::iter::once(&output)).enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: buffer.as_entire_binding(),
        });
    }
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("covariance-output-bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, words as u64 * 4);
    runtime.queue().submit(Some(encoder.finish()));
    crate::readback::read_staging_f32(runtime.device(), &staging, words)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
fn covariance_fallback_overwrites_dirty_tail_and_preserves_supported_rows() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let old =
        pipeline(&runtime, include_str!("../../tests/fixtures/covariance_output_previous.wgsl"));
    let current = pipeline(&runtime, COV_GRID_WGSL);
    let mut words = 0;
    for n in [1, 2, 3, 6, 7, 31, 255, 256, 257, 1025] {
        let x = positions(n);
        for epsilon in [0.001, 1.0, -0.0] {
            let oracle = bits(&isotropic(n, epsilon));
            for poison in [0x3fc00000, 0xc0100000, 0x7fc00055] {
                let previous = raw(&runtime, &old, &x, epsilon, poison);
                let actual = raw(&runtime, &current, &x, epsilon, poison);
                for i in 0..n {
                    let tail = &actual[(n + i) * 4..(n + i) * 4 + 4];
                    if x.iter().filter(|&&value| value == x[i]).count() < 3 {
                        assert_eq!(&actual[i * 4..i * 4 + 4], &oracle[i][..4]);
                        assert_eq!(&previous[i * 4..i * 4 + 4], &oracle[i][..4]);
                        assert_eq!(
                            tail, &[0; 4],
                            "unwritten covariance tail: n={n}, i={i}, epsilon={epsilon}"
                        );
                        assert_eq!(&previous[(n + i) * 4..(n + i) * 4 + 4], &[poison; 4]);
                    } else {
                        // Degenerate supported samples need not choose one particular
                        // eigenbasis. Preserve the frozen numerical path instead.
                        assert_eq!(&actual[i * 4..i * 4 + 4], &previous[i * 4..i * 4 + 4]);
                        assert_eq!(tail, &previous[(n + i) * 4..(n + i) * 4 + 4]);
                    }
                }
                assert_eq!(&actual[8 * n..], &[poison; 4]);
                assert_eq!(&previous[8 * n..], &[poison; 4]);
                words += actual.len() + previous.len();
            }
        }
    }
    eprintln!("COVARIANCE_OUTPUT verified {words} raw current/frozen output and canary words.");
}

fn seed_pool(runtime: &WgpuRuntime, n: usize, buffers: usize) {
    let bytes = n as u64 * 32;
    let held: Vec<_> = (0..buffers)
        .map(|i| {
            let buffer =
                runtime.scratch_pool.acquire_storage(runtime, "covariance-output-poison", bytes);
            runtime.queue().write_buffer(
                &buffer,
                0,
                bytemuck::cast_slice(&vec![0x7fc00055u32 + i as u32; n * 8]),
            );
            buffer
        })
        .collect();
    // Submit every write before making any lease reusable.
    runtime.queue().submit(std::iter::empty());
    runtime.wait_idle();
    for buffer in held {
        runtime.scratch_pool.recycle(bytes, buffer);
    }
}

#[test]
fn public_covariance_fallback_reuses_poisoned_storage_with_exact_isotropic_values() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [1, 2, 3, 31, 255, 256, 257, 1025] {
        let x = fallback_positions(n);
        let zero = vec![0.0; n];
        for epsilon in [0.001, 1.0, -0.0] {
            runtime.clear_scratch_buffer_pool();
            seed_pool(&runtime, n, 1);
            let before = runtime.scratch_buffer_pool_stats();
            let wanted = bits(&isotropic(n, epsilon));
            let first =
                estimate_plane_covariances_grid_gpu(&runtime, &x, &zero, &zero, 1.0, epsilon)
                    .unwrap();
            assert_eq!(bits(&first), wanted);
            assert!(runtime.scratch_buffer_pool_stats().reuse_count > before.reuse_count);
            let warmed = runtime.scratch_buffer_pool_stats();
            let second =
                estimate_plane_covariances_grid_gpu(&runtime, &x, &zero, &zero, 1.0, epsilon)
                    .unwrap();
            assert_eq!(bits(&second), wanted);
            assert_eq!(bits(&first), wanted);
            let after = runtime.scratch_buffer_pool_stats();
            assert_eq!(after.allocation_count, warmed.allocation_count);
            assert!(after.reuse_count >= warmed.reuse_count + 6);
        }
    }
}

#[test]
fn concurrent_covariance_fallbacks_keep_independent_live_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let n = 257;
    seed_pool(&runtime, n, 8);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let runtime = &runtime;
                scope.spawn(move || {
                    let x = fallback_positions(n);
                    let zero = vec![0.0; n];
                    let epsilon = (worker + 1) as f32 * 0.125;
                    let wanted = bits(&isotropic(n, epsilon));
                    let outputs: Vec<_> = (0..3)
                        .map(|_| {
                            estimate_plane_covariances_grid_gpu(
                                runtime, &x, &zero, &zero, 1.0, epsilon,
                            )
                            .unwrap()
                        })
                        .collect();
                    for output in &outputs {
                        assert_eq!(bits(output), wanted);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

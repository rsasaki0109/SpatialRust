use super::*;
use spatialrust_math::{Quat, Vec3};

#[path = "icp_trim_precision_tests.rs"]
mod precision_tests;

#[path = "icp_trim_math_tests.rs"]
mod math_tests;

fn read(runtime: &WgpuRuntime, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("trim-test-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    runtime.device().poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    words
}

fn tree(points: &[[f32; 3]]) -> KdTree {
    let columns: [Vec<_>; 3] = std::array::from_fn(|i| points.iter().map(|p| p[i]).collect());
    KdTree::from_slices(&columns[0], &columns[1], &columns[2])
}

#[test]
fn trim_plan_and_canonical_shader_anchors_are_checked() {
    assert_eq!(std::mem::size_of::<TrimParams>(), 32);
    let limits = wgpu::Limits::default();
    assert_eq!(plan_trim(65, &limits).unwrap(), 2);
    for case in 0..4 {
        let mut limits = limits.clone();
        match case {
            0 => limits.max_storage_buffers_per_shader_stage = 6,
            1 => limits.max_bindings_per_bind_group = 7,
            2 => limits.max_compute_workgroup_storage_size = 2047,
            _ => limits.max_storage_buffer_binding_size = 2047,
        }
        assert!(plan_trim(65, &limits).is_err());
    }
    assert!(trimmed_shader().contains("result.reserved0=selection.gated"));
}

#[test]
fn integer_fraction_floor_matches_binary64_for_rounding_boundaries_and_large_counts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = TrimPipelines::new(runtime.device());
    let mut counts: Vec<u32> = (0..4097).collect();
    for bit in 1..32 {
        for delta in -3i64..=3 {
            counts.push(((1i64 << bit) + delta).clamp(0, i64::from(u32::MAX)) as u32);
        }
    }
    counts.extend([u32::MAX, u32::MAX - 1, u32::MAX - 2]);
    let mut seed = 20261013u32;
    for _ in 0..10000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        counts.push(seed);
    }
    let matches: Vec<[u32; 2]> = counts.iter().map(|&n| [n, 0]).collect();
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&matches),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: counts.len() as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let selection = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER}\n@compute @workgroup_size(64) fn verify_fraction(@builtin(global_invocation_id) id:vec3<u32>) {{ if id.x<params.count {{ histograms[id.x]=retained_count(matches[id.x].index); }} }}").into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("verify_fraction"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut fractions = vec![f64::from_bits(1), f64::MIN_POSITIVE, 1e-20, 1e-9, 1.0];
    for fraction in [0.1_f64, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.999999, 1.0] {
        fractions.push(f64::from_bits(fraction.to_bits() - 1));
        fractions.push(fraction);
        if fraction < 1.0 {
            fractions.push(f64::from_bits(fraction.to_bits() + 1));
        }
    }
    for fraction in fractions {
        let params = TrimParams::new(counts.len() as u32, 1, fraction, 1.0, 1);
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let entries: Vec<_> = [&uniform, &input, &output, &selection]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        dispatch(&mut encoder, &pipeline, &group, (counts.len() as u32).div_ceil(64), 1);
        runtime.queue().submit(Some(encoder.finish()));
        let actual = read(&runtime, &output, output.size());
        for (&n, &k) in counts.iter().zip(&actual) {
            assert_eq!(k, (f64::from(n) * fraction).floor() as u32, "n={n} fraction={fraction:?}");
        }
    }
}

#[test]
fn stable_radix_selection_matches_independent_sort_with_ties_missing_and_boundaries() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = TrimPipelines::new(runtime.device());
    for n in [1, 3, 63, 64, 65, 257, 4097, 16385, 65537] {
        for kind in 0..3 {
            let matches: Vec<[u32; 2]> = (0..n)
                .map(|i| {
                    let distance = match kind {
                        0 => ((i * 7919 % 1024) as f32) / 1024.0,
                        1 => 0.25,
                        _ => [
                            0.0,
                            -0.0,
                            f32::from_bits(1),
                            f32::from_bits(0x007fffff),
                            0.5,
                            1.0,
                            f32::from_bits(0x3f7fffff),
                        ][i % 7],
                    };
                    if kind != 1 && i % 7 == 0 {
                        [u32::MAX, 1.0f32.to_bits()]
                    } else {
                        [(i % 31) as u32, distance.to_bits()]
                    }
                })
                .collect();
            let mut order: Vec<_> = matches
                .iter()
                .enumerate()
                .filter(|(_, m)| m[0] != u32::MAX)
                .map(|(i, m)| (if f32::from_bits(m[1]) == 0.0 { 0 } else { m[1] }, i as u32))
                .collect();
            order.sort_unstable();
            let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&matches),
                usage: wgpu::BufferUsages::STORAGE,
            });
            for fraction in [0.1, 0.3, 0.5, 0.9, f64::from_bits(1.0f64.to_bits() - 1)] {
                let mut scratch = ScratchBuffers::new(&runtime);
                let selection = scratch.acquire("trim-test-selection", 32);
                let mut encoder = runtime.device().create_command_encoder(&Default::default());
                cache.encode(
                    &runtime,
                    &mut encoder,
                    &mut scratch,
                    TrimParams::new(
                        n as u32,
                        plan_trim(n as u64, &runtime.device().limits()).unwrap(),
                        fraction,
                        1.0,
                        31,
                    ),
                    &input,
                    &selection,
                );
                // Copy before scratch recycling; do not retain a pooled output.
                let owned = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 32,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(&selection, 0, &owned, 0, 32);
                scratch.submit_and_recycle(encoder);
                let actual = read(&runtime, &owned, 32);
                let k = (order.len() as f64 * fraction).floor() as usize;
                assert_eq!((actual[0] as usize, actual[1] as usize), (order.len(), k));
                if k > 0 {
                    assert_eq!(
                        (actual[3], actual[4]),
                        order[k - 1],
                        "n={n} kind={kind} fraction={fraction}"
                    );
                    let chosen: Vec<_> = order
                        .iter()
                        .copied()
                        .filter(|&pair| pair <= (actual[3], actual[4]))
                        .collect();
                    assert_eq!(chosen, order[..k]);
                }
            }
        }
    }
}

#[test]
fn trimmed_moments_preserve_all_gated_fitness_and_stable_input_order() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [3, 63, 64, 65, 257, 4097] {
        let source: Vec<_> = (0..n).map(|i| [0.125, [0.0, 0.0625, -0.0625][i % 3], 0.0]).collect();
        let input = GpuIcpInput::upload(&runtime, &source, &tree(&[[0.0; 3]])).unwrap();
        let all = input
            .moments_resident(&runtime, Isometry3::identity(), 0.25)
            .unwrap()
            .download(&runtime)
            .unwrap();
        for fraction in [0.1, 0.3, 0.5, 0.8, 1.0] {
            let actual = input
                .trimmed_moments_resident(&runtime, Isometry3::identity(), 0.25, fraction)
                .unwrap()
                .download(&runtime)
                .unwrap();
            let mut order: Vec<_> = (0..n).collect();
            order.sort_unstable_by_key(|&i| {
                ((source[i][0] * source[i][0] + source[i][1] * source[i][1]).to_bits(), i)
            });
            let k = (n as f64 * fraction).floor() as usize;
            assert_eq!(actual.moments.correspondences as usize, k);
            assert_eq!(actual.evaluated_correspondences as usize, n);
            assert_eq!(actual.moments.squared_error_sum, all.squared_error_sum);
            assert_eq!(actual.moments.transfers.device_to_host_bytes(), 176);
            for (a, &centroid) in actual.moments.source_centroid.iter().enumerate() {
                let expected = if k == 0 {
                    0.0
                } else {
                    order[..k].iter().map(|&i| f64::from(source[i][a])).sum::<f64>() / k as f64
                };
                assert!((centroid - expected).abs() < 1e-8);
            }
            assert_eq!(actual.moments.target_centroid, [0.0; 3]);
            assert_eq!(actual.moments.cross_covariance, [[0.0; 3]; 3]);
            if fraction == 1.0 {
                assert_eq!(actual.moments, all);
            }
        }
    }
}

#[test]
fn selected_centroids_and_cross_covariance_match_independent_f64_math() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [17, 63, 64, 65, 257, 4097] {
        for scale in [1e-4_f32, 1.0, 1e4] {
            let target: Vec<_> = (0..n + 17)
                .map(|i| {
                    [
                        (i % 13) as f32 * 0.5 * scale,
                        (i / 13 % 11) as f32 * 0.5 * scale,
                        (i / 143) as f32 * 0.5 * scale,
                    ]
                })
                .collect();
            let source: Vec<_> = target[..n]
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    [
                        p[0] + [0.015625, 0.03125, 0.125][i % 3] * scale,
                        p[1] - 0.015625 * scale,
                        p[2] + 0.0078125 * scale,
                    ]
                })
                .collect();
            let target_tree = tree(&target);
            let input = GpuIcpInput::upload(&runtime, &source, &target_tree).unwrap();
            // GPU distance words define ordering on this precision path. Read
            // them only in this oracle, verify known scalar membership, then
            // independently comparison-sort and accumulate in f64.
            let (neighbors, _) = input
                .tree
                .nearest_one_within_download(&runtime, &source, (0.25 * scale).powi(2))
                .unwrap();
            let mut pairs: Vec<_> = neighbors
                .iter()
                .enumerate()
                .filter_map(|(i, m)| {
                    m.map(|m| {
                        assert_eq!(m.index, i);
                        (m.distance_squared, i, m.index)
                    })
                })
                .collect();
            pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for fraction in [0.3, 0.5, 0.8] {
                let actual = input
                    .trimmed_moments_resident(
                        &runtime,
                        Isometry3::identity(),
                        0.25 * scale,
                        fraction,
                    )
                    .unwrap()
                    .download(&runtime)
                    .unwrap();
                let k = (pairs.len() as f64 * fraction).floor() as usize;
                assert_eq!(actual.moments.correspondences as usize, k);
                assert_eq!(actual.evaluated_correspondences as usize, pairs.len());
                let mean_s: [f64; 3] = std::array::from_fn(|a| {
                    pairs[..k].iter().map(|p| f64::from(source[p.1][a])).sum::<f64>() / k as f64
                });
                let mean_t: [f64; 3] = std::array::from_fn(|a| {
                    pairs[..k].iter().map(|p| f64::from(target[p.2][a])).sum::<f64>() / k as f64
                });
                for a in 0..3 {
                    assert!(
                        (actual.moments.source_centroid[a] - mean_s[a]).abs()
                            < 1e-5 * f64::from(scale)
                    );
                    assert!(
                        (actual.moments.target_centroid[a] - mean_t[a]).abs()
                            < 1e-5 * f64::from(scale)
                    );
                    for b in 0..3 {
                        let products: Vec<_> = pairs[..k]
                            .iter()
                            .map(|p| {
                                (f64::from(source[p.1][a]) - mean_s[a])
                                    * (f64::from(target[p.2][b]) - mean_t[b])
                            })
                            .collect();
                        let expected = products.iter().sum::<f64>();
                        let absolute = products.iter().map(|v| v.abs()).sum::<f64>();
                        assert!(
                            (actual.moments.cross_covariance[a][b] - expected).abs()
                                < 1e-7 * f64::from(scale).powi(2) + 1e-6 * absolute,
                            "n={n} scale={scale} fraction={fraction} a={a} b={b}"
                        );
                    }
                }
                let all_error = pairs
                    .iter()
                    .map(|p| {
                        (0..3)
                            .map(|a| f64::from(source[p.1][a] - target[p.2][a]).powi(2))
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                assert!(
                    (actual.moments.squared_error_sum - all_error).abs()
                        < 1e-7 * f64::from(scale).powi(2) + 1e-6 * all_error
                );
            }
        }
    }
}

#[test]
fn trimmed_ownership_concurrency_empty_gate_and_invalid_inputs_are_explicit() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let points: Vec<_> = (0..65)
        .map(|i| [(i % 5) as f32 * 0.5, (i / 5 % 5) as f32 * 0.5, (i / 25) as f32 * 0.5])
        .collect();
    let input = GpuIcpInput::upload(&runtime, &points, &tree(&points)).unwrap();
    for fraction in [0.0, -1.0, 1.1, f64::NAN, f64::INFINITY] {
        assert!(input
            .trimmed_moments_resident(&runtime, Isometry3::identity(), 0.25, fraction)
            .is_err());
    }
    assert!(input.trimmed_moments_resident(&other, Isometry3::identity(), 0.25, 0.5).is_err());
    let retained =
        input.trimmed_moments_resident(&runtime, Isometry3::identity(), 0.25, 0.5).unwrap();
    let expected = retained.download(&runtime).unwrap();
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        assert_eq!(
                            input
                                .trimmed_moments_resident(
                                    &runtime,
                                    Isometry3::identity(),
                                    0.25,
                                    0.5
                                )
                                .unwrap()
                                .download(&runtime)
                                .unwrap(),
                            expected
                        );
                    }
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
    });
    let far = input
        .trimmed_moments_resident(
            &runtime,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
            0.25,
            0.5,
        )
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(far.evaluated_correspondences, 0);
    assert_eq!(far.moments.source_centroid, [0.0; 3]);
    assert_eq!(far.moments.squared_error_sum, 0.0);
    let empty = GpuIcpInput::upload(&runtime, &[], &tree(&points))
        .unwrap()
        .trimmed_moments_resident(&runtime, Isometry3::identity(), 0.25, 0.5)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(empty.evaluated_correspondences, 0);
    assert_eq!(empty.moments.transfers.device_to_host_bytes(), 0);
    drop(input);
    assert_eq!(retained.download(&runtime).unwrap(), expected);
    assert!(retained.download(&other).is_err());
}

#[test]
fn trimmed_gate_overflow_and_malformed_fixed_readbacks_reject_all_payload() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let input =
        GpuIcpInput::upload(&runtime, &[[0.25, 0.0, 0.0], [0.125, 0.0, 0.0]], &tree(&[[0.0; 3]]))
            .unwrap();
    let edge = input
        .trimmed_moments_resident(&runtime, Isometry3::identity(), 0.25, 0.5)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(edge.evaluated_correspondences, 2);
    assert_eq!(edge.moments.correspondences, 1);
    assert_eq!(edge.moments.source_centroid, [0.125, 0.0, 0.0]);
    assert_eq!(edge.moments.squared_error_sum, 0.078125);
    for distance in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input
            .trimmed_moments_resident(&runtime, Isometry3::identity(), distance, 0.5)
            .is_err());
    }
    let extreme =
        GpuIcpInput::upload(&runtime, &[[f32::MAX, 0.0, 0.0]], &tree(&[[0.0; 3]])).unwrap();
    let invalid = extreme
        .trimmed_moments_resident(
            &runtime,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(f32::MAX, 0.0, 0.0)),
            0.25,
            0.5,
        )
        .unwrap();
    let words = read(&runtime, invalid.buffer(), 176);
    assert_ne!(words[0] & 1, 0);
    assert!(words[1..].iter().all(|&v| v == 0));
    assert!(invalid.download(&runtime).is_err());
    for case in 0..7 {
        let mut words = [0u32; 44];
        match case {
            0 => words[0] = 2,
            1 => {
                words[1] = 3;
                words[2] = 2;
            }
            2 => words[2] = 4,
            3 => words[3] = 1,
            4 => words[42] = 1.0f32.to_bits(),
            5 => {
                words[2] = 2;
                words[4] = 1.0f32.to_bits();
            }
            _ => {
                words[2] = 2;
                words[42] = f32::NAN.to_bits();
            }
        }
        let buffer = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let result = GpuIcpTrimmedMoments {
            inner: GpuIcpMoments {
                buffer,
                count: 3,
                scale: 1.0,
                bound: 1.0,
                identity: runtime.profiling_identity(),
            },
            trimmed: true,
        };
        assert!(result.download(&runtime).is_err(), "malformed case {case}");
    }
}

#[test]
fn explicit_f32_shader_preserves_the_pinned_previous_source() {
    assert_eq!(
        trimmed_shader(),
        include_str!("../../tests/fixtures/icp_trimmed_f32_previous.wgsl").replace("\r\n", "\n")
    );
}

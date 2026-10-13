//! Fixed-pose independent f64 oracles; copies occur before scratch recycling.
use super::*;

fn snapshot(
    runtime: &WgpuRuntime,
    input: &GpuIcpInput,
    pose: Isometry3<f32>,
    precision: GpuIcpTrimPrecision,
) -> (GpuIcpTrimmedStatistics, Vec<[f32; 4]>, Vec<[u32; 2]>, Vec<u32>) {
    let count = input.source_count();
    let groups = plan_trim(u64::from(count), &runtime.device().limits()).unwrap();
    let r = pose.rotation().to_mat3();
    let t = pose.translation();
    let params = Params {
        rows: [
            [r.m[0][0], r.m[0][1], r.m[0][2], t.x],
            [r.m[1][0], r.m[1][1], r.m[1][2], t.y],
            [r.m[2][0], r.m[2][1], r.m[2][2], t.z],
        ],
        source_anchor: input.source_anchor,
        target_anchor: input.target_anchor,
        count,
        targets: input.target.query_count(),
        groups,
        unused: 0,
        bound: 0.0625,
        max_distance: 0.25,
        scale: input.scale,
        padding: 0.0,
    };
    let result = GpuIcpTrimmedMoments {
        inner: GpuIcpMoments {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 176,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            count,
            scale: input.scale,
            bound: 0.0625,
            identity: runtime.profiling_identity(),
        },
        trimmed: true,
    };
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut scratch = ScratchBuffers::new(runtime);
    let queries = scratch.acquire("precision-query", u64::from(count) * 16);
    let matches = scratch.acquire("precision-matches", u64::from(count) * 8);
    let partials = scratch.acquire("precision-partials", u64::from(groups) * 144);
    let selection = scratch.acquire("precision-selection", 32);
    let cache = runtime.icp_trim_pipelines.get_or_init(|| TrimPipelines::new(runtime.device()));
    let reduction = cache.reduction(runtime.device(), precision);
    let entries: Vec<_> = [
        &uniform,
        input.source.buffer().unwrap(),
        input.target.buffer().unwrap(),
        &queries,
        &matches,
        &partials,
        result.buffer(),
        &selection,
    ]
    .iter()
    .enumerate()
    .map(|(binding, buffer)| wgpu::BindGroupEntry {
        binding: binding as u32,
        resource: buffer.as_entire_binding(),
    })
    .collect();
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &reduction.layout,
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    dispatch(&mut encoder, &reduction.pipelines[0], &group, count.div_ceil(64), 1);
    input.tree.encode_nearest(runtime, &mut encoder, &queries, count, 0.0625, &matches).unwrap();
    cache.encode(
        runtime,
        &mut encoder,
        &mut scratch,
        TrimParams::new(count, groups, 0.8, 0.0625, params.targets),
        &matches,
        &selection,
    );
    for (index, x, y) in [(1, groups, 7), (2, 1, 1), (3, groups, 9), (4, 1, 1)] {
        dispatch(&mut encoder, &reduction.pipelines[index], &group, x, y);
    }
    let owned = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(count) * 24 + 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&queries, 0, &owned, 0, u64::from(count) * 16);
    encoder.copy_buffer_to_buffer(&matches, 0, &owned, u64::from(count) * 16, u64::from(count) * 8);
    encoder.copy_buffer_to_buffer(&selection, 0, &owned, u64::from(count) * 24, 32);
    scratch.submit_and_recycle(encoder);
    let words = read(runtime, &owned, owned.size());
    let query_words = &words[..count as usize * 4];
    let queries = query_words
        .chunks_exact(4)
        .map(|p| std::array::from_fn(|i| f32::from_bits(p[i])))
        .collect();
    let matches = words[count as usize * 4..count as usize * 6]
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();
    let selection = words[count as usize * 6..].to_vec();
    let raw = read(runtime, result.buffer(), 176);
    if precision == GpuIcpTrimPrecision::DoubleSingle {
        assert!(raw[13..].chunks_exact(2).any(|p| p[0] != 0));
    }
    let statistics = result.download(runtime).unwrap();
    assert_eq!(
        statistics,
        input
            .trimmed_moments_resident_with_precision(runtime, pose, 0.25, 0.8, precision)
            .unwrap()
            .download(runtime)
            .unwrap()
    );
    (statistics, queries, matches, selection)
}

#[test]
fn precise_moments_match_fixed_pose_independent_f64_oracles() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let n = 262145usize;
    let target: Vec<_> = (0..n + 17)
        .map(|i| [(i % 32) as f32 * 0.5, (i / 32 % 32) as f32 * 0.5, (i / 1024) as f32 * 0.5])
        .collect();
    let source: Vec<_> = target[..n]
        .iter()
        .enumerate()
        .map(|(i, p)| {
            [
                p[0] + if i % 7 == 0 { 0.1875 } else { 0.03125 },
                p[1] - 0.015625,
                p[2] + 0.0078125 + if i % 11 == 0 { 0.375 } else { 0.0 },
            ]
        })
        .collect();
    let target_tree = tree(&target);
    let input = GpuIcpInput::upload(&runtime, &source, &target_tree).unwrap();
    let poses = [
        ("identity", Isometry3::identity()),
        (
            "cpu1",
            Isometry3::new(
                Quat::new(-2.984037e-8, -2.9834867e-11, 1.7670175e-12, 1.0),
                Vec3::new(-0.031249996, 0.015621266, -0.004607767),
            ),
        ),
        (
            "cpu2",
            Isometry3::new(
                Quat::new(-9.731036e-7, -6.82904e-10, -5.5494216e-13, 1.0),
                Vec3::new(-0.031249914, 0.015503033, -0.0045932755),
            ),
        ),
    ];
    let mut maximum_mean_error = 0.0_f64;
    let mut maximum_covariance_error = 0.0_f64;
    for (label, pose) in poses {
        let mut previous = None;
        for precision in [GpuIcpTrimPrecision::CompensatedF32, GpuIcpTrimPrecision::DoubleSingle] {
            let (actual, queries, matches, selection) = snapshot(&runtime, &input, pose, precision);
            let mut gpu_pairs: Vec<_> = matches
                .iter()
                .enumerate()
                .filter(|(_, m)| m[0] != u32::MAX)
                .map(|(i, m)| (m[1] & 0x7fffffff, i, m[0] as usize))
                .collect();
            gpu_pairs.sort_unstable();
            let k = (gpu_pairs.len() as f64 * 0.8).floor() as usize;
            assert_eq!(selection[1] as usize, k);
            assert_eq!(
                (selection[3], selection[4] as usize),
                (gpu_pairs[k - 1].0, gpu_pairs[k - 1].1)
            );
            let mean_s: [f64; 3] = std::array::from_fn(|a| {
                gpu_pairs[..k].iter().map(|p| f64::from(queries[p.1][a])).sum::<f64>() / k as f64
            });
            let mean_t: [f64; 3] = std::array::from_fn(|a| {
                gpu_pairs[..k].iter().map(|p| f64::from(target[p.2][a])).sum::<f64>() / k as f64
            });
            let h: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    gpu_pairs[..k]
                        .iter()
                        .map(|p| {
                            (f64::from(queries[p.1][a]) - mean_s[a])
                                * (f64::from(target[p.2][b]) - mean_t[b])
                        })
                        .sum()
                })
            });
            let mean_error = (0..3)
                .flat_map(|a| {
                    [
                        (actual.moments.source_centroid[a] - mean_s[a]).abs(),
                        (actual.moments.target_centroid[a] - mean_t[a]).abs(),
                    ]
                })
                .fold(0.0, f64::max);
            let covariance_error = (0..3)
                .flat_map(|a| {
                    (0..3).map(move |b| (actual.moments.cross_covariance[a][b] - h[a][b]).abs())
                })
                .fold(0.0, f64::max);
            if let Some((old_queries, old_matches, old_selection, old_error)) = &previous {
                assert_eq!(&queries, old_queries);
                assert_eq!(&matches, old_matches);
                assert_eq!(&selection, old_selection);
                assert_eq!(&actual.moments.squared_error_sum, old_error);
            } else {
                previous = Some((
                    queries.clone(),
                    matches.clone(),
                    selection.clone(),
                    actual.moments.squared_error_sum,
                ));
            }
            if precision == GpuIcpTrimPrecision::DoubleSingle {
                assert!(mean_error < 1e-9, "label={label} mean_error={mean_error}");
                for a in 0..3 {
                    for b in 0..3 {
                        let absolute = gpu_pairs[..k]
                            .iter()
                            .map(|p| {
                                ((f64::from(queries[p.1][a]) - mean_s[a])
                                    * (f64::from(target[p.2][b]) - mean_t[b]))
                                    .abs()
                            })
                            .sum::<f64>();
                        assert!(
                            (actual.moments.cross_covariance[a][b] - h[a][b]).abs()
                                < 1e-9 + 5e-12 * absolute,
                            "label={label} a={a} b={b}"
                        );
                    }
                }
                maximum_mean_error = maximum_mean_error.max(mean_error);
                maximum_covariance_error = maximum_covariance_error.max(covariance_error);
            }
        }
    }
    eprintln!("GPU_ICP_PRECISE_ORACLES poses=3 source_points=262145 maximum_mean_error={maximum_mean_error} maximum_covariance_error={maximum_covariance_error}");
}

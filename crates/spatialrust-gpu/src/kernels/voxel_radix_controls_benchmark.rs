use super::*;
use std::{hint::black_box, time::Instant};

struct Work {
    source: wgpu::Buffer,
    entries: [wgpu::Buffer; 2],
    histograms: wgpu::Buffer,
    offsets: wgpu::Buffer,
    ranks: wgpu::Buffer,
}

fn control_groups(r: &WgpuRuntime, work: &Work, n: u32, current: bool) -> Vec<wgpu::BindGroup> {
    let groups = n.div_ceil(256);
    let batch = if current { RadixControls::new(r.device(), n, groups) } else { None };
    let pipelines = &r.pipelines().voxel_radix;
    let mut bindings = Vec::with_capacity(24);
    for digit in 0..24u32 {
        let params = super::super::super::RadixParams {
            point_count: n,
            group_count: groups,
            axis: 2 - digit / 8,
            shift: digit % 8 * 4,
        };
        let uniform = batch.is_none().then(|| {
            r.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel-radix-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });
        let control = match &batch {
            Some(batch) => batch.binding(params.axis, params.shift),
            None => uniform.as_ref().unwrap().as_entire_binding(),
        };
        let input = if digit == 0 { &work.source } else { &work.entries[digit as usize % 2] };
        let output = &work.entries[(digit as usize + 1) % 2];
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: control }];
        entries.extend(
            [input, output, &work.histograms, &work.offsets, &work.ranks]
                .into_iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: b.as_entire_binding(),
                }),
        );
        bindings.push(r.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-radix-bind-group"),
            layout: &pipelines.bind_group_layout,
            entries: &entries,
        }));
    }
    bindings
}

fn measure(r: &WgpuRuntime, n: usize, pattern: usize) {
    let data = keys(n, pattern);
    let coords = cpu_keys(&data);
    let oracle = expected(&data);
    let work = Work {
        source: buffer(r, bytemuck::cast_slice(&data)),
        entries: std::array::from_fn(|_| buffer(r, &vec![0; n * 4])),
        histograms: buffer(r, &vec![0; n.div_ceil(256) * 16]),
        offsets: buffer(r, &vec![0; n.div_ceil(256) * 16]),
        ranks: buffer(r, &vec![0; n]),
    };
    let mut samples: [[Vec<f64>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        for route_offset in 0..3 {
            let route = (round + route_offset) % 3;
            for offset in 0..2 {
                let current = (round + offset) % 2 == 1;
                r.wait_idle();
                let elapsed = if route == 0 {
                    let mut owned = Vec::with_capacity(4);
                    let start = Instant::now();
                    for _ in 0..4 {
                        owned.push(control_groups(
                            black_box(r),
                            black_box(&work),
                            n as u32,
                            current,
                        ));
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000. / 4.;
                    assert!(owned.iter().all(|groups| groups.len() == 24));
                    elapsed
                } else if route == 1 {
                    let start = Instant::now();
                    let output = if current {
                        build_voxel_segments_gpu_from_keys_buffer(
                            black_box(r),
                            black_box(&work.source),
                            n as u32,
                            n as u32,
                        )
                    } else {
                        old_sort::build_voxel_segments_gpu_from_keys_buffer(
                            black_box(r),
                            black_box(&work.source),
                            n as u32,
                            n as u32,
                        )
                    }
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(output.to_voxel_segments(r).unwrap(), oracle);
                    assert_eq!(output.construction_readback_bytes(), 4);
                    elapsed
                } else {
                    let start = Instant::now();
                    let output = if current {
                        build_voxel_segments_gpu(black_box(r), black_box(&coords))
                    } else {
                        old_sort::build_voxel_segments_gpu(black_box(r), black_box(&coords))
                    }
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(output, oracle);
                    elapsed
                };
                if round >= 3 {
                    samples[route][usize::from(current)].push(elapsed);
                }
            }
        }
    }
    println!("RADIX_CONTROLS_RECORD {{\"count\":{n},\"pattern\":{pattern},\"setup_batch\":4,\"setup_old_ms\":{:?},\"setup_ms\":{:?},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",samples[0][0],samples[0][1],samples[1][0],samples[1][1],samples[2][0],samples[2][1]);
}

#[test]
#[ignore = "paired control setup/resident GPU-key/full CPU-key calls; run alone in release mode"]
fn benchmark_complete_radix_controls_calls() {
    let r = WgpuRuntime::new_headless().unwrap();
    let plan = layout(&r.device().limits()).unwrap();
    println!("RADIX_CONTROLS_ENV backend={} device_type={} available_workers={} uniform_stride={} uniform_bytes={} warmups=3 pairs=30 pair_order=alternating route_order=rotating setup_batch=4 checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()),plan.stride,plan.bytes);
    for (n, pattern) in [
        (2, 0),
        (256, 2),
        (257, 2),
        (65_537, 0),
        (65_537, 1),
        (524_288, 0),
        (524_289, 0),
        (1_000_003, 0),
        (1_000_003, 1),
        (1_000_003, 2),
    ] {
        measure(&r, n, pattern);
    }
}

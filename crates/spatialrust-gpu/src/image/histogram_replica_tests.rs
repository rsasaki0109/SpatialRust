use super::*;
use spatialrust_image::{Image, ImageRegion, ImageView};

pub(super) const FROZEN: &str = include_str!("../../tests/fixtures/histogram_previous.wgsl");
pub(super) const REPLICATED: &str = include_str!("histogram/replicated.wgsl");
pub(super) const RUNS: &str = include_str!("../../tests/fixtures/histogram_runs_experiment.wgsl");

pub(super) fn fixture(width: usize, height: usize, pattern: usize) -> (Image<u8, 1>, [u32; 256]) {
    let data: Vec<_> = (0..width * height)
        .map(|i| match pattern {
            0 => ((i * 37 + i / 257) % 256) as u8,
            1 => 73,
            2 => {
                if (i % width + i / width) % 2 == 0 {
                    0
                } else {
                    255
                }
            }
            _ => unreachable!(),
        })
        .collect();
    let mut bins = [0; 256];
    for &value in &data {
        bins[value as usize] += 1;
    }
    (Image::try_new(width, height, data).unwrap(), bins)
}

fn check_channels<const C: usize>(
    runtime: &WgpuRuntime,
    frozen: &HistogramPipelines,
    replicated: &HistogramPipelines,
    runs: &HistogramPipelines,
) -> usize {
    let mut cases = 0;
    for (width, height) in [
        (1, 1),
        (255, 1),
        (256, 1),
        (257, 1),
        (1, 257),
        (255, 257),
        (256, 256),
        (257, 257),
        (513, 517),
        (1001, 1003),
    ] {
        for pattern in 0..3 {
            let parent_w = width + 5;
            let stride = parent_w * C + 11;
            let data: Vec<_> = (0..(height + 7) * stride)
                .map(|i| match pattern {
                    0 => ((i * 37 + i / 257) % 256) as u8,
                    1 => 73,
                    _ => {
                        if i % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                })
                .collect();
            let view = ImageView::<u8, C>::new(parent_w, height + 7, stride, &data)
                .unwrap()
                .subview(ImageRegion::new(2, 3, width, height))
                .unwrap();
            let source = GpuImage::upload_u8(runtime, view).unwrap();
            for channel in 0..C {
                let mut expected = [0; 256];
                for y in 0..height {
                    for x in 0..width {
                        expected[view.get(x, y).unwrap()[channel] as usize] += 1;
                    }
                }
                let mut outputs = Vec::new();
                for pipelines in [frozen, replicated, runs] {
                    let mut output = experiment::forced_histogram(
                        runtime,
                        &source,
                        channel as u32,
                        None,
                        pipelines,
                    )
                    .unwrap();
                    assert_eq!(output.pixel_count(), (width * height) as u32);
                    assert_eq!(output.readback_u32(runtime).unwrap(), expected);
                    outputs.push(output);
                }
                assert_eq!(outputs[0].receipt(), outputs[1].receipt());
                assert_eq!(outputs[0].receipt(), outputs[2].receipt());
                let mut public = histogram_u8_gpu(runtime, &source, channel as u32).unwrap();
                assert_eq!(public.readback_u32(runtime).unwrap(), expected);
                assert_eq!(outputs[0].receipt(), public.receipt());
                public.recycle(runtime);
                for output in outputs {
                    output.recycle(runtime);
                }
                cases += 1;
            }
            source.recycle(runtime);
        }
    }
    cases
}

#[test]
fn replicated_histograms_match_frozen_and_cpu_counts_for_all_channels_and_rois() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let frozen = experiment::pipelines(runtime.device(), FROZEN);
    let replicated = experiment::pipelines(runtime.device(), REPLICATED);
    let runs = experiment::pipelines(runtime.device(), RUNS);
    let cases = check_channels::<1>(&runtime, &frozen, &replicated, &runs)
        + check_channels::<2>(&runtime, &frozen, &replicated, &runs)
        + check_channels::<3>(&runtime, &frozen, &replicated, &runs)
        + check_channels::<4>(&runtime, &frozen, &replicated, &runs);
    assert_eq!(cases, 300);
    eprintln!(
        "Verified {cases} frozen/CPU histogram cases and {} exact bin comparisons.",
        cases * 256 * 4
    );
}

#[test]
fn selection_respects_exact_pixel_and_shared_memory_boundaries() {
    let mut limits = wgpu::Limits::default();
    for (w, h, selected) in [
        (4095, 1024, false),
        (4096, 1024, true),
        (2048, 2048, true),
        (8192, 513, true),
        (8192, 1024, true),
        (8192, 1025, false),
        (4096, 4096, false),
    ] {
        assert_eq!(use_replicated(&dispatch_plan(w, h, &limits).unwrap(), &limits), selected);
    }
    limits.max_compute_workgroup_storage_size = REPLICATED_WORKGROUP_BYTES;
    assert!(use_replicated(&dispatch_plan(2048, 2048, &limits).unwrap(), &limits));
    limits.max_compute_workgroup_storage_size -= 1;
    assert!(!use_replicated(&dispatch_plan(2048, 2048, &limits).unwrap(), &limits));
    limits.max_compute_workgroup_storage_size = 1024;
    assert!(!use_replicated(&dispatch_plan(2048, 2048, &limits).unwrap(), &limits));
    limits.max_compute_workgroup_storage_size = 1023;
    assert!(dispatch_plan(2048, 2048, &limits).is_err());
}

#[test]
fn large_public_histograms_keep_two_passes_and_concurrent_lazy_cache() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    let tiny = GpuImage::upload_u8(&runtime, fixture(17, 19, 1).0.view()).unwrap();
    histogram_u8_gpu(&runtime, &tiny, 0).unwrap().recycle(&runtime);
    let cache = runtime.image_histogram_pipelines.get().unwrap();
    assert!(cache.replicated.get().is_none());
    let (large_image, _) = fixture(2048, 2048, 1);
    let rejected_source = GpuImage::upload_u8(&runtime, large_image.view()).unwrap();
    let mut insufficient = GpuStageProfiler::new(&runtime, 1).unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    assert!(histogram_u8_gpu_profiled(&runtime, &rejected_source, 0, &mut insufficient).is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    assert!(cache.replicated.get().is_none());
    assert!(insufficient.collect(&runtime).unwrap().stages.is_empty());
    rejected_source.recycle(&runtime);
    tiny.recycle(&runtime);
    let frozen = experiment::pipelines(runtime.device(), FROZEN);
    for pattern in 0..3 {
        let (image, expected) = fixture(2048, 2048, pattern);
        let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
        let outputs = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| histogram_u8_gpu(&runtime, &source, 0).unwrap()))
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(cache.replicated.get().is_some());
        let mut profiler = GpuStageProfiler::new(&runtime, 2).unwrap();
        let mut output = histogram_u8_gpu_profiled(&runtime, &source, 0, &mut profiler).unwrap();
        let mut parent = experiment::forced_histogram(&runtime, &source, 0, None, &frozen).unwrap();
        assert_eq!(output.receipt(), parent.receipt());
        assert_eq!(output.readback_u32(&runtime).unwrap(), expected);
        assert_eq!(parent.readback_u32(&runtime).unwrap(), expected);
        let report = profiler.collect(&runtime).unwrap();
        assert_eq!(report.stages.len(), 2);
        assert_eq!(report.device_to_host_bytes, 32);
        source.recycle(&runtime);
        for mut live in outputs {
            assert_eq!(live.readback_u32(&runtime).unwrap(), expected);
            live.recycle(&runtime);
        }
        output.recycle(&runtime);
        parent.recycle(&runtime);
    }
    let data: Vec<_> = (0..2048 * 2048 * 4).map(|i| ((i * 37 + i / 257) % 256) as u8).collect();
    let mut bins = [[0; 256]; 4];
    for pixel in data.chunks_exact(4) {
        for (channel, &value) in pixel.iter().enumerate() {
            bins[channel][value as usize] += 1;
        }
    }
    let image = Image::<u8, 4>::try_new(2048, 2048, data).unwrap();
    let source = GpuImage::upload_u8(&runtime, image.view()).unwrap();
    for (channel, expected) in bins.iter().enumerate() {
        let mut public = histogram_u8_gpu(&runtime, &source, channel as u32).unwrap();
        let mut old =
            experiment::forced_histogram(&runtime, &source, channel as u32, None, &frozen).unwrap();
        assert_eq!(public.readback_u32(&runtime).unwrap(), *expected);
        assert_eq!(old.readback_u32(&runtime).unwrap(), *expected);
        assert_eq!(public.receipt(), old.receipt());
        public.recycle(&runtime);
        old.recycle(&runtime);
    }
    source.recycle(&runtime);
}

#[test]
fn replica_merge_preserves_maximum_counter_mass() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut counts = vec![0u32; 8 * 256];
    for bin in 0..256 {
        let remainder = (bin * 37) as u32;
        counts[(bin % 8) * 256 + bin] = u32::MAX - remainder;
        counts[((bin + 1) % 8) * 256 + bin] = remainder;
    }
    let probe = format!(
        "{REPLICATED}\n{}",
        r#"
@group(0) @binding(4) var<storage, read> probe_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> probe_output: array<u32>;
@compute @workgroup_size(256)
fn merge_probe(@builtin(local_invocation_index) lane: u32) {
    for (var replica = 0u; replica < 8u; replica++) {
        atomicStore(&local_bins[replica * 257u + lane], probe_counts[replica * 256u + lane]);
    }
    workgroupBarrier();
    probe_output[lane] = sum_replicas(lane);
}"#
    );
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(probe.into()),
    });
    let pipeline = runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("merge_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&counts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: HISTOGRAM_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bindings = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 4, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder =
        runtime.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    assert_eq!(readback_bins_u32(&runtime, &output).unwrap(), [u32::MAX; 256]);
}
